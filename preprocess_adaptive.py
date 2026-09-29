"""
Pretraitement ADAPTATIF pour SpatialLM (version experimentale).

N'altere PAS preprocess_for_spatiallm.py : c'est un fichier a part, pour comparer.

Principe : mesurer -> decider -> corriger le minimum.
Chaque etape mesure un indicateur sur le nuage courant. Si l'indicateur est sous
la tolerance, l'etape est ignoree (le nuage n'est pas touche) et le journal le dit.
Un fichier deja propre, aligne et a l'echelle ressort donc (quasi) inchange.

Difference avec la version actuelle :
  - outliers : mediane + MAD sur log(distance aux voisins), ET point vraiment isole
    (distance > 3 x mediane), avec regle d'arret (rien si < 0.5 % de points signales)
  - verticale : normales des surfaces (aucun RANSAC, donc deterministe), ignoree < 0.5 deg
  - cap des murs : histogramme circulaire des normales murales modulo 90 deg,
    angle dans [-45, +45], ignore < 0.5 deg ou si la piece n'est pas "Manhattan"
  - echelle : conservee si la hauteur (percentiles 0.5-99.5 %) est plausible en metres,
    conversion mm/cm si detectee, sinon mise a l'echelle cible
  - toutes les transformations sont composees dans une matrice 4x4 enregistree
    dans <sortie>.transform.json (pour relier le layout/IFC au repere d'origine)

Usage :
    python preprocess_adaptive.py -i scan.ply -o clean.ply
    python preprocess_adaptive.py -i scan.ply -o clean.ply --dry_run      # decisions seulement
    python preprocess_adaptive.py -i scan.ply -o clean.ply --force        # tout appliquer
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

SEED = 42
Z = np.array([0.0, 0.0, 1.0])

DEFAULT_TOL = dict(
    outlier_frac=0.005,    # en dessous : etape debruitage ignoree
    tilt_deg=0.5,          # en dessous : verticale ignoree
    yaw_deg=0.5,           # en dessous : cap ignore
    yaw_conf=0.25,         # concentration minimale des directions murales (0..1)
    ground_z=0.05,         # en dessous : pas de translation du sol vers z=0
    height_min=2.0,        # hauteur plausible d'une piece (m)
    height_max=6.0,
    color_tol=0.02,
)


# ---------------------------------------------------------------------------
# Outils geometriques
# ---------------------------------------------------------------------------

def rot_between(a, b):
    """Rotation 3x3 qui envoie le vecteur unitaire a sur b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_about(R, center):
    """Matrice 4x4 : rotation R autour du point center."""
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = center - R @ center
    return M


class Cloud:
    """Nuage courant + matrice composee + indices d'origine des points conserves."""

    def __init__(self, pts, cols=None):
        self.pts = np.asarray(pts, dtype=np.float64).copy()
        self.cols = None if cols is None else np.asarray(cols, dtype=np.float64).copy()
        self.idx = np.arange(len(self.pts))
        self.M = np.eye(4)

    def keep(self, mask):
        self.pts = self.pts[mask]
        if self.cols is not None:
            self.cols = self.cols[mask]
        self.idx = self.idx[mask]

    def transform(self, M):
        self.pts = self.pts @ M[:3, :3].T + M[:3, 3]
        self.M = M @ self.M


class Journal:
    def __init__(self, verbose=True):
        self.rows = []
        self.verbose = verbose

    def add(self, step, action, measured, note=""):
        self.rows.append(dict(step=step, action=action, measured=measured, note=note))
        if self.verbose:
            print(f"  [{step:<10}] {action:<8} {measured}" + (f"  ({note})" if note else ""))


# ---------------------------------------------------------------------------
# Mesures (diagnostic)
# ---------------------------------------------------------------------------

def estimate_spacing(points, tree, n=20000):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(points), min(n, len(points)), replace=False)
    d, _ = tree.query(points[idx], k=2, workers=-1)
    return float(np.median(d[:, 1]))


def knn_mean_dist(points, tree, k=16, chunk=500_000):
    out = np.empty(len(points))
    for s in range(0, len(points), chunk):
        d, _ = tree.query(points[s:s + chunk], k=k + 1, workers=-1)
        out[s:s + chunk] = d[:, 1:].mean(axis=1)
    return out


def flag_outliers(dbar, z_thresh=3.5, iso_factor=3.0):
    """Point aberrant = statistiquement eloigne (z robuste sur log d) ET vraiment isole."""
    ld = np.log(np.maximum(dbar, 1e-12))
    med = np.median(ld)
    mad = np.median(np.abs(ld - med)) * 1.4826 + 1e-12
    z = (ld - med) / mad
    return (z > z_thresh) & (dbar > iso_factor * np.median(dbar))


def sample_normals(points, spacing, max_pts=300_000):
    """Sous-echantillon a densite uniforme (voxel) + normales."""
    voxel = 4.0 * spacing
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    d = p
    for _ in range(10):
        d = p.voxel_down_sample(voxel)
        if len(d.points) <= max_pts:
            break
        voxel *= 1.5
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.5 * voxel, max_nn=40))
    return np.asarray(d.points), np.asarray(d.normals)


def refine_axis(normals, v, iters=3, cos_thr=0.96):
    for _ in range(iters):
        sel = np.abs(normals @ v) > cos_thr
        if sel.sum() < 50:
            break
        _, V = np.linalg.eigh(normals[sel].T @ normals[sel])
        nv = V[:, -1]
        if nv @ v < 0:
            nv = -nv
        v = nv
    return v / np.linalg.norm(v)


def find_vertical(points, normals):
    """Axe vertical (du sol vers le plafond) a partir des normales. Retourne (v, incertain)."""
    _, V = np.linalg.eigh(normals.T @ normals)
    cand = V[:, int(np.argmax(np.abs(V.T @ Z)))]
    if abs(cand @ Z) > np.cos(np.radians(30)):
        v = cand if cand @ Z > 0 else -cand
        return refine_axis(normals, v), False
    # scan couche : la verticale est l'axe de plus faible etendue, le sol l'extremite la plus dense
    ext = [np.subtract(*np.percentile(points @ V[:, i], [99, 1])) for i in range(3)]
    v = V[:, int(np.argmin(ext))]
    proj = points @ v
    lo, hi = np.percentile(proj, [1, 99])
    band = 0.05 * (hi - lo)
    if (proj > hi - band).sum() > (proj < lo + band).sum():
        v = -v
    return refine_axis(normals, v), True


def yaw_offset(normals):
    """Ecart de cap (deg, dans [-45, 45]) des murs par rapport aux axes X/Y, et concentration."""
    sel = np.abs(normals[:, 2]) < 0.2
    if sel.sum() < 200:
        return None, 0.0
    th = np.arctan2(normals[sel, 1], normals[sel, 0])
    z4 = np.mean(np.exp(4j * th))
    return float(np.degrees(np.angle(z4)) / 4.0), float(np.abs(z4))


# ---------------------------------------------------------------------------
# Etapes
# ---------------------------------------------------------------------------

def step_outliers(c, tol, J, force=False, k=16):
    tree = cKDTree(c.pts, balanced_tree=False, compact_nodes=False)
    dbar = knn_mean_dist(c.pts, tree, k)
    flags = flag_outliers(dbar)
    frac = float(flags.mean())
    if frac < tol["outlier_frac"] and not force:
        J.add("outliers", "ignore", f"{100 * frac:.2f} % de points isoles",
              f"< {100 * tol['outlier_frac']:.1f} %")
        return
    n0 = len(c.pts)
    c.keep(~flags)
    J.add("outliers", "applique", f"{n0:,} -> {len(c.pts):,} pts (-{100 * frac:.2f} %)")


def step_vertical(c, spacing, tol, J, force=False):
    pts, nrm = sample_normals(c.pts, spacing)
    v, uncertain = find_vertical(pts, nrm)
    tilt = float(np.degrees(np.arccos(np.clip(v @ Z, -1, 1))))
    note = "sens haut/bas suppose (scan couche)" if uncertain else ""
    if tilt < tol["tilt_deg"] and not force:
        J.add("verticale", "ignore", f"inclinaison {tilt:.2f} deg", f"< {tol['tilt_deg']} deg")
        return
    R = rot_between(v, Z)
    c.transform(rotation_about(R, c.pts.mean(axis=0)))
    J.add("verticale", "applique", f"inclinaison {tilt:.2f} deg corrigee", note)


def step_yaw(c, spacing, tol, J, force=False):
    pts, nrm = sample_normals(c.pts, spacing)
    delta, conf = yaw_offset(nrm)
    if delta is None:
        J.add("cap", "ignore", "pas assez de surfaces murales")
        return
    if conf < tol["yaw_conf"] and not force:
        J.add("cap", "ignore", f"piece peu 'Manhattan' (concentration {conf:.2f})",
              f"< {tol['yaw_conf']}")
        return
    if abs(delta) < tol["yaw_deg"] and not force:
        J.add("cap", "ignore", f"ecart de cap {delta:+.2f} deg", f"< {tol['yaw_deg']} deg")
        return
    c.transform(rotation_about(rot_z(-np.radians(delta)), c.pts.mean(axis=0)))
    J.add("cap", "applique", f"ecart de cap {delta:+.2f} deg corrige (concentration {conf:.2f})")


def step_scale(c, tol, J, target_height=2.5, force_scale=False):
    lo, hi = np.percentile(c.pts[:, 2], [0.5, 99.5])
    h = float(hi - lo)
    if h < 1e-9:
        J.add("echelle", "ignore", "hauteur nulle")
        return
    if force_scale:
        s = target_height / h
        note = "force"
    elif tol["height_min"] <= h <= tol["height_max"]:
        J.add("echelle", "ignore", f"hauteur {h:.2f} m plausible en metres")
        return
    elif 200 <= h <= 600:
        s, note = 0.01, "unite detectee : cm"
    elif 2000 <= h <= 6000:
        s, note = 0.001, "unite detectee : mm"
    else:
        s, note = target_height / h, "echelle inconnue : mise a la hauteur cible"
    M = np.diag([s, s, s, 1.0])
    c.transform(M)
    J.add("echelle", "applique", f"hauteur {h:.3g} -> {h * s:.2f} m (x{s:.4g})", note)


def step_ground(c, tol, J, force=False):
    z0 = float(np.percentile(c.pts[:, 2], 0.5))
    if abs(z0) < tol["ground_z"] and not force:
        J.add("sol z=0", "ignore", f"sol a z={z0:+.3f} m", f"< {tol['ground_z']} m")
        return
    M = np.eye(4)
    M[2, 3] = -z0
    c.transform(M)
    J.add("sol z=0", "applique", f"sol ramene de z={z0:+.3f} m a 0")


def step_dbscan(c, J):
    if len(c.pts) > 3_000_000:
        J.add("dbscan", "ignore", "nuage trop gros pour DBSCAN")
        return
    tree = cKDTree(c.pts, balanced_tree=False, compact_nodes=False)
    eps = 3.0 * estimate_spacing(c.pts, tree)
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(c.pts)
    labels = np.array(p.cluster_dbscan(eps=eps, min_points=20, print_progress=False))
    if labels.max() < 0:
        J.add("dbscan", "ignore", "aucun cluster")
        return
    counts = np.bincount(labels[labels >= 0])
    keep = labels == int(np.argmax(counts))
    if keep.all():
        J.add("dbscan", "ignore", f"un seul cluster (eps={eps:.3g})")
        return
    n0 = len(c.pts)
    c.keep(keep)
    J.add("dbscan", "applique", f"{n0:,} -> {len(c.pts):,} pts (eps={eps:.3g})")


def step_colors(c, tol, J, force=False):
    if c.cols is None:
        return
    tm, ts = np.array([0.62, 0.60, 0.57]), np.array([0.20, 0.21, 0.21])
    m, s = c.cols.mean(axis=0), c.cols.std(axis=0)
    if np.abs(m - tm).max() < tol["color_tol"] and np.abs(s - ts).max() < tol["color_tol"] and not force:
        J.add("couleurs", "ignore", "statistiques deja conformes")
        return
    s = np.where(s < 1e-6, 1.0, s)
    c.cols = np.clip((c.cols - m) / s * ts + tm, 0, 1)
    J.add("couleurs", "applique", "normalisees")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(points, colors=None, tol=None, force=False, target_height=2.5, force_scale=False,
        no_denoise=False, no_align=False, no_scale=False, keep_largest_cluster=False,
        voxel_size=0.0, normalize_colors=False, verbose=True):
    """Applique le pretraitement adaptatif. Retourne (Cloud, Journal)."""
    tol = {**DEFAULT_TOL, **(tol or {})}
    c = Cloud(points, colors)
    J = Journal(verbose)
    tree = cKDTree(c.pts, balanced_tree=False, compact_nodes=False)
    spacing = estimate_spacing(c.pts, tree)
    del tree
    J.add("espacement", "mesure", f"{spacing:.4g} (unite du scan)")

    if not no_denoise:
        step_outliers(c, tol, J, force)
    if keep_largest_cluster and not no_denoise:
        step_dbscan(c, J)
    if not no_align:
        step_vertical(c, spacing, tol, J, force)
        step_yaw(c, spacing, tol, J, force)
    if not no_scale:
        step_scale(c, tol, J, target_height, force_scale or force)
    if not no_align:
        step_ground(c, tol, J, force)
    if voxel_size > 0:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(c.pts)
        n0 = len(c.pts)
        p = p.voxel_down_sample(voxel_size)
        # les indices d'origine ne sont plus suivis apres un voxel
        c.pts = np.asarray(p.points)
        c.cols = None
        c.idx = None
        J.add("voxel", "applique", f"{n0:,} -> {len(c.pts):,} pts")
    if normalize_colors:
        step_colors(c, tol, J, force)
    return c, J


def matrix_summary(M):
    """Rotation (inclinaison, lacet) et facteur d'echelle d'une matrice 4x4."""
    A = M[:3, :3]
    s = float(np.cbrt(np.linalg.det(A)))
    R = A / s
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return dict(scale=s, tilt_deg=tilt, yaw_deg=yaw)


def process_one(in_path, out_path, args):
    print(f"\n[{os.path.basename(in_path)}]")
    pcd = o3d.io.read_point_cloud(in_path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    n_in = len(pts)
    print(f"  Charge : {n_in:,} points | couleurs={cols is not None}")
    tol = dict(outlier_frac=args.tol_outlier, tilt_deg=args.tol_tilt, yaw_deg=args.tol_yaw)
    c, J = run(pts, cols, tol=tol, force=args.force, target_height=args.target_height,
               force_scale=args.force_scale, no_denoise=args.no_denoise, no_align=args.no_align,
               no_scale=args.no_scale, keep_largest_cluster=args.keep_largest_cluster,
               voxel_size=args.voxel_size, normalize_colors=args.normalize_colors)
    summ = matrix_summary(c.M)
    print(f"  Resultat : {len(c.pts):,} pts ({100 * len(c.pts) / n_in:.1f} %) | "
          f"rotation totale : inclinaison {summ['tilt_deg']:.2f} deg, lacet {summ['yaw_deg']:+.2f} deg | "
          f"echelle x{summ['scale']:.4f}")
    if args.dry_run:
        print("  (dry_run : aucun fichier ecrit)")
        return
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(c.pts)
    if c.cols is not None:
        out.colors = o3d.utility.Vector3dVector(c.cols)
    elif cols is not None and c.idx is not None:
        out.colors = o3d.utility.Vector3dVector(cols[c.idx])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    o3d.io.write_point_cloud(out_path, out)
    meta = dict(input=os.path.abspath(in_path), n_in=n_in, n_out=len(c.pts),
                transform_4x4=c.M.tolist(), summary=summ, journal=J.rows)
    with open(os.path.splitext(out_path)[0] + ".transform.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Sauvegarde -> {out_path} (+ .transform.json)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help=".ply ou dossier de .ply")
    p.add_argument("-o", "--output", required=True, help=".ply ou dossier de sortie")
    p.add_argument("--dry_run", action="store_true", help="afficher les decisions sans ecrire")
    p.add_argument("--force", action="store_true", help="appliquer toutes les etapes quoi qu'il arrive")
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--no_align", action="store_true")
    p.add_argument("--no_scale", action="store_true")
    p.add_argument("--keep_largest_cluster", action="store_true")
    p.add_argument("--voxel_size", type=float, default=0.0)
    p.add_argument("--normalize_colors", action="store_true")
    p.add_argument("--target_height", type=float, default=2.5)
    p.add_argument("--force_scale", action="store_true", help="toujours ramener a --target_height")
    p.add_argument("--tol_outlier", type=float, default=DEFAULT_TOL["outlier_frac"])
    p.add_argument("--tol_tilt", type=float, default=DEFAULT_TOL["tilt_deg"])
    p.add_argument("--tol_yaw", type=float, default=DEFAULT_TOL["yaw_deg"])
    return p


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if os.path.isfile(args.input):
        process_one(args.input, args.output, args)
    else:
        for f in sorted(glob.glob(os.path.join(args.input, "*.ply"))):
            process_one(f, os.path.join(args.output, os.path.basename(f)), args)
    print("\nDone.")
