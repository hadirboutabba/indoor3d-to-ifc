"""
Tests du pretraitement adaptatif (preprocess_adaptive.py) sur une piece SYNTHETIQUE
dont l'inclinaison, le cap, l'echelle et les outliers sont connus, puis comparaison
avec le pretraitement actuel (preprocess_for_spatiallm.py) sur des fichiers reels.

Usage :
    python test_preprocess_adaptive.py                      # tests synthetiques
    python test_preprocess_adaptive.py --real a.ply b.ply   # + fichiers reels (adaptatif vs actuel)
    pytest test_preprocess_adaptive.py                      # (si pytest est installe)

Aucun fichier existant n'est modifie ni ecrit.
"""

import sys
import argparse
import contextlib
import io
import numpy as np
import open3d as o3d

import preprocess_adaptive as A

TOL_REMOVED = 0.005
TOL_ANGLE = 0.5
TOL_SCALE = 0.005


# ---------------------------------------------------------------------------
# Piece synthetique
# ---------------------------------------------------------------------------

def _rect(rng, origin, u, v, n):
    return (np.asarray(origin) + rng.random((n, 1)) * np.asarray(u)
            + rng.random((n, 1)) * np.asarray(v))


def _box(rng, o, size, density):
    ox, oy, oz = o
    sx, sy, sz = size
    faces = [
        ((ox, oy, oz + sz), (sx, 0, 0), (0, sy, 0)),      # dessus
        ((ox, oy, oz), (sx, 0, 0), (0, 0, sz)),            # cotes
        ((ox, oy + sy, oz), (sx, 0, 0), (0, 0, sz)),
        ((ox, oy, oz), (0, sy, 0), (0, 0, sz)),
        ((ox + sx, oy, oz), (0, sy, 0), (0, 0, sz)),
    ]
    out = []
    for org, u, v in faces:
        area = np.linalg.norm(u) * np.linalg.norm(v)
        out.append(_rect(rng, org, u, v, max(int(area * density), 10)))
    return np.vstack(out)


def make_room(seed=0, L=6.0, W=4.0, H=2.6, density=3000, noise=0.003, outlier_frac=0.0):
    """Retourne (points, couleurs, est_outlier). Piece alignee sur X/Y, sol a z=0."""
    rng = np.random.default_rng(seed)
    parts = [
        _rect(rng, (0, 0, 0), (L, 0, 0), (0, W, 0), int(L * W * density)),      # sol
        _rect(rng, (0, 0, H), (L, 0, 0), (0, W, 0), int(L * W * density)),      # plafond
        _rect(rng, (0, 0, 0), (0, W, 0), (0, 0, H), int(W * H * density)),      # murs
        _rect(rng, (L, 0, 0), (0, W, 0), (0, 0, H), int(W * H * density)),
        _rect(rng, (0, 0, 0), (L, 0, 0), (0, 0, H), int(L * H * density)),
        _rect(rng, (0, W, 0), (L, 0, 0), (0, 0, H), int(L * H * density)),
        _box(rng, (2.0, 1.0, 0.0), (1.2, 0.8, 0.75), density),                  # table
        _box(rng, (0.0, 2.5, 0.0), (0.5, 1.0, 1.8), density),                   # armoire
    ]
    pts = np.vstack(parts)
    pts = pts + rng.normal(0, noise, pts.shape)
    cols = np.clip(rng.normal(0.6, 0.1, (len(pts), 3)), 0, 1)
    is_out = np.zeros(len(pts), dtype=bool)
    n_out = int(outlier_frac * len(pts))
    if n_out:
        lo, hi = pts.min(0) - 0.5, pts.max(0) + 0.5
        out = rng.uniform(lo, hi, (n_out, 3))
        pts = np.vstack([pts, out])
        cols = np.vstack([cols, rng.random((n_out, 3))])
        is_out = np.concatenate([is_out, np.ones(n_out, dtype=bool)])
    return pts, cols, is_out


def apply_similarity(pts, tilt_x=0.0, tilt_y=0.0, yaw=0.0, scale=1.0, t=(0, 0, 0)):
    """p' = scale * R p + t. Retourne (points, G 4x4)."""
    rx, ry, rz = np.radians([tilt_x, tilt_y, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R = A.rot_z(rz) @ Rx @ Ry
    G = np.eye(4)
    G[:3, :3] = scale * R
    G[:3, 3] = np.asarray(t, dtype=float)
    return pts @ G[:3, :3].T + G[:3, 3], G


def wrap90(deg):
    return (deg + 45.0) % 90.0 - 45.0


def residual(M, G):
    """Rotation residuelle de la chaine piece -> scan (G) -> resultat (M) : inclinaison et cap mod 90."""
    s = A.matrix_summary(M @ G)
    return s["tilt_deg"], wrap90(s["yaw_deg"]), s["scale"]


def height(pts):
    lo, hi = np.percentile(pts[:, 2], [0.5, 99.5])
    return float(hi - lo)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run(pts, cols, **kw):
    c, J = A.run(pts, cols, verbose=False, **kw)
    return c, J


def test_clean_identity():
    """Piece propre, deja alignee : le pretraitement ne doit (quasi) rien changer."""
    pts, cols, _ = make_room(seed=1)
    c, J = _run(pts, cols)
    s = A.matrix_summary(c.M)
    removed = 1 - len(c.pts) / len(pts)
    ok = (removed < TOL_REMOVED and abs(s["yaw_deg"]) < TOL_ANGLE and s["tilt_deg"] < TOL_ANGLE
          and abs(s["scale"] - 1) < TOL_SCALE)
    return ok, (f"retires {100 * removed:.2f} % | inclinaison {s['tilt_deg']:.2f} deg | "
                f"lacet {s['yaw_deg']:+.2f} deg | echelle x{s['scale']:.4f}")


def test_recover():
    """Scan incline (3, -2 deg), tourne de 33 deg, 1 % d'outliers, decale : tout est corrige.

    Rappel des outliers : mesure sur ceux qui sont a plus de 10 cm de toute surface (points
    flottants). Un point a moins de 5 cm d'un mur est geometriquement indiscernable du mur :
    aucun filtre ne peut le retirer sans abimer la surface. Le detail par distance est affiche.
    """
    from scipy.spatial import cKDTree
    room, cols, is_out = make_room(seed=2, outlier_frac=0.01)
    obs, G = apply_similarity(room, 3.0, -2.0, 33.0, 1.0, (10, -5, 3))
    c, J = _run(obs, cols)
    tilt, yaw, _ = residual(c.M, G)
    removed_mask = np.ones(len(obs), dtype=bool)
    removed_mask[c.idx] = False
    d, _ = cKDTree(obs[~is_out]).query(obs[is_out], workers=-1)
    rem = removed_mask[is_out]
    recall_far = rem[d > 0.10].mean()
    bins = " ".join(f"[{lo:.2f}-{hi:.2f} m: {100 * rem[(d >= lo) & (d < hi)].mean():.0f} %]"
                    for lo, hi in ((0, 0.05), (0.05, 0.10), (0.10, 99)) if ((d >= lo) & (d < hi)).any())
    false_rm = removed_mask[~is_out].mean()
    ok = tilt < TOL_ANGLE and abs(yaw) < TOL_ANGLE and recall_far >= 0.99 and false_rm < TOL_REMOVED
    return ok, (f"inclinaison residuelle {tilt:.2f} deg | cap residuel {yaw:+.2f} deg | "
                f"outliers flottants (>10 cm) retires {100 * recall_far:.1f} % | "
                f"inliers retires a tort {100 * false_rm:.2f} % | par distance a la surface : {bins}")


def test_idempotence():
    """Deuxieme passage sur un resultat deja pretraite : ne doit plus rien changer."""
    room, cols, _ = make_room(seed=3, outlier_frac=0.01)
    obs, _ = apply_similarity(room, 3.0, -2.0, 33.0, 1.0, (10, -5, 3))
    c1, _ = _run(obs, cols)
    c2, _ = _run(c1.pts, c1.cols)
    s = A.matrix_summary(c2.M)
    removed = 1 - len(c2.pts) / len(c1.pts)
    ok = (removed < TOL_REMOVED and abs(s["yaw_deg"]) < TOL_ANGLE and s["tilt_deg"] < TOL_ANGLE
          and abs(s["scale"] - 1) < TOL_SCALE)
    return ok, (f"2e passage : retires {100 * removed:.2f} % | inclinaison {s['tilt_deg']:.2f} deg | "
                f"lacet {s['yaw_deg']:+.2f} deg | echelle x{s['scale']:.4f}")


def test_determinism():
    """Trois lancements sur la meme entree donnent exactement le meme resultat."""
    room, cols, _ = make_room(seed=4, outlier_frac=0.01)
    obs, _ = apply_similarity(room, 3.0, -2.0, 33.0, 1.0, (10, -5, 3))
    outs = [_run(obs, cols)[0].pts for _ in range(3)]
    ok = all(o.shape == outs[0].shape and np.array_equal(o, outs[0]) for o in outs[1:])
    return ok, "3 lancements identiques" if ok else "resultats differents"


def test_units():
    """Scan en millimetres -> metres ; echelle arbitraire -> hauteur cible."""
    room, cols, _ = make_room(seed=5)
    mm, _ = apply_similarity(room, scale=1000.0)
    h_mm = height(_run(mm, cols)[0].pts)
    arb, _ = apply_similarity(room, scale=0.37)
    h_arb = height(_run(arb, cols)[0].pts)
    ok = abs(h_mm - 2.6) < 0.05 and abs(h_arb - 2.5) < 0.05
    return ok, f"mm -> hauteur {h_mm:.2f} m (attendu 2.60) | echelle x0.37 -> {h_arb:.2f} m (attendu 2.50)"


def test_tilted_45():
    """Scan penche de 45 deg autour de X : la verticale est retrouvee."""
    room, cols, _ = make_room(seed=6)
    obs, G = apply_similarity(room, tilt_x=45.0)
    c, J = _run(obs, cols)
    tilt, yaw, _ = residual(c.M, G)
    return tilt < TOL_ANGLE, f"inclinaison residuelle {tilt:.2f} deg (scan penche de 45 deg)"


def known_limits():
    """Limites connues, affichees a titre informatif (non comptees comme echec).

    Le pretraitement suppose que le scan arrive avec Z vers le haut (comme le dit PIPELINE.md).
    Un scan couche a 90 deg ou a l'envers (180 deg) est ambigu : une autre direction 'Manhattan'
    coincide avec Z, ou le sens haut/bas est indeterminable. Le pretraitement actuel a la meme limite.
    """
    room, cols, _ = make_room(seed=6)
    out = []
    for a in (90.0, 180.0):
        obs, G = apply_similarity(room, tilt_x=a)
        c, _ = _run(obs, cols)
        out.append(f"scan tourne de {a:.0f} deg -> inclinaison residuelle {residual(c.M, G)[0]:.1f} deg")
    return out


ALL_TESTS = [test_clean_identity, test_recover, test_idempotence, test_determinism,
             test_units, test_tilted_45]


def test_all():
    """Point d'entree pytest : chaque test doit reussir."""
    for t in ALL_TESTS:
        ok, detail = t()
        assert ok, f"{t.__name__} : {detail}"


# ---------------------------------------------------------------------------
# Comparaison sur fichiers reels : adaptatif vs pretraitement actuel
# ---------------------------------------------------------------------------

def umeyama(P, Q):
    """R, s tels que Q ~ s R P (correspondance point a point)."""
    muP, muQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - muP, Q - muQ
    U, D, Vt = np.linalg.svd(Qc.T @ Pc / len(P))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / ((Pc ** 2).sum() / len(P)))
    return R, s


def legacy_metrics(pts):
    import preprocess_for_spatiallm as L
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    with contextlib.redirect_stdout(io.StringIO()):
        d = L.denoise(pcd, "moderate", verbose=False)
        p0 = np.asarray(d.points).copy()
        d = L.align_z_up_pca(d)
        d = L.align_manhattan(d)
        d = L.scale_to_metric(d, 2.5)
    R, s = umeyama(p0, np.asarray(d.points))
    M = np.eye(4)
    M[:3, :3] = s * R
    m = A.matrix_summary(M)
    return dict(removed=1 - len(p0) / len(pts), **m)


def real_report(paths):
    print("\n=== Fichiers reels : pretraitement ACTUEL vs ADAPTATIF ===")
    for path in paths:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else None
        print(f"\n[{path.split(chr(92))[-1]}]  {len(pts):,} pts | hauteur {height(pts):.2f}")
        lg = legacy_metrics(pts)
        c1, J1 = A.run(pts, cols, verbose=False)
        s1 = A.matrix_summary(c1.M)
        c2, _ = A.run(c1.pts, c1.cols, verbose=False)
        s2 = A.matrix_summary(c2.M)
        rows = [
            ("ACTUEL   (moderate)", lg["removed"], lg["tilt_deg"], lg["yaw_deg"], lg["scale"]),
            ("ADAPTATIF 1er passage", 1 - len(c1.pts) / len(pts), s1["tilt_deg"], s1["yaw_deg"], s1["scale"]),
            ("ADAPTATIF 2e passage", 1 - len(c2.pts) / len(c1.pts), s2["tilt_deg"], s2["yaw_deg"], s2["scale"]),
        ]
        print(f"  {'':<24}{'retires':>9}{'inclin.':>9}{'lacet':>9}{'echelle':>9}")
        for name, rem, t, y, s in rows:
            print(f"  {name:<24}{100 * rem:>8.2f}%{t:>8.2f}d{y:>+8.2f}d{s:>9.4f}")
        print("  decisions adaptatives (1er passage) :")
        for r in J1.rows:
            print(f"    {r['step']:<11}{r['action']:<9}{r['measured']}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*", default=[], help="fichiers .ply reels a comparer")
    args = ap.parse_args()
    print("=== Tests synthetiques (verite terrain connue) ===")
    fails = 0
    for t in ALL_TESTS:
        ok, detail = t()
        fails += (not ok)
        print(f"[{'OK ' if ok else 'ECHEC'}] {t.__name__:<22} {detail}")
    print(f"\n{len(ALL_TESTS) - fails}/{len(ALL_TESTS)} tests reussis")
    print("Limites connues (informatif, non compte comme echec) :")
    for line in known_limits():
        print("   -", line)
    if args.real:
        real_report(args.real)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
