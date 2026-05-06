"""
Préprocessing pour SpatialLM inference.

Modes de nettoyage :
  --mode conservative  : 1 passe douce   → préserve tous les détails fins
  --mode moderate      : 1 passe moyenne → bon compromis [DÉFAUT]
  --mode aggressive    : 2 passes + radius → nettoyage maximum

Usage:
    python preprocess_for_spatiallm.py -i noisy.ply -o clean.ply
    python preprocess_for_spatiallm.py -i noisy.ply -o clean.ply --mode conservative
    python preprocess_for_spatiallm.py -i folder/ -o clean_folder/ --mode moderate
"""

import os
import glob
import argparse
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------

def denoise(pcd, mode="moderate", nb_neighbors=None, std_ratio=None,
            use_radius=False, radius=0.05, radius_min_points=8, verbose=True):
    n0 = len(pcd.points)
    presets = {
        "conservative": dict(passes=[(20, 2.5)],            use_radius=False),
        "moderate":     dict(passes=[(20, 2.0)],            use_radius=False),
        "aggressive":   dict(passes=[(20, 2.0), (10, 1.5)], use_radius=True),
    }
    if nb_neighbors is not None and std_ratio is not None:
        passes = [(nb_neighbors, std_ratio)]
        use_radius_final = use_radius
        label = f"custom (nb={nb_neighbors}, std={std_ratio})"
    else:
        cfg = presets[mode]
        passes = cfg["passes"]
        use_radius_final = use_radius if use_radius else cfg["use_radius"]
        label = mode

    for nb, std in passes:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    if use_radius_final:
        pcd, _ = pcd.remove_radius_outlier(nb_points=radius_min_points, radius=radius)

    if verbose:
        n1 = len(pcd.points)
        suffix = " + radius" if use_radius_final else ""
        print(f"  Denoise [{label}{suffix}]: {n0:,} -> {n1:,} pts (-{100*(n0-n1)/n0:.1f}%)")
    return pcd


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def keep_largest_cluster(pcd, eps=0.05, min_points=20, verbose=True):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.max() < 0:
        if verbose:
            print("  DBSCAN: aucun cluster trouvé, rien changé")
        return pcd
    valid = labels >= 0
    counts = np.bincount(labels[valid])
    largest_label = np.argmax(counts)
    keep = labels == largest_label
    if verbose:
        print(f"  DBSCAN: {labels.max()+1} clusters → garde le + gros ({keep.sum():,} pts)")
    return pcd.select_by_index(np.where(keep)[0])


# ---------------------------------------------------------------------------
# ÉTAPE 1 : Alignement Z-up — détection robuste du SOL
# ---------------------------------------------------------------------------

def align_z_up_pca(pcd, verbose=True):
    """
    Stratégie en cascade pour trouver le plan de sol réel :

    1. RANSAC sur les points du bas (percentile 0-15%)  → cible le sol directement
    2. Si score < 0.85 : RANSAC sur percentile 0-25%   → sol élargi
    3. Si score < 0.85 : RANSAC sur percentile 0-10%   → sol strict
    4. Fallback PCA globale si tout échoue

    Un plan est considéré "horizontal" si |normal.Z| > 0.85 (angle < 32° par rapport au sol).
    La meilleure normale (score = |dot(normal, Z)|) est conservée.
    """
    points = np.asarray(pcd.points)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_range = z_max - z_min

    best_normal = None
    best_score = -1.0

    # Tranches de points à tester, de la plus ciblée à la plus large
    slices = [
        ("sol 0-15%",  0.00, 0.15),
        ("sol 0-25%",  0.00, 0.25),
        ("sol 0-10%",  0.00, 0.10),
        ("sol 5-20%",  0.05, 0.20),
    ]

    for label_s, z_lo, z_hi in slices:
        mask = (points[:, 2] >= z_min + z_lo * z_range) & \
               (points[:, 2] <= z_min + z_hi * z_range)
        pts_slice = points[mask]

        if len(pts_slice) < 50:
            continue

        slice_pcd = o3d.geometry.PointCloud()
        slice_pcd.points = o3d.utility.Vector3dVector(pts_slice)

        try:
            plane_model, inliers = slice_pcd.segment_plane(
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=2000
            )
        except Exception:
            continue

        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        score = abs(normal[2])  # 1.0 = parfaitement horizontal

        if verbose:
            print(f"  [RANSAC {label_s}] score={score:.3f} "
                  f"normal=[{a:.2f},{b:.2f},{c:.2f}] "
                  f"inliers={len(inliers):,}/{len(pts_slice):,}")

        if score > best_score:
            best_score = score
            best_normal = normal.copy()

        # Assez bon → on s'arrête
        if best_score >= 0.97:
            break

    # Fallback PCA si aucune tranche n'a donné un plan horizontal
    if best_normal is None or best_score < 0.85:
        if verbose:
            print(f"  ⚠️  RANSAC insuffisant (best_score={best_score:.3f}), fallback PCA")
        centroid = points.mean(axis=0)
        cov = np.cov((points - centroid).T)
        _, eigenvectors = np.linalg.eigh(cov)
        best_normal = eigenvectors[:, 0]
        best_score = abs(best_normal[2])

    # S'assurer que la normale pointe vers le haut
    if best_normal[2] < 0:
        best_normal = -best_normal

    # Construire la rotation qui aligne best_normal sur (0,0,1)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(best_normal, z)
    s = np.linalg.norm(v)
    c_dot = np.dot(best_normal, z)
    angle_deg = np.degrees(np.arccos(np.clip(c_dot, -1, 1)))

    if s < 1e-6:
        R = np.eye(3) if c_dot > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2],  0, -v[0]],
                       [-v[1], v[0],  0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_dot) / (s ** 2))

    centroid = points.mean(axis=0)
    rotated = (R @ (points - centroid).T).T
    rotated[:, 2] -= rotated[:, 2].min()
    pcd.points = o3d.utility.Vector3dVector(rotated)

    if verbose:
        print(f"  ✅ Z-up final: correction={angle_deg:.2f}° | "
              f"score_horizontal={best_score:.3f} "
              f"({'excellent' if best_score > 0.97 else 'bon' if best_score > 0.90 else 'moyen'})")
    return pcd


# ---------------------------------------------------------------------------
# ÉTAPE 2 : Alignement Manhattan (rotation XY)
# ---------------------------------------------------------------------------

def align_manhattan(pcd, verbose=True):
    """
    Après Z-up, aligne les murs sur les axes X/Y (convention SpatialLM).
    PCA 2D sur les points muraux (tranche Z 20%-80%).
    """
    points = np.asarray(pcd.points)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_range = z_max - z_min
    mask = (points[:, 2] > z_min + 0.2 * z_range) & \
           (points[:, 2] < z_min + 0.8 * z_range)
    wall_pts = points[mask, :2]

    if len(wall_pts) < 100:
        if verbose:
            print("  Manhattan: pas assez de points muraux, skip")
        return pcd

    cov2d = np.cov(wall_pts.T)
    _, vecs = np.linalg.eigh(cov2d)
    dominant = vecs[:, 1]
    angle = np.arctan2(dominant[1], dominant[0])

    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    rotated = (R_z @ points.T).T
    rotated[:, 2] -= rotated[:, 2].min()
    pcd.points = o3d.utility.Vector3dVector(rotated)

    if verbose:
        print(f"  Manhattan: rotation Z de {np.degrees(angle):.1f}° → murs alignés X/Y")
    return pcd


# ---------------------------------------------------------------------------
# Mise à l'échelle métrique
# ---------------------------------------------------------------------------

def scale_to_metric(pcd, target_height=2.5, verbose=True):
    points = np.asarray(pcd.points)
    height = points[:, 2].max() - points[:, 2].min()
    if height < 1e-3:
        if verbose:
            print("  WARN: hauteur Z nulle, scaling ignoré")
        return pcd
    scale = target_height / height
    pcd.points = o3d.utility.Vector3dVector(points * scale)
    if verbose:
        print(f"  Scale: {height:.3f}m -> {target_height}m (x{scale:.3f})")
    return pcd


# ---------------------------------------------------------------------------
# Normalisation couleurs
# ---------------------------------------------------------------------------

def normalize_colors(pcd, verbose=True):
    if not pcd.has_colors():
        if verbose:
            print("  WARN: pas de couleurs, normalisation ignorée")
        return pcd
    colors = np.asarray(pcd.colors).copy()
    target_mean = np.array([0.62, 0.60, 0.57])
    target_std  = np.array([0.20, 0.21, 0.21])
    src_mean = colors.mean(axis=0)
    src_std  = np.where(colors.std(axis=0) < 1e-6, 1.0, colors.std(axis=0))
    colors_norm = np.clip((colors - src_mean) / src_std * target_std + target_mean, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    if verbose:
        after = colors_norm.mean(axis=0)
        print(f"  Normalize colors: [{src_mean[0]:.2f},{src_mean[1]:.2f},{src_mean[2]:.2f}]"
              f" -> [{after[0]:.2f},{after[1]:.2f},{after[2]:.2f}]")
    return pcd


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def process_one(in_path, out_path, args):
    print(f"\n[{os.path.basename(in_path)}]")
    pcd = o3d.io.read_point_cloud(in_path)
    n_init = len(pcd.points)
    print(f"  Chargé : {n_init:,} points | couleurs={pcd.has_colors()}")

    # 1. Nettoyage du bruit
    if not args.no_denoise:
        pcd = denoise(pcd, mode=args.mode, nb_neighbors=args.nb_neighbors,
                      std_ratio=args.std_ratio, use_radius=args.use_radius,
                      radius=args.radius, radius_min_points=args.radius_min_points)
    else:
        print("  Denoise: ignoré (--no_denoise)")

    # 2. Garder uniquement le plus gros cluster
    if args.keep_largest_cluster and not args.no_denoise:
        pcd = keep_largest_cluster(pcd, eps=args.dbscan_eps, min_points=args.dbscan_min_points)

    # 3. Alignement Z-up (RANSAC sol) + Manhattan (XY)
    if not args.no_align:
        pcd = align_z_up_pca(pcd)   # détection robuste du sol
        pcd = align_manhattan(pcd)  # murs alignés sur X/Y

    # 4. Mise à l'échelle métrique
    if not args.no_scale:
        pcd = scale_to_metric(pcd, target_height=args.target_height)

    # 5. Voxel downsample (optionnel)
    if args.voxel_size > 0:
        n_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"  Voxel ({args.voxel_size}m) : {n_before:,} -> {len(pcd.points):,} pts")

    # 6. Normalisation couleurs (optionnelle)
    if args.normalize_colors:
        pcd = normalize_colors(pcd)

    # Stats finales
    pts = np.asarray(pcd.points)
    n_final = len(pts)
    print(f"  ✅ Résultat : {n_final:,} pts conservés ({100*n_final/n_init:.1f}%)")
    print(f"  Dims : X={pts[:,0].ptp():.2f}m  Y={pts[:,1].ptp():.2f}m  Z={pts[:,2].ptp():.2f}m")
    if pcd.has_colors():
        c = np.asarray(pcd.colors)
        print(f"  Mean RGB : [{c[:,0].mean():.3f}, {c[:,1].mean():.3f}, {c[:,2].mean():.3f}]")
        print(f"  Std  RGB : [{c[:,0].std():.3f},  {c[:,1].std():.3f},  {c[:,2].std():.3f}]")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  Sauvegardé -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )

    p.add_argument("-i", "--input",  required=True, help=".ply ou dossier de .ply")
    p.add_argument("-o", "--output", required=True, help=".ply ou dossier de sortie")

    grp = p.add_argument_group("Nettoyage")
    grp.add_argument("--no_denoise", action="store_true",
                     help="Skip denoising and DBSCAN — keep all points, only run alignment + scaling")
    grp.add_argument("--mode", choices=["conservative", "moderate", "aggressive"],
                     default="moderate")
    grp.add_argument("--nb_neighbors", type=int, default=None)
    grp.add_argument("--std_ratio", type=float, default=None)
    grp.add_argument("--use_radius", action="store_true")
    grp.add_argument("--radius", type=float, default=0.05)
    grp.add_argument("--radius_min_points", type=int, default=8)

    grp2 = p.add_argument_group("Clustering")
    grp2.add_argument("--keep_largest_cluster", action="store_true")
    grp2.add_argument("--dbscan_eps", type=float, default=0.05)
    grp2.add_argument("--dbscan_min_points", type=int, default=20)

    grp3 = p.add_argument_group("Géométrie")
    grp3.add_argument("--no_align", action="store_true",
                      help="Skip Z-up + Manhattan alignment")
    grp3.add_argument("--no_scale", action="store_true")
    grp3.add_argument("--target_height", type=float, default=2.5)
    grp3.add_argument("--voxel_size", type=float, default=0.0)

    grp4 = p.add_argument_group("Couleurs")
    grp4.add_argument("--normalize_colors", action="store_true")

    args = p.parse_args()

    if (args.nb_neighbors is None) != (args.std_ratio is None):
        p.error("--nb_neighbors et --std_ratio doivent être fournis ensemble")

    if os.path.isfile(args.input):
        process_one(args.input, args.output, args)
    else:
        ply_files = sorted(glob.glob(os.path.join(args.input, "*.ply")))
        if not ply_files:
            print(f"Aucun .ply trouvé dans {args.input}")
        for f in ply_files:
            out = os.path.join(args.output, os.path.basename(f))
            process_one(f, out, args)

    print("\nDone.")
