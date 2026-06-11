"""
Preprocessing for SpatialLM — large LAS / E57 building scans.

Designed for full-building point clouds (multi-room, multi-floor, 5-6 GB+).
Key differences from preprocess_for_spatiallm.py:
  - Supports .las / .laz / .e57 / .ply input
  - Chunked loading for .las to avoid OOM on large files
  - NO fixed-height rescaling — real metric units are preserved
  - Voxel downsample applied ON LOAD (before anything else) to tame memory
  - Gentler denoising defaults suited to heterogeneous building scans
  - Optional floor segmentation: splits multi-floor scans into per-floor .ply files

Modes:
  --mode conservative  : light denoising, preserves detail
  --mode moderate      : balanced [DEFAULT]
  --mode aggressive    : heavier denoising, max clean-up

Usage:
    # single file, 5 cm voxel
    python preprocess_large_building.py -i building.las -o clean.ply --voxel_size 0.05

    # E57, split into per-floor files
    python preprocess_large_building.py -i building.e57 -o floors/building.ply --split_floors

    # already-downsampled PLY, skip denoising
    python preprocess_large_building.py -i building.ply -o clean.ply --no_denoise

    # whole folder of LAS files
    python preprocess_large_building.py -i scans/ -o clean/ --voxel_size 0.05

Dependencies:
    pip install open3d laspy[lazrs] pye57 scipy numpy
"""

import os
import sys
import glob
import argparse
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _pcd_from_xyz_colors(xyz, rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb is not None:
        # convert in-place to avoid a temporary double-sized array
        rgb64 = rgb.astype(np.float64)
        np.clip(rgb64, 0, 1, out=rgb64)
        pcd.colors = o3d.utility.Vector3dVector(rgb64)
    return pcd


def load_las(path, voxel_size_early=0.0, chunk_size=8_000_000):
    """
    Load .las / .laz in chunks to avoid loading the full file into RAM at once.
    Each chunk is voxel-downsampled before accumulating, keeping peak memory low.
    """
    try:
        import laspy
    except ImportError:
        sys.exit("laspy not installed — run:  pip install laspy[lazrs]")

    print(f"  [LAS] {path}")

    chunks_xyz = []
    chunks_rgb = []
    total_raw  = 0

    with laspy.open(path) as las_file:
        dim_names = las_file.header.point_format.dimension_names
        has_rgb = all(d in dim_names for d in ("red", "green", "blue"))
        has_intensity = "intensity" in dim_names

        for chunk in las_file.chunk_iterator(chunk_size):
            xyz = np.vstack([chunk.x, chunk.y, chunk.z]).T.astype(np.float32)
            total_raw += len(xyz)

            if has_rgb:
                r = np.asarray(chunk.red,   dtype=np.float32) / 65535.0
                g = np.asarray(chunk.green, dtype=np.float32) / 65535.0
                b = np.asarray(chunk.blue,  dtype=np.float32) / 65535.0
                rgb = np.vstack([r, g, b]).T
            elif has_intensity:
                intensity = np.asarray(chunk.intensity, dtype=np.float32)
                vmin, vmax = intensity.min(), intensity.max()
                intensity = (intensity - vmin) / (vmax - vmin + 1e-9)
                rgb = np.vstack([intensity, intensity, intensity]).T
            else:
                rgb = None

            if voxel_size_early > 0:
                chunk_pcd = _pcd_from_xyz_colors(xyz, rgb)
                chunk_pcd = chunk_pcd.voxel_down_sample(voxel_size=voxel_size_early)
                xyz = np.asarray(chunk_pcd.points, dtype=np.float32)
                rgb = np.asarray(chunk_pcd.colors, dtype=np.float32) if chunk_pcd.has_colors() else None

            chunks_xyz.append(xyz)
            if rgb is not None:
                chunks_rgb.append(rgb)

    print(f"  Raw points : {total_raw:,}")
    all_xyz = np.concatenate(chunks_xyz, axis=0)
    all_rgb = np.concatenate(chunks_rgb, axis=0) if chunks_rgb else None

    # Final global voxel pass to deduplicate chunk boundaries
    pcd = _pcd_from_xyz_colors(all_xyz, all_rgb)
    if voxel_size_early > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_early)

    color_src = "RGB" if has_rgb else ("intensity" if has_intensity else "none")
    print(f"  After voxel({voxel_size_early}m) : {len(pcd.points):,} pts | colors={color_src}")
    return pcd


def load_e57(path, voxel_size_early=0.0):
    """
    Load .e57 via Open3D native reader (>=0.16).
    Falls back to pye57 if Open3D returns an empty cloud.
    Multi-scan E57: all scans are merged.
    """
    print(f"  [E57] {path}")
    pcd = o3d.io.read_point_cloud(path)

    if len(pcd.points) == 0:
        try:
            import pye57
        except ImportError:
            sys.exit(
                "Open3D could not read this E57 file and pye57 is not installed.\n"
                "Install:  pip install pye57"
            )
        e57_file = pye57.E57(path)
        all_xyz, all_rgb = [], []
        for scan_idx in range(e57_file.scan_count):
            try:
                data = e57_file.read_scan_raw(scan_idx)
            except Exception as exc:
                print(f"    scan {scan_idx}: read error ({exc}), skipping")
                continue
            xyz = np.vstack([
                data["cartesianX"], data["cartesianY"], data["cartesianZ"]
            ]).T.astype(np.float32)
            all_xyz.append(xyz)
            if "colorRed" in data:
                r = data["colorRed"] / 255.0
                g = data["colorGreen"] / 255.0
                b = data["colorBlue"] / 255.0
                all_rgb.append(np.vstack([r, g, b]).T.astype(np.float32))

        merged_xyz = np.concatenate(all_xyz, axis=0)
        merged_rgb = np.concatenate(all_rgb, axis=0) if all_rgb else None
        pcd = _pcd_from_xyz_colors(merged_xyz, merged_rgb)
        print(f"  pye57 merged {e57_file.scan_count} scan(s): {len(pcd.points):,} raw pts")
    else:
        print(f"  Open3D native: {len(pcd.points):,} raw pts")

    if voxel_size_early > 0 and len(pcd.points) > 0:
        n_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_early)
        print(f"  After voxel({voxel_size_early}m) : {n_before:,} -> {len(pcd.points):,} pts")

    return pcd


def load_file(path, voxel_size_early=0.0, chunk_size=8_000_000):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".las", ".laz"):
        return load_las(path, voxel_size_early, chunk_size)
    elif ext == ".e57":
        return load_e57(path, voxel_size_early)
    else:
        print(f"  [O3D] {path}")
        pcd = o3d.io.read_point_cloud(path)
        print(f"  Raw points : {len(pcd.points):,} | colors={pcd.has_colors()}")
        if voxel_size_early > 0 and len(pcd.points) > 0:
            n_before = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size_early)
            print(f"  After voxel({voxel_size_early}m) : {n_before:,} -> {len(pcd.points):,} pts")
        return pcd


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def denoise(pcd, mode="moderate", nb_neighbors=None, std_ratio=None,
            use_radius=False, radius=0.1, radius_min_points=5, verbose=True):
    """
    Statistical outlier removal with presets tuned for heterogeneous building scans.
    Conservative/moderate defaults are gentler than the single-room script
    because full-building scans have legitimate density variations between areas.
    """
    presets = {
        "conservative": dict(passes=[(30, 3.5)],            use_radius=False),
        "moderate":     dict(passes=[(20, 2.5)],            use_radius=False),
        "aggressive":   dict(passes=[(20, 2.0), (10, 1.5)], use_radius=True),
    }
    n0 = len(pcd.points)

    if nb_neighbors is not None and std_ratio is not None:
        passes  = [(nb_neighbors, std_ratio)]
        use_r   = use_radius
        label   = f"custom(nb={nb_neighbors},std={std_ratio})"
    else:
        cfg    = presets[mode]
        passes = cfg["passes"]
        use_r  = use_radius or cfg["use_radius"]
        label  = mode

    for nb, std in passes:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    if use_r:
        pcd, _ = pcd.remove_radius_outlier(nb_points=radius_min_points, radius=radius)

    if verbose:
        n1 = len(pcd.points)
        suffix = "+radius" if use_r else ""
        print(f"  Denoise [{label}{suffix}] : {n0:,} -> {n1:,} pts (-{100*(n0-n1)/n0:.1f}%)")
    return pcd


# ---------------------------------------------------------------------------
# Z-up alignment (RANSAC floor detection)
# ---------------------------------------------------------------------------

def align_z_up(pcd, verbose=True):
    """
    Detect the lowest floor plane via RANSAC, rotate so it is Z=0.
    Works on multi-floor buildings: only the bottom percentile is used so
    upper floors don't pollute the floor-normal estimate.
    """
    points = np.asarray(pcd.points)
    z_min  = points[:, 2].min()
    z_max  = points[:, 2].max()
    z_range = z_max - z_min

    best_normal, best_score = None, -1.0

    # Try several narrow bottom slices — narrower = more focused on the ground
    slices = [
        ("0-10%",  0.00, 0.10),
        ("0-20%",  0.00, 0.20),
        ("5-15%",  0.05, 0.15),
        ("0-5%",   0.00, 0.05),
    ]
    for label_s, lo, hi in slices:
        mask = (points[:, 2] >= z_min + lo * z_range) & \
               (points[:, 2] <= z_min + hi * z_range)
        pts_slice = points[mask]
        if len(pts_slice) < 100:
            continue

        slice_pcd = o3d.geometry.PointCloud()
        slice_pcd.points = o3d.utility.Vector3dVector(pts_slice)
        try:
            plane_model, inliers = slice_pcd.segment_plane(
                distance_threshold=0.05, ransac_n=3, num_iterations=2000)
        except Exception:
            continue

        a, b, c, _ = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        score  = abs(normal[2])   # 1.0 = perfectly horizontal

        if verbose:
            print(f"  [RANSAC {label_s}] score={score:.3f}  "
                  f"normal=[{a:.2f},{b:.2f},{c:.2f}]  "
                  f"inliers={len(inliers):,}/{len(pts_slice):,}")
        if score > best_score:
            best_score, best_normal = score, normal.copy()
        if best_score >= 0.97:
            break

    if best_normal is None or best_score < 0.85:
        if verbose:
            print(f"  RANSAC insufficient (score={best_score:.3f}), fallback to PCA")
        centroid = points.mean(axis=0)
        cov = np.cov((points - centroid).T)
        _, eigvecs = np.linalg.eigh(cov)
        best_normal = eigvecs[:, 0]

    if best_normal[2] < 0:
        best_normal = -best_normal

    z     = np.array([0.0, 0.0, 1.0])
    v     = np.cross(best_normal, z)
    s     = np.linalg.norm(v)
    c_dot = np.dot(best_normal, z)
    angle_deg = np.degrees(np.arccos(np.clip(c_dot, -1, 1)))

    if s < 1e-6:
        R = np.eye(3) if c_dot > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2],  0, -v[0]],
                       [-v[1], v[0],  0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_dot) / s**2)

    centroid = points.mean(axis=0)
    rotated  = (R @ (points - centroid).T).T
    rotated[:, 2] -= rotated[:, 2].min()   # ground floor at Z=0
    pcd.points = o3d.utility.Vector3dVector(rotated)

    if verbose:
        q = "excellent" if best_score > 0.97 else "good" if best_score > 0.90 else "fair"
        print(f"  Z-up: correction={angle_deg:.2f}deg  score={best_score:.3f} ({q})")
    return pcd


# ---------------------------------------------------------------------------
# Manhattan alignment (XY walls)
# ---------------------------------------------------------------------------

def align_manhattan(pcd, verbose=True):
    """
    Rotate around Z so that dominant wall directions align to X and Y axes.
    Uses 2D PCA on wall-height points (20%-80% of total height).
    Appropriate for rectilinear buildings; has no effect on curved structures.
    """
    points  = np.asarray(pcd.points)
    z_min   = points[:, 2].min()
    z_range = points[:, 2].max() - z_min
    mask    = (points[:, 2] > z_min + 0.2 * z_range) & \
              (points[:, 2] < z_min + 0.8 * z_range)
    wall_pts = points[mask, :2]

    if len(wall_pts) < 200:
        if verbose:
            print("  Manhattan: not enough wall points, skipped")
        return pcd

    cov2d  = np.cov(wall_pts.T)
    _, vecs = np.linalg.eigh(cov2d)
    dominant = vecs[:, 1]
    angle    = np.arctan2(dominant[1], dominant[0])

    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R_z = np.array([[cos_a, -sin_a, 0],
                    [sin_a,  cos_a, 0],
                    [0,      0,     1]])
    rotated = (R_z @ points.T).T
    rotated[:, 2] -= rotated[:, 2].min()
    pcd.points = o3d.utility.Vector3dVector(rotated)

    if verbose:
        print(f"  Manhattan: Z-rotation {np.degrees(angle):.1f}deg -> walls aligned to X/Y")
    return pcd


# ---------------------------------------------------------------------------
# Color normalization
# ---------------------------------------------------------------------------

def normalize_colors(pcd, verbose=True):
    if not pcd.has_colors():
        if verbose:
            print("  Colors: none, skipping normalization")
        return pcd
    colors = np.asarray(pcd.colors).copy()
    target_mean = np.array([0.62, 0.60, 0.57])
    target_std  = np.array([0.20, 0.21, 0.21])
    src_mean = colors.mean(axis=0)
    src_std  = np.where(colors.std(axis=0) < 1e-6, 1.0, colors.std(axis=0))
    colors_norm = np.clip(
        (colors - src_mean) / src_std * target_std + target_mean, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    if verbose:
        after = colors_norm.mean(axis=0)
        print(f"  Normalize colors: "
              f"[{src_mean[0]:.2f},{src_mean[1]:.2f},{src_mean[2]:.2f}] -> "
              f"[{after[0]:.2f},{after[1]:.2f},{after[2]:.2f}]")
    return pcd


# ---------------------------------------------------------------------------
# Floor segmentation
# ---------------------------------------------------------------------------

def split_floors(pcd, floor_gap=0.5, min_floor_height=2.0, z_bin=0.1, verbose=True):
    """
    Split a multi-floor building point cloud into per-floor sub-clouds.

    Strategy:
      1. Build a Z histogram (bin size = z_bin metres).
      2. Smooth it, then find low-density gaps between floors.
      3. Each contiguous band above the density threshold that is taller
         than min_floor_height becomes one floor.
      4. Each floor is translated so its lowest point is at Z=0.

    Returns list of (floor_index, pcd) tuples.
    Requires scipy.
    """
    try:
        from scipy.ndimage import uniform_filter1d
    except ImportError:
        sys.exit("scipy is required for --split_floors — run: pip install scipy")

    points  = np.asarray(pcd.points)
    z       = points[:, 2]
    z_min, z_max = z.min(), z.max()

    n_bins = max(10, int((z_max - z_min) / z_bin))
    hist, edges = np.histogram(z, bins=n_bins)
    centers     = (edges[:-1] + edges[1:]) / 2

    smooth    = uniform_filter1d(hist.astype(float), size=7)
    threshold = smooth.max() * 0.04   # 4% of peak density = "gap"

    # Walk the smoothed histogram to identify floor bands
    floor_bands = []
    in_band     = False
    band_start  = None

    for i, density in enumerate(smooth):
        if density >= threshold and not in_band:
            band_start = centers[i]
            in_band    = True
        elif density < threshold and in_band:
            band_end = centers[i]
            if band_end - band_start >= min_floor_height:
                floor_bands.append((band_start, band_end))
            in_band = False

    if in_band and centers[-1] - band_start >= min_floor_height:
        floor_bands.append((band_start, z_max))

    if verbose:
        print(f"  Floor split: {len(floor_bands)} floor(s) detected "
              f"(Z range {z_min:.2f}m – {z_max:.2f}m, "
              f"building height={z_max-z_min:.2f}m)")
        for i, (lo, hi) in enumerate(floor_bands):
            n = np.sum((z >= lo - floor_gap) & (z <= hi + floor_gap))
            print(f"    Floor {i}: Z [{lo:.2f}m – {hi:.2f}m]  "
                  f"height={hi-lo:.2f}m  ~{n:,} pts")

    if len(floor_bands) <= 1:
        if verbose and len(floor_bands) == 0:
            print("  No floor bands found — returning whole cloud as floor 0")
        return [(0, pcd)]

    floors = []
    for i, (lo, hi) in enumerate(floor_bands):
        mask       = (z >= lo - floor_gap) & (z <= hi + floor_gap)
        idx        = np.where(mask)[0]
        floor_pcd  = pcd.select_by_index(idx)
        fpts       = np.asarray(floor_pcd.points)
        fpts[:, 2] -= fpts[:, 2].min()   # each floor starts at Z=0
        floor_pcd.points = o3d.utility.Vector3dVector(fpts)
        floors.append((i, floor_pcd))

    return floors


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process(in_path, out_path, args):
    print(f"\n{'='*64}")
    print(f"Input : {in_path}")

    # 1. Load + early voxel downsample (done inside the loader)
    pcd    = load_file(in_path,
                       voxel_size_early=args.voxel_size,
                       chunk_size=args.chunk_size)
    n_init = len(pcd.points)

    if n_init == 0:
        print("  ERROR: empty point cloud after loading — skipping")
        return

    print(f"  Loaded : {n_init:,} pts | colors={pcd.has_colors()}")

    # 2. Denoising
    if not args.no_denoise:
        pcd = denoise(pcd, mode=args.mode,
                      nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio,
                      use_radius=args.use_radius,
                      radius=args.radius, radius_min_points=args.radius_min_points)
    else:
        print("  Denoise: skipped (--no_denoise)")

    # 3. Alignment
    if not args.no_align:
        pcd = align_z_up(pcd)
        pcd = align_manhattan(pcd)
    else:
        print("  Alignment: skipped (--no_align)")

    # 4. Color normalization
    if args.normalize_colors:
        pcd = normalize_colors(pcd)

    # 5. Print overall stats
    pts   = np.asarray(pcd.points)
    n_out = len(pts)
    print(f"\n  Total after processing : {n_out:,} pts "
          f"({100*n_out/n_init:.1f}% of loaded)")
    print(f"  Dims : X={pts[:,0].ptp():.2f}m  "
          f"Y={pts[:,1].ptp():.2f}m  "
          f"Z={pts[:,2].ptp():.2f}m  "
          f"(building height = {pts[:,2].max():.2f}m)")

    # 6. Floor split or single output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if args.split_floors:
        floors = split_floors(pcd,
                              floor_gap=args.floor_gap,
                              min_floor_height=args.min_floor_height)
        if len(floors) == 1:
            _, fpcd = floors[0]
            o3d.io.write_point_cloud(out_path, fpcd)
            print(f"\n  Saved -> {out_path}  (single floor)")
        else:
            base, ext = os.path.splitext(out_path)
            ext = ext or ".ply"
            for floor_idx, floor_pcd in floors:
                fpts    = np.asarray(floor_pcd.points)
                out_f   = f"{base}_floor{floor_idx}{ext}"
                print(f"  Saving floor {floor_idx} : "
                      f"{len(fpts):,} pts | "
                      f"Z={fpts[:,2].ptp():.2f}m -> {out_f}")
                o3d.io.write_point_cloud(out_f, floor_pcd)
    else:
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )

    p.add_argument("-i", "--input",  required=True,
                   help=".las / .laz / .e57 / .ply  or a folder containing them")
    p.add_argument("-o", "--output", required=True,
                   help="output .ply  (or base path when --split_floors)")

    grp = p.add_argument_group("Downsampling (applied on load)")
    grp.add_argument("--voxel_size", type=float, default=0.05,
                     help="voxel size in metres (default 0.05 = 5 cm).  0 = disabled.\n"
                          "Recommended range: 0.03–0.10 for 5-6 GB files.")
    grp.add_argument("--chunk_size", type=int, default=3_000_000,
                     help="points per chunk when reading .las (default 3M)")

    grp2 = p.add_argument_group("Denoising")
    grp2.add_argument("--no_denoise", action="store_true",
                      help="skip denoising entirely")
    grp2.add_argument("--mode", choices=["conservative", "moderate", "aggressive"],
                      default="moderate")
    grp2.add_argument("--nb_neighbors",      type=int,   default=None,
                      help="custom StatisticalOutlierRemoval neighbours (pair with --std_ratio)")
    grp2.add_argument("--std_ratio",         type=float, default=None,
                      help="custom StatisticalOutlierRemoval std multiplier")
    grp2.add_argument("--use_radius",        action="store_true")
    grp2.add_argument("--radius",            type=float, default=0.1,
                      help="radius for radius outlier filter (default 0.1 m)")
    grp2.add_argument("--radius_min_points", type=int,   default=5)

    grp3 = p.add_argument_group("Alignment")
    grp3.add_argument("--no_align", action="store_true",
                      help="skip Z-up + Manhattan alignment")

    grp4 = p.add_argument_group("Floor splitting")
    grp4.add_argument("--split_floors", action="store_true",
                      help="detect floor levels and write one .ply per floor")
    grp4.add_argument("--floor_gap", type=float, default=0.5,
                      help="overlap margin added above/below each floor band (default 0.5 m)")
    grp4.add_argument("--min_floor_height", type=float, default=2.0,
                      help="minimum height for a band to count as a floor (default 2.0 m)")

    grp5 = p.add_argument_group("Colors")
    grp5.add_argument("--normalize_colors", action="store_true",
                      help="normalise RGB to SpatialLM target statistics")

    args = p.parse_args()

    if (args.nb_neighbors is None) != (args.std_ratio is None):
        p.error("--nb_neighbors and --std_ratio must both be provided together")

    supported_exts = (".las", ".laz", ".e57", ".ply", ".pcd", ".xyz")

    if os.path.isfile(args.input):
        process(args.input, args.output, args)
    else:
        files = sorted([
            f for f in glob.glob(os.path.join(args.input, "*"))
            if os.path.splitext(f)[1].lower() in supported_exts
        ])
        if not files:
            print(f"No supported files found in {args.input}")
            print(f"Supported extensions: {', '.join(supported_exts)}")
            sys.exit(1)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            out  = os.path.join(args.output, stem + ".ply")
            process(f, out, args)

    print("\nDone.")
