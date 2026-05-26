"""
postprocess.py
--------------
Post-processing step between SpatialLM inference and IFC generation.

Fixes common SpatialLM detection errors using geometry:
  1. Merges collinear walls (e.g. one wall split into two by the model)
  2. Refits bounding boxes to the actual point cloud
  3. Outputs a corrected TXT layout file

Uses only numpy + scipy — runs in any environment.

Usage:
    python postprocess.py \
        --point_cloud clean_scan.ply \
        --layout raw_layout.txt \
        --output refined_layout.txt
"""

import argparse
import math
import re
import sys
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from scipy.spatial import ConvexHull, Delaunay
from typing import List, Tuple, Optional


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class WallData:
    name: str
    x1: float; y1: float; z1: float
    x2: float; y2: float; z2: float
    height: float
    thickness: float

@dataclass
class DoorData:
    name: str
    parent_wall: str
    cx: float; cy: float; cz: float
    width: float; height: float

@dataclass
class WindowData:
    name: str
    parent_wall: str
    cx: float; cy: float; cz: float
    width: float; height: float

@dataclass
class BboxData:
    name: str
    obj_class: str
    cx: float; cy: float; cz: float
    rotation: float
    length: float; width: float; height: float


# ── Parser ────────────────────────────────────────────────────────────

def parse_txt(filepath):
    walls, doors, windows, bboxes, other_lines = [], [], [], [], []
    with open(filepath) as f:
        for line in f:
            raw = line.rstrip("\n")
            line = raw.strip()
            if not line or line.startswith("#"):
                other_lines.append(raw)
                continue

            m = re.match(r"(wall_\w+)=Wall\(([^)]+)\)", line)
            if m:
                vals = [float(v) for v in m.group(2).split(",")]
                walls.append(WallData(m.group(1), *vals))
                continue

            m = re.match(r"(door_\w+)=Door\((\w+),([^)]+)\)", line)
            if m:
                vals = [float(v) for v in m.group(3).split(",")]
                doors.append(DoorData(m.group(1), m.group(2), *vals))
                continue

            m = re.match(r"(window_\w+)=Window\((\w+),([^)]+)\)", line)
            if m:
                vals = [float(v) for v in m.group(3).split(",")]
                windows.append(WindowData(m.group(1), m.group(2), *vals))
                continue

            m = re.match(r"(\w+)=Bbox\((\w+),([^)]+)\)", line)
            if m:
                vals = [float(v) for v in m.group(3).split(",")]
                bboxes.append(BboxData(m.group(1), m.group(2), *vals))
                continue

            other_lines.append(raw)

    return walls, doors, windows, bboxes


# ── Wall geometry helpers ─────────────────────────────────────────────

def wall_angle(w: WallData) -> float:
    return math.atan2(w.y2 - w.y1, w.x2 - w.x1)

def wall_length(w: WallData) -> float:
    return math.sqrt((w.x2 - w.x1)**2 + (w.y2 - w.y1)**2)

def wall_unit_vec(w: WallData) -> Tuple[float, float]:
    dx, dy = w.x2 - w.x1, w.y2 - w.y1
    length = math.sqrt(dx**2 + dy**2)
    return (dx / length, dy / length) if length > 1e-6 else (1.0, 0.0)

def perpendicular_distance(w1: WallData, px: float, py: float) -> float:
    """Distance from point (px, py) to the infinite line defined by w1."""
    dx, dy = w1.x2 - w1.x1, w1.y2 - w1.y1
    length = math.sqrt(dx**2 + dy**2)
    if length < 1e-6:
        return float("inf")
    return abs((py - w1.y1) * dx - (px - w1.x1) * dy) / length


def are_collinear(w1: WallData, w2: WallData,
                  angle_thresh_deg: float = 10.0,
                  dist_thresh: float = 0.35,
                  gap_thresh: float = 0.6) -> bool:
    """
    Two walls are collinear when:
      1. Their directions are nearly parallel
      2. They lie on the same line (perpendicular distance < dist_thresh)
      3. Their projections on the common axis nearly touch or overlap
    """
    # 1. Angle check (handle 180° ambiguity)
    a1 = math.degrees(wall_angle(w1)) % 180
    a2 = math.degrees(wall_angle(w2)) % 180
    angle_diff = min(abs(a1 - a2), 180 - abs(a1 - a2))
    if angle_diff > angle_thresh_deg:
        return False

    # 2. Perpendicular distance: midpoint of w2 from w1's line
    mx, my = (w2.x1 + w2.x2) / 2, (w2.y1 + w2.y2) / 2
    if perpendicular_distance(w1, mx, my) > dist_thresh:
        return False

    # 3. Projection gap on w1's axis
    ux, uy = wall_unit_vec(w1)
    t_w1 = [0.0, wall_length(w1)]
    t_w2 = [
        (w2.x1 - w1.x1) * ux + (w2.y1 - w1.y1) * uy,
        (w2.x2 - w1.x1) * ux + (w2.y2 - w1.y1) * uy,
    ]
    t_w2.sort()
    gap = max(0.0, max(min(t_w1), min(t_w2)) - min(max(t_w1), max(t_w2)))
    return gap <= gap_thresh


def merge_walls(w1: WallData, w2: WallData, merged_name: str) -> WallData:
    """Merge two collinear walls into one spanning the full extent."""
    ux, uy = wall_unit_vec(w1)
    # Project all 4 endpoints onto w1's axis
    pts = [(w1.x1, w1.y1), (w1.x2, w1.y2), (w2.x1, w2.y1), (w2.x2, w2.y2)]
    ts = [(p[0] - w1.x1) * ux + (p[1] - w1.y1) * uy for p in pts]
    t_min, t_max = min(ts), max(ts)

    new_x1 = w1.x1 + t_min * ux
    new_y1 = w1.y1 + t_min * uy
    new_x2 = w1.x1 + t_max * ux
    new_y2 = w1.y1 + t_max * uy
    new_z  = min(w1.z1, w2.z1)

    return WallData(
        name=merged_name,
        x1=new_x1, y1=new_y1, z1=new_z,
        x2=new_x2, y2=new_y2, z2=new_z,
        height=max(w1.height, w2.height),
        thickness=max(w1.thickness, w2.thickness) or 0.2,
    )


def merge_collinear_walls(walls: List[WallData],
                          angle_thresh_deg: float = 10.0,
                          dist_thresh: float = 0.35,
                          gap_thresh: float = 0.6) -> List[WallData]:
    """Iteratively merge collinear walls until no more merges are possible."""
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(walls)
        for i in range(len(walls)):
            if used[i]:
                continue
            w = walls[i]
            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue
                if are_collinear(w, walls[j], angle_thresh_deg, dist_thresh, gap_thresh):
                    w = merge_walls(w, walls[j], w.name)
                    used[j] = True
                    changed = True
            merged.append(w)
            used[i] = True
        walls = merged
    return walls


# ── Bbox refinement ───────────────────────────────────────────────────

def bbox_corners_3d(b: BboxData) -> np.ndarray:
    """Compute 8 corner points of a BboxData object."""
    cos_r, sin_r = math.cos(b.rotation), math.sin(b.rotation)
    hl, hw, hh = b.length / 2, b.width / 2, b.height / 2
    # Local corners (unrotated)
    local = np.array([
        [-hl, -hw, -hh], [+hl, -hw, -hh], [+hl, +hw, -hh], [-hl, +hw, -hh],
        [-hl, -hw, +hh], [+hl, -hw, +hh], [+hl, +hw, +hh], [-hl, +hw, +hh],
    ])
    # Rotate around Z
    rot = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    rotated = local @ rot.T
    return rotated + np.array([b.cx, b.cy, b.cz])


def points_in_convex_hull(corners: np.ndarray, points: np.ndarray):
    """Return points inside the convex hull of corners."""
    try:
        hull = Delaunay(corners)
        mask = hull.find_simplex(points) >= 0
        return points[mask], np.where(mask)[0]
    except Exception:
        return np.empty((0, 3)), np.array([])


def fit_min_bbox_2d(points: np.ndarray):
    """
    Fit a minimum-area oriented bounding box to a point cloud.
    Returns (center_x, center_y, angle_rad, length, width).
    """
    pts2d = points[:, :2]
    try:
        hull = ConvexHull(pts2d)
    except Exception:
        return None

    hull_pts = pts2d[hull.vertices]
    edges = np.diff(np.vstack([hull_pts, hull_pts[0]]), axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    best_area = float("inf")
    best = None
    for angle in angles:
        c, s = math.cos(-angle), math.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        rotated = pts2d @ rot.T
        min_x, max_x = rotated[:, 0].min(), rotated[:, 0].max()
        min_y, max_y = rotated[:, 1].min(), rotated[:, 1].max()
        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            best = (angle, min_x, max_x, min_y, max_y)

    angle, min_x, max_x, min_y, max_y = best
    c, s = math.cos(angle), math.sin(angle)
    rot_inv = np.array([[c, -s], [s, c]])
    center_local = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    cx, cy = center_local @ rot_inv.T
    length = max_x - min_x
    width  = max_y - min_y

    return cx, cy, angle, length, width


def refine_bboxes(bboxes: List[BboxData], all_points: np.ndarray,
                  min_points: int = 50) -> List[BboxData]:
    """Refit each bbox to the actual point cloud. Skip if too few points found."""
    refined = []
    for b in bboxes:
        corners = bbox_corners_3d(b)
        pts_in, _ = points_in_convex_hull(corners, all_points)

        if len(pts_in) < min_points:
            refined.append(b)
            continue

        result = fit_min_bbox_2d(pts_in)
        if result is None:
            refined.append(b)
            continue

        cx, cy, angle, length, width = result
        z_vals = pts_in[:, 2]
        cz     = (z_vals.min() + z_vals.max()) / 2
        height = float(z_vals.max() - z_vals.min())
        length, width = max(length, width), min(length, width)

        refined.append(BboxData(
            name=b.name, obj_class=b.obj_class,
            cx=float(cx), cy=float(cy), cz=float(cz),
            rotation=float(angle),
            length=float(length), width=float(width), height=float(height),
        ))

    return refined


# ── TXT writer ────────────────────────────────────────────────────────

def write_txt(filepath, walls, doors, windows, bboxes):
    with open(filepath, "w") as f:
        for w in walls:
            f.write(f"{w.name}=Wall({w.x1},{w.y1},{w.z1},"
                    f"{w.x2},{w.y2},{w.z2},{w.height},{w.thickness})\n")
        for d in doors:
            f.write(f"{d.name}=Door({d.parent_wall},{d.cx},{d.cy},{d.cz},"
                    f"{d.width},{d.height})\n")
        for wn in windows:
            f.write(f"{wn.name}=Window({wn.parent_wall},{wn.cx},{wn.cy},{wn.cz},"
                    f"{wn.width},{wn.height})\n")
        for b in bboxes:
            f.write(f"{b.name}=Bbox({b.obj_class},{b.cx},{b.cy},{b.cz},"
                    f"{b.rotation},{b.length},{b.width},{b.height})\n")


# ── Main ──────────────────────────────────────────────────────────────

def postprocess(layout_path: str, ply_path: str, output_path: str,
                angle_thresh: float = 10.0,
                dist_thresh: float = 0.35,
                gap_thresh: float = 0.6,
                min_points: int = 50):

    print(f"\n── Parsing {layout_path} ──")
    walls, doors, windows, bboxes = parse_txt(layout_path)
    print(f"   Input : {len(walls)} walls, {len(doors)} doors, "
          f"{len(windows)} windows, {len(bboxes)} objects")

    # ── Step 1: merge collinear walls ──
    print(f"\n── Merging collinear walls ──")
    walls_merged = merge_collinear_walls(walls, angle_thresh, dist_thresh, gap_thresh)
    n_merged = len(walls) - len(walls_merged)
    print(f"   {len(walls)} → {len(walls_merged)} walls  "
          f"({n_merged} merge{'s' if n_merged != 1 else ''})")
    for i, w in enumerate(walls_merged):
        print(f"   {w.name}: L={math.sqrt((w.x2-w.x1)**2+(w.y2-w.y1)**2):.2f}m "
              f"H={w.height:.2f}m  angle={math.degrees(wall_angle(w)):.1f}°")

    # ── Step 2: refit bboxes to point cloud ──
    print(f"\n── Refitting bboxes to point cloud ──")
    pcd = o3d.io.read_point_cloud(ply_path)
    all_points = np.asarray(pcd.points)
    print(f"   Loaded {len(all_points):,} points from {ply_path}")
    bboxes_refined = refine_bboxes(bboxes, all_points, min_points)
    n_refined = sum(
        1 for a, b in zip(bboxes, bboxes_refined)
        if abs(a.length - b.length) > 0.01 or abs(a.width - b.width) > 0.01
    )
    print(f"   {n_refined}/{len(bboxes)} objects refined from point cloud")

    # ── Write output ──
    write_txt(output_path, walls_merged, doors, windows, bboxes_refined)
    print(f"\n✅ Refined layout saved → {output_path}")

    return walls_merged, doors, windows, bboxes_refined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process SpatialLM layout output")
    parser.add_argument("-l", "--layout",      required=True, help="Raw SpatialLM TXT layout")
    parser.add_argument("-p", "--point_cloud", required=True, help="Clean PLY point cloud")
    parser.add_argument("-o", "--output",      required=True, help="Refined TXT output path")
    parser.add_argument("--angle_thresh", type=float, default=10.0,
                        help="Max angle difference (degrees) to consider walls collinear (default: 10)")
    parser.add_argument("--dist_thresh",  type=float, default=0.35,
                        help="Max perpendicular distance (m) between collinear walls (default: 0.35)")
    parser.add_argument("--gap_thresh",   type=float, default=0.6,
                        help="Max gap (m) between collinear wall projections (default: 0.6)")
    parser.add_argument("--min_points",   type=int,   default=50,
                        help="Min points inside bbox to trigger refinement (default: 50)")
    args = parser.parse_args()

    postprocess(
        layout_path=args.layout,
        ply_path=args.point_cloud,
        output_path=args.output,
        angle_thresh=args.angle_thresh,
        dist_thresh=args.dist_thresh,
        gap_thresh=args.gap_thresh,
        min_points=args.min_points,
    )
