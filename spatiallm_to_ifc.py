"""
spatiallm_to_ifc_with_voids.py
-------------------------------
Convertit la sortie TXT de SpatialLM en fichier IFC standard avec vides sémantiques.

Usage:
    python spatiallm_to_ifc_with_voids.py --input scene.txt --output scene.ifc

Format d'entrée attendu (SpatialLM):
    wall_0=Wall(x1,y1,z1, x2,y2,z2, height, thickness)
    door_0=Door(parent_wall, cx,cy,cz, width, height)
    window_0=Window(parent_wall, cx,cy,cz, width, height)
    bbox_0=Bbox(class, cx,cy,cz, rotation, l,w,h)

Dépendances:
    pip install ifcopenshell numpy scipy
"""

import argparse
import math
import re
import uuid
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    import ifcopenshell
except ImportError:
    print("ERROR: ifcopenshell non installé. Lancer: pip install ifcopenshell")
    raise


# ─────────────────────────────────────────────
#  Structures de données
# ─────────────────────────────────────────────

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
    width: float
    height: float

@dataclass
class WindowData:
    name: str
    parent_wall: str
    cx: float; cy: float; cz: float
    width: float
    height: float

@dataclass
class BboxData:
    name: str
    obj_class: str
    cx: float; cy: float; cz: float
    rotation: float
    length: float; width: float; height: float


# ─────────────────────────────────────────────
#  Parser du format SpatialLM
# ─────────────────────────────────────────────

def parse_spatiallm_txt(filepath: str):
    walls, doors, windows, bboxes = [], [], [], []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Wall
            m = re.match(r"(wall_\w+)=Wall\(([^)]+)\)", line)
            if m:
                name = m.group(1)
                vals = [float(v) for v in m.group(2).split(",")]
                walls.append(WallData(name, *vals))
                continue

            # Door
            m = re.match(r"(door_\w+)=Door\((\w+),([^)]+)\)", line)
            if m:
                name = m.group(1)
                parent = m.group(2)
                vals = [float(v) for v in m.group(3).split(",")]
                doors.append(DoorData(name, parent, *vals))
                continue

            # Window
            m = re.match(r"(window_\w+)=Window\((\w+),([^)]+)\)", line)
            if m:
                name = m.group(1)
                parent = m.group(2)
                vals = [float(v) for v in m.group(3).split(",")]
                windows.append(WindowData(name, parent, *vals))
                continue

            # Bbox
            m = re.match(r"(bbox_\w+)=Bbox\((\w+),([^)]+)\)", line)
            if m:
                name = m.group(1)
                cls = m.group(2)
                vals = [float(v) for v in m.group(3).split(",")]
                bboxes.append(BboxData(name, cls, *vals))
                continue

    print(f"  Parsed: {len(walls)} murs, {len(doors)} portes, "
          f"{len(windows)} fenêtres, {len(bboxes)} objets")
    return walls, doors, windows, bboxes


# ─────────────────────────────────────────────
#  Helpers géométriques
# ─────────────────────────────────────────────

def wall_length(w: WallData) -> float:
    return math.sqrt((w.x2 - w.x1)**2 + (w.y2 - w.y1)**2)

def wall_direction(w: WallData) -> Tuple[float, float]:
    dx = w.x2 - w.x1
    dy = w.y2 - w.y1
    length = math.sqrt(dx**2 + dy**2)
    if length < 1e-6:
        return (1.0, 0.0)
    return (dx / length, dy / length)

def new_guid() -> str:
    return ifcopenshell.guid.compress(uuid.uuid4().hex)


# ─────────────────────────────────────────────
#  Construction du modèle IFC avec vides sémantiques
# ─────────────────────────────────────────────

def build_ifc(walls, doors, windows, bboxes, output_path: str):
    # ── Initialisation du modèle IFC ──
    model = ifcopenshell.file(schema="IFC4")

    # Unités
    unit_assignment = model.createIfcUnitAssignment([
        model.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE"),
        model.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE"),
        model.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE"),
        model.createIfcSIUnit(None, "PLANEANGLEUNIT", None, "RADIAN"),
    ])

    # Contexte géométrique
    ctx = model.createIfcGeometricRepresentationContext(
        None, "Model", 3, 1e-5,
        model.createIfcAxis2Placement3D(
            model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
            model.createIfcDirection((0.0, 0.0, 1.0)),
            model.createIfcDirection((1.0, 0.0, 0.0)),
        ),
        None,
    )
    body_ctx = model.createIfcGeometricRepresentationSubContext(
        "Body", "Model", None, None, None, None, ctx, None,
        "MODEL_VIEW", None,
    )

    # Hiérarchie projet > site > bâtiment > étage
    person = model.createIfcPerson(None, "User", None, None, None, None, None, None)
    org    = model.createIfcOrganization(None, "SpatialLM2IFC", None, None, None)
    pao    = model.createIfcPersonAndOrganization(person, org, None)
    app    = model.createIfcApplication(org, "1.0", "spatiallm_to_ifc", "SL2IFC")
    owner  = model.createIfcOwnerHistory(pao, app, None, "ADDED", None, pao, app,
                                         int(__import__("time").time()))

    project = model.createIfcProject(
        new_guid(), owner, "SpatialLM Project", None,
        None, None, None, [ctx], unit_assignment,
    )

    site_placement = model.createIfcLocalPlacement(
        None,
        model.createIfcAxis2Placement3D(
            model.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None),
    )
    site = model.createIfcSite(
        new_guid(), owner, "Site", None, None,
        site_placement, None, None, "ELEMENT", None, None, None, None, None,
    )
    model.createIfcRelAggregates(new_guid(), owner, None, None, project, [site])

    building_placement = model.createIfcLocalPlacement(
        site_placement,
        model.createIfcAxis2Placement3D(
            model.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None),
    )
    building = model.createIfcBuilding(
        new_guid(), owner, "Building", None, None,
        building_placement, None, None, "ELEMENT", None, None, None,
    )
    model.createIfcRelAggregates(new_guid(), owner, None, None, site, [building])

    storey_placement = model.createIfcLocalPlacement(
        building_placement,
        model.createIfcAxis2Placement3D(
            model.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None),
    )
    storey = model.createIfcBuildingStorey(
        new_guid(), owner, "Ground Floor", None, None,
        storey_placement, None, None, "ELEMENT", 0.0,
    )
    model.createIfcRelAggregates(new_guid(), owner, None, None, building, [storey])

    elements = []
    ifc_walls = {}

    # ─────────────────────────────────────────
    #  DALLE DE SOL
    # ─────────────────────────────────────────
    if len(walls) >= 3:
        try:
            from scipy.spatial import ConvexHull
            SLAB_THICKNESS = 0.1

            pts_xy = np.array(
                [[w.x1, w.y1] for w in walls] + [[w.x2, w.y2] for w in walls]
            )
            z_min = min(min(w.z1, w.z2) for w in walls)

            hull = ConvexHull(pts_xy)
            hull_pts = pts_xy[hull.vertices]

            ifc_pts = [
                model.createIfcCartesianPoint((float(p[0]), float(p[1])))
                for p in hull_pts
            ]
            ifc_pts.append(ifc_pts[0])
            polyline = model.createIfcPolyline(ifc_pts)
            profile  = model.createIfcArbitraryClosedProfileDef("AREA", None, polyline)

            slab_solid = model.createIfcExtrudedAreaSolid(
                profile,
                model.createIfcAxis2Placement3D(
                    model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
                    model.createIfcDirection((0.0, 0.0, 1.0)),
                    model.createIfcDirection((1.0, 0.0, 0.0)),
                ),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                SLAB_THICKNESS,
            )
            slab_shape = model.createIfcShapeRepresentation(
                body_ctx, "Body", "SweptSolid", [slab_solid]
            )
            slab_repr = model.createIfcProductDefinitionShape(None, None, [slab_shape])

            slab_placement = model.createIfcLocalPlacement(
                storey_placement,
                model.createIfcAxis2Placement3D(
                    model.createIfcCartesianPoint((0.0, 0.0, float(z_min) - SLAB_THICKNESS)),
                    model.createIfcDirection((0.0, 0.0, 1.0)),
                    model.createIfcDirection((1.0, 0.0, 0.0)),
                ),
            )

            ifc_slab = model.createIfcSlab(
                new_guid(), owner, "Floor", "Dalle de sol",
                None, slab_placement, slab_repr, None, "FLOOR",
            )
            elements.append(ifc_slab)
            print(f"  ✓ Dalle de sol: {len(hull_pts)} sommets, z={float(z_min) - SLAB_THICKNESS:.3f}m")
        except Exception as e:
            print(f"  ⚠ Dalle de sol ignorée: {e}")

    # ─────────────────────────────────────────
    #  MURS
    # ─────────────────────────────────────────
    for w in walls:
        length = wall_length(w)
        if length < 0.01:
            print(f"  SKIP {w.name}: longueur quasi nulle ({length:.4f}m)")
            continue

        thickness = w.thickness if w.thickness > 0.001 else 0.2
        angle_rad = math.atan2(w.y2 - w.y1, w.x2 - w.x1)

        placement = model.createIfcLocalPlacement(
            storey_placement,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((w.x1, w.y1, w.z1)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((
                    math.cos(angle_rad), math.sin(angle_rad), 0.0
                )),
            ),
        )

        profile = model.createIfcRectangleProfileDef(
            "AREA", None,
            model.createIfcAxis2Placement2D(
                model.createIfcCartesianPoint((length / 2, 0.0)), None
            ),
            length, thickness,
        )
        solid = model.createIfcExtrudedAreaSolid(
            profile,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((1.0, 0.0, 0.0)),
            ),
            model.createIfcDirection((0.0, 0.0, 1.0)),
            w.height,
        )
        shape = model.createIfcShapeRepresentation(
            body_ctx, "Body", "SweptSolid", [solid]
        )
        prod_repr = model.createIfcProductDefinitionShape(None, None, [shape])

        ifc_wall = model.createIfcWall(
            new_guid(), owner, w.name, f"Wall L={length:.2f}m H={w.height:.2f}m",
            None, placement, prod_repr, None,
        )
        ifc_walls[w.name] = ifc_wall
        elements.append(ifc_wall)
        print(f"  ✓ {w.name}: L={length:.2f}m, H={w.height:.2f}m, "
              f"angle={math.degrees(angle_rad):.1f}°")

    # Lookup direction et épaisseur de chaque mur
    wall_dir_map:       dict[str, tuple[float, float]] = {}
    wall_thickness_map: dict[str, float]               = {}
    for w in walls:
        dx = w.x2 - w.x1;  dy = w.y2 - w.y1
        ln = math.sqrt(dx*dx + dy*dy)
        wall_dir_map[w.name]       = (dx/ln, dy/ln) if ln > 1e-6 else (1.0, 0.0)
        wall_thickness_map[w.name] = w.thickness if w.thickness > 0.001 else 0.2

    # ─────────────────────────────────────────
    #  PORTES avec vides sémantiques
    # ─────────────────────────────────────────
    for d in doors:
        parent_wall = ifc_walls.get(d.parent_wall)
        if not parent_wall:
            print(f"  ⚠ {d.name}: mur parent {d.parent_wall} introuvable")
            continue
        
        dir_x, dir_y = wall_dir_map.get(d.parent_wall, (1.0, 0.0))
        wall_t = wall_thickness_map.get(d.parent_wall, 0.2)
        
        # ── Créer l'ouverture (IfcOpeningElement) ──
        opening_placement = model.createIfcLocalPlacement(
            storey_placement,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((d.cx, d.cy, d.cz - d.height / 2)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((dir_x, dir_y, 0.0)),
            ),
        )
        
        opening_profile = model.createIfcRectangleProfileDef(
            "AREA", None,
            model.createIfcAxis2Placement2D(
                model.createIfcCartesianPoint((0.0, 0.0)), None
            ),
            d.width, wall_t + 0.02,  # légèrement plus épais pour garantir la coupe
        )
        opening_solid = model.createIfcExtrudedAreaSolid(
            opening_profile,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((1.0, 0.0, 0.0)),
            ),
            model.createIfcDirection((0.0, 0.0, 1.0)),
            d.height,
        )
        opening_shape = model.createIfcShapeRepresentation(
            body_ctx, "Body", "SweptSolid", [opening_solid]
        )
        opening_repr = model.createIfcProductDefinitionShape(None, None, [opening_shape])
        
        opening = model.createIfcOpeningElement(
            new_guid(), owner, f"{d.name}_opening", f"Opening for {d.name}",
            None, opening_placement, opening_repr, None,
        )
        
        # ── Créer le vide dans le mur (IfcRelVoidsElement) ──
        model.createIfcRelVoidsElement(
            new_guid(), owner, None, None, parent_wall, opening
        )
        
        # ── Créer la porte (IfcDoor) ──
        door_placement = model.createIfcLocalPlacement(
            storey_placement,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((d.cx, d.cy, d.cz - d.height / 2)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((dir_x, dir_y, 0.0)),
            ),
        )
        
        door_profile = model.createIfcRectangleProfileDef(
            "AREA", None,
            model.createIfcAxis2Placement2D(
                model.createIfcCartesianPoint((0.0, 0.0)), None
            ),
            d.width, wall_t,
        )
        door_solid = model.createIfcExtrudedAreaSolid(
            door_profile,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((1.0, 0.0, 0.0)),
            ),
            model.createIfcDirection((0.0, 0.0, 1.0)),
            d.height,
        )
        door_shape = model.createIfcShapeRepresentation(
            body_ctx, "Body", "SweptSolid", [door_solid]
        )
        door_repr = model.createIfcProductDefinitionShape(None, None, [door_shape])
        
        ifc_door = model.createIfcDoor(
            new_guid(), owner, d.name, f"Door W={d.width:.2f}m H={d.height:.2f}m",
            None, door_placement, door_repr, None, d.height, d.width,
        )
        
        # ── Lier la porte à l'ouverture (IfcRelFillsElement) ──
        model.createIfcRelFillsElement(
            new_guid(), owner, None, None, opening, ifc_door
        )
        
        elements.append(ifc_door)
        print(f"  ✓ {d.name}: parent={d.parent_wall}, W={d.width:.2f}m, H={d.height:.2f}m "
              f"(avec vide sémantique)")

    # ─────────────────────────────────────────
    #  FENÊTRES avec vides sémantiques
    # ─────────────────────────────────────────
    for wn in windows:
        parent_wall = ifc_walls.get(wn.parent_wall)
        if not parent_wall:
            print(f"  ⚠ {wn.name}: mur parent {wn.parent_wall} introuvable")
            continue
        
        dir_x, dir_y = wall_dir_map.get(wn.parent_wall, (1.0, 0.0))
        wall_t = wall_thickness_map.get(wn.parent_wall, 0.2)
        
        # ── Créer l'ouverture ──
        opening_placement = model.createIfcLocalPlacement(
            storey_placement,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((wn.cx, wn.cy, wn.cz - wn.height / 2)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((dir_x, dir_y, 0.0)),
            ),
        )
        
        opening_profile = model.createIfcRectangleProfileDef(
            "AREA", None,
            model.createIfcAxis2Placement2D(
                model.createIfcCartesianPoint((0.0, 0.0)), None
            ),
            wn.width, wall_t + 0.02,
        )
        opening_solid = model.createIfcExtrudedAreaSolid(
            opening_profile,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((1.0, 0.0, 0.0)),
            ),
            model.createIfcDirection((0.0, 0.0, 1.0)),
            wn.height,
        )
        opening_shape = model.createIfcShapeRepresentation(
            body_ctx, "Body", "SweptSolid", [opening_solid]
        )
        opening_repr = model.createIfcProductDefinitionShape(None, None, [opening_shape])
        
        opening = model.createIfcOpeningElement(
            new_guid(), owner, f"{wn.name}_opening", f"Opening for {wn.name}",
            None, opening_placement, opening_repr, None,
        )
        
        # ── Créer le vide dans le mur ──
        model.createIfcRelVoidsElement(
            new_guid(), owner, None, None, parent_wall, opening
        )
        
        # ── Créer la fenêtre ──
        window_placement = model.createIfcLocalPlacement(
            storey_placement,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((wn.cx, wn.cy, wn.cz - wn.height / 2)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((dir_x, dir_y, 0.0)),
            ),
        )
        
        window_profile = model.createIfcRectangleProfileDef(
            "AREA", None,
            model.createIfcAxis2Placement2D(
                model.createIfcCartesianPoint((0.0, 0.0)), None
            ),
            wn.width, wall_t,
        )
        window_solid = model.createIfcExtrudedAreaSolid(
            window_profile,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((1.0, 0.0, 0.0)),
            ),
            model.createIfcDirection((0.0, 0.0, 1.0)),
            wn.height,
        )
        window_shape = model.createIfcShapeRepresentation(
            body_ctx, "Body", "SweptSolid", [window_solid]
        )
        window_repr = model.createIfcProductDefinitionShape(None, None, [window_shape])
        
        ifc_window = model.createIfcWindow(
            new_guid(), owner, wn.name, f"Window W={wn.width:.2f}m H={wn.height:.2f}m",
            None, window_placement, window_repr, None, wn.height, wn.width,
        )
        
        # ── Lier la fenêtre à l'ouverture ──
        model.createIfcRelFillsElement(
            new_guid(), owner, None, None, opening, ifc_window
        )
        
        elements.append(ifc_window)
        print(f"  ✓ {wn.name}: parent={wn.parent_wall}, W={wn.width:.2f}m, H={wn.height:.2f}m "
              f"(avec vide sémantique)")

    # ─────────────────────────────────────────
    #  OBJETS (Bbox) → IfcFurnishingElement
    # ─────────────────────────────────────────
    for b in bboxes:
        cos_r = math.cos(b.rotation)
        sin_r = math.sin(b.rotation)

        placement = model.createIfcLocalPlacement(
            storey_placement,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((b.cx, b.cy, b.cz - b.height / 2)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((cos_r, sin_r, 0.0)),
            ),
        )
        profile = model.createIfcRectangleProfileDef(
            "AREA", None,
            model.createIfcAxis2Placement2D(
                model.createIfcCartesianPoint((0.0, 0.0)), None
            ),
            b.length, b.width,
        )
        solid = model.createIfcExtrudedAreaSolid(
            profile,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((1.0, 0.0, 0.0)),
            ),
            model.createIfcDirection((0.0, 0.0, 1.0)),
            b.height,
        )
        shape = model.createIfcShapeRepresentation(body_ctx, "Body", "SweptSolid", [solid])
        prod_repr = model.createIfcProductDefinitionShape(None, None, [shape])

        ifc_obj = model.createIfcFurnishingElement(
            new_guid(), owner, b.name, b.obj_class,
            None, placement, prod_repr, None,
        )
        elements.append(ifc_obj)

    # ── Contenir tous les éléments dans l'étage ──
    if elements:
        model.createIfcRelContainedInSpatialStructure(
            new_guid(), owner, "Ground Floor Elements", None,
            elements, storey,
        )

    # ── Sauvegarde ──
    model.write(output_path)
    print(f"\n✅ IFC sauvegardé : {output_path}")
    print(f"   Éléments : 1 dalle + {len(walls)} murs, {len(doors)} portes, "
          f"{len(windows)} fenêtres, {len(bboxes)} objets")
    print(f"   Vides sémantiques créés : {len(doors) + len(windows)} IfcOpeningElement")


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convertit la sortie TXT de SpatialLM en IFC avec vides sémantiques"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Fichier TXT de sortie SpatialLM")
    parser.add_argument("--output", "-o", default="output.ifc",
                        help="Fichier IFC de sortie (défaut: output.ifc)")
    args = parser.parse_args()

    print(f"\n── Parsing {args.input} ──")
    walls, doors, windows, bboxes = parse_spatiallm_txt(args.input)

    print(f"\n── Construction du modèle IFC avec vides sémantiques ──")
    build_ifc(walls, doors, windows, bboxes, args.output)


if __name__ == "__main__":
    main()