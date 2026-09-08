"""
generate_schemas.py
--------------------
Genere les schemas explicatifs du pipeline SpatialLM -> IFC pour le rapport PFE
et la soutenance : architecture du pipeline, architecture du modele SpatialLM,
et conversion vers IFC (format + modele semantique).

Style volontairement sobre (noir/blanc + une seule couleur d'accentuation) pour
un rendu de figure technique classique, sans decor ni texte de titre/legende
integre a l'image (les titres et legendes se mettent dans le corps du rapport).

Sortie : un fichier par schema, dans plusieurs formats, sous schemas_output/ :
  - .pdf  : a inserer dans LaTeX (\\includegraphics), vectoriel
  - .svg  : vectoriel, editable (Inkscape, PowerPoint recent)
  - .png  : raster haute resolution (300 dpi), pour PowerPoint / Beamer

Usage :
    python generate_schemas.py
    python generate_schemas.py --outdir figures --formats pdf,png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# Style — sobre et monochrome, une seule couleur d'accentuation reservee
# au mecanisme mis en avant dans chaque schema.
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
GRAY = "#5c5c5c"
ACCENT = "#2f4f6b"
WHITE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

MM = 1 / 25.4


def line_h(fontsize_pt):
    return fontsize_pt * 0.3528 * 1.4


def new_canvas(w_mm, h_mm):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(h_mm, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.patch.set_visible(False)
    return fig, ax


def box(
    ax,
    x,
    y,
    w,
    h,
    title=None,
    mono=None,
    desc=None,
    edge=INK,
    lw=1.0,
    title_size=8.6,
    mono_size=7.2,
    desc_size=7.2,
    pad=3.2,
    align="left",
    valign="center",
):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=1.4",
        linewidth=lw,
        edgecolor=edge,
        facecolor=WHITE,
        zorder=2,
    )
    ax.add_patch(rect)

    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + pad

    n_mono = len(mono) if isinstance(mono, (list, tuple)) else (1 if mono else 0)
    n_desc = len(desc) if isinstance(desc, (list, tuple)) else (1 if desc else 0)
    block_h = (
        (line_h(title_size) if title else 0)
        + n_mono * line_h(mono_size)
        + n_desc * line_h(desc_size)
    )
    if valign == "top":
        # anchor the text block to the top of the box, leaving the box's
        # vertical center free — used where an arrow must cross the box
        # midline (e.g. a horizontal chain of tall, multi-line boxes).
        ty = y + pad + line_h(title_size) * 0.35
    else:
        ty = y + (h - block_h) / 2 + line_h(title_size) * 0.35

    if title:
        ax.text(tx, ty, title, fontsize=title_size, color=INK, weight="bold", ha=ha, va="center", zorder=3)
        ty += line_h(title_size)
    if mono:
        for line in mono if isinstance(mono, (list, tuple)) else [mono]:
            ax.text(tx, ty, line, fontsize=mono_size, color=GRAY, family="monospace", ha=ha, va="center", zorder=3)
            ty += line_h(mono_size)
    if desc:
        for line in desc if isinstance(desc, (list, tuple)) else [desc]:
            ax.text(tx, ty, line, fontsize=desc_size, color=GRAY, ha=ha, va="center", zorder=3)
            ty += line_h(desc_size)


def arrow(
    ax,
    p1,
    p2,
    label=None,
    color=INK,
    lw=1.1,
    dashed=False,
    curve=0.0,
    label_size=7.0,
    label_dy=-1.6,
    label_pos=0.5,
):
    arr = patches.FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        linestyle="dashed" if dashed else "solid",
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos
        my = p1[1] + (p2[1] - p1[1]) * label_pos
        ax.text(mx, my + label_dy, label, fontsize=label_size, color=color, family="monospace", ha="center", va="bottom", zorder=3)


def save(fig, outdir, name, formats):
    for fmt in formats:
        path = os.path.join(outdir, f"{name}.{fmt}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Schema 1 — architecture generale du pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    W, H = 300, 72
    fig, ax = new_canvas(W, H)
    y, h = 12, 48
    cy = y + h / 2

    specs = [
        (4, 46, "Nuage de points brut", ".ply", ["Scan 3D brut", "(LiDAR / photogrammetrie)"]),
        (56, 46, "Pretraitement", ["preprocess_for_", "spatiallm.py"], ["Filtrage statistique + DBSCAN", "+ normalisation de hauteur"]),
        (108, 46, "Inference SpatialLM", "inference.py", ["SpatialLM1.1", "(Llama-1B ou Qwen-0.5B)"]),
        (160, 46, "Post-traitement", "postprocess.py", ["Fusion des murs colineaires", "+ reajustement des bbox"]),
        (212, 46, "Generation IFC", ["spatiallm_to_ifc.py"], ["ifcopenshell : dalle, murs,", "ouvertures, mobilier"]),
    ]
    for x, w, title, mono, desc in specs:
        box(ax, x, y, w, h, title=title, mono=mono, desc=desc, valign="top")

    final_x, final_w = 268, 28
    box(ax, final_x, y + 10, final_w, h - 20, title="Modele IFC", mono=".ifc", lw=1.6, align="center")

    files = ["scan.ply", "scan_clean.ply", "raw_layout.txt", "refined_layout.txt"]
    edges = [(50, 56), (102, 108), (154, 160), (206, 212)]
    for (x1, x2), f in zip(edges, files):
        arrow(ax, (x1, cy), (x2, cy), label=f)

    arrow(ax, (258, cy), (final_x, cy), lw=1.5, color=ACCENT)

    return fig


# ---------------------------------------------------------------------------
# Schema 2 — architecture du modele SpatialLM
# ---------------------------------------------------------------------------
def fig_spatiallm_architecture():
    W, H = 202, 234
    fig, ax = new_canvas(W, H)

    col1_x, col1_w = 6, 90
    col2_x, col2_w = 104, 92
    cx1 = col1_x + col1_w / 2
    cx2 = col2_x + col2_w / 2

    # Level 1 — raw inputs
    y1, h1 = 8, 34
    box(ax, col1_x, y1, col1_w, h1, title="Nuage de points pretraite",
        desc=["Coordonnees (x, y, z), couleur RGB,", "grille de voxels (grid_coord)"],
        mono="point_cloud : (n_points, n_features)")
    box(ax, col2_x, y1, col2_w, h1, title="Prompt texte",
        mono=["<point_start><point_pad><point_end>", "Detect walls, doors, windows, boxes.", "Reference code: {code_template}"])

    # Level 2 — encoders
    y2, h2 = y1 + h1 + 8, 28
    box(ax, col1_x, y2, col1_w, h2, title="Encodeur de nuage de points",
        mono=["Sonata (point transformer)", "ou SceneScript (sparse conv)"])
    box(ax, col2_x, y2, col2_w, h2, title="Tokenizer",
        desc=["Decoupe le prompt en tokens du", "vocabulaire Llama / Qwen"])

    # Level 3 — projection
    y3, h3 = y2 + h2 + 8, 26
    box(ax, col1_x, y3, col1_w, h3, title="Projecteur (lineaire ou MLP)",
        desc=["Projette les features de points vers", "la dimension cachee du LLM"], lw=1.4)
    box(ax, col2_x, y3, col2_w, h3, title="Embeddings de tokens texte",
        desc="Une ligne d'embedding par token du prompt")

    # Level 4 — merged sequence
    y4, h4 = y3 + h3 + 10, 26
    seg1_x, seg1_w = 6, 56
    seg2_x, seg2_w = 66, 70
    seg3_x, seg3_w = 140, 56
    box(ax, seg1_x, y4, seg1_w, h4, desc=["Debut de la", "conversation"], align="center")
    box(ax, seg2_x, y4, seg2_w, h4, title="Tokens de points",
        mono="n tokens, remplacent <point_pad>", edge=ACCENT, lw=1.7, align="center")
    box(ax, seg3_x, y4, seg3_w, h4, desc=["Consigne + template", "de code (texte)"], align="center")

    # Level 5 — LLM backbone
    y5, h5 = y4 + h4 + 10, 28
    box(ax, 6, y5, 190, h5, title="Modele de langage (LLM)",
        mono="Llama-1B ou Qwen-0.5B", desc="Decodeur transformer, attention causale")

    # Level 6 — generation + output
    y6, h6 = y5 + h5 + 10, 32
    box(ax, col1_x, y6, col1_w, h6, title="Generation token par token",
        desc=["lm_head + decodage glouton", "(do_sample = False)"])
    box(ax, col2_x, y6, col2_w, h6, title="Code de layout",
        mono=["wall_0=Wall(...)", "door_0=Door(...)", "bbox_0=Bbox(...)"], edge=ACCENT, lw=1.5)

    # Arrows
    arrow(ax, (cx1, y1 + h1), (cx1, y2))
    arrow(ax, (cx2, y1 + h1), (cx2, y2))
    arrow(ax, (cx1, y2 + h2), (cx1, y3))
    arrow(ax, (cx2, y2 + h2), (cx2, y3))
    arrow(ax, (cx1, y3 + h3), (seg2_x + seg2_w / 2, y4), color=ACCENT, lw=1.4)
    arrow(ax, (cx2, y3 + h3), (seg3_x + seg3_w / 2, y4))
    mid4 = (seg1_x + seg3_x + seg3_w) / 2
    arrow(ax, (mid4, y4 + h4), (mid4, y5))
    arrow(ax, (col1_x + col1_w / 2, y5 + h5), (col1_x + col1_w / 2, y6))
    arrow(ax, (col1_x + col1_w, y6 + h6 / 2), (col2_x, y6 + h6 / 2), color=ACCENT, lw=1.4)

    return fig


# ---------------------------------------------------------------------------
# Schema 3 — conversion IFC : du texte SpatialLM aux entites IFC
# ---------------------------------------------------------------------------
def fig_ifc_format():
    W, H = 300, 190
    fig, ax = new_canvas(W, H)

    c1x, c1w = 6, 96
    c2x, c2w = 112, 62
    c3x, c3w = 184, 110
    rows_y = [6, 52, 98, 144]
    row_h = 36

    rows = [
        {
            "txt": ["wall_0=Wall(x1,y1,z1,", "  x2,y2,z2, h, epaisseur)"],
            "cls": "WallData",
            "cls_desc": ["Extremites, hauteur,", "epaisseur"],
            "ifc": "IfcWall",
            "ifc_desc": ["Solide extrude, place et oriente", "dans l'etage"],
        },
        {
            "txt": ["door_0=Door(wall_0,", "  cx,cy,cz, largeur, h)"],
            "cls": "DoorData",
            "cls_desc": "parent_wall = wall_0",
            "ifc": "IfcOpeningElement + IfcDoor",
            "ifc_desc": ["Vide perce dans IfcWall,", "porte inseree dans le vide"],
        },
        {
            "txt": ["window_0=Window(wall_2,", "  cx,cy,cz, largeur, h)"],
            "cls": "WindowData",
            "cls_desc": "parent_wall = wall_2",
            "ifc": "IfcOpeningElement + IfcWindow",
            "ifc_desc": ["Meme mecanisme de vide", "que pour une porte"],
        },
        {
            "txt": ["bbox_0=Bbox(classe,cx,", "  cy,cz, rot, l,p,h)"],
            "cls": "BboxData",
            "cls_desc": "Classe, pose, dimensions",
            "ifc": "IfcFurnishingElement",
            "ifc_desc": "Mobilier / objets detectes",
        },
    ]

    for ry, row in zip(rows_y, rows):
        box(ax, c1x, ry, c1w, row_h, mono=row["txt"], mono_size=7.4)
        box(ax, c2x, ry, c2w, row_h, title=row["cls"], desc=row["cls_desc"], title_size=8, desc_size=6.9)
        box(ax, c3x, ry, c3w, row_h, title=row["ifc"], desc=row["ifc_desc"], title_size=8, desc_size=7)
        cy = ry + row_h / 2
        arrow(ax, (c1x + c1w, cy), (c2x, cy))
        arrow(ax, (c2x + c2w, cy), (c3x, cy))

    # mechanism: DoorData looks back up at WallData (parent wall lookup)
    wall_bottom = rows_y[0] + row_h
    door_top = rows_y[1]
    link_x = c2x + 10
    arrow(ax, (link_x, door_top), (link_x, wall_bottom), color=ACCENT, dashed=True, lw=1.2)
    ax.text(
        link_x + 6, (wall_bottom + door_top) / 2, "wall_dir_map / wall_thickness_map",
        fontsize=6.3, color=ACCENT, family="monospace", ha="left", va="center", zorder=3,
    )

    return fig


# ---------------------------------------------------------------------------
# Schema 4 — conversion IFC : hierarchie spatiale et vides semantiques
# ---------------------------------------------------------------------------
def fig_ifc_semantic():
    W, H = 300, 175
    fig, ax = new_canvas(W, H)

    # --- spatial hierarchy chain ---
    y1, h1 = 6, 26
    chain = [
        (4, 58, "IfcProject", "SpatialLM Project"),
        (72, 58, "IfcSite", "Origine (0,0,0)"),
        (140, 58, "IfcBuilding", None),
        (208, 78, "IfcBuildingStorey", "Ground Floor"),
    ]
    for x, w, title, mono in chain:
        box(ax, x, y1, w, h1, title=title, mono=mono, align="center", title_size=8.2)
    for i in range(len(chain) - 1):
        x1e = chain[i][0] + chain[i][1]
        x2s = chain[i + 1][0]
        arrow(ax, (x1e, y1 + h1 / 2), (x2s, y1 + h1 / 2))

    storey_cx = chain[-1][0] + chain[-1][1] / 2
    y_trunk = y1 + h1 + 14
    arrow(ax, (storey_cx, y1 + h1), (storey_cx, y_trunk), color=ACCENT)

    y2, h2 = y_trunk + 12, 30
    elems = [
        (18, 52, "IfcSlab"),
        (78, 52, "IfcWall (n)"),
        (138, 60, "IfcDoor / IfcWindow"),
        (206, 52, "Furnishing (n)"),
    ]
    elem_cxs = [x + w / 2 for x, w, _ in elems]
    x_lo, x_hi = min(elem_cxs), max(elem_cxs)
    ax.plot([x_lo, x_hi], [y_trunk, y_trunk], color=ACCENT, linewidth=1.3, zorder=1)
    ax.text((x_lo + x_hi) / 2, y_trunk - 2.2, "IfcRelContainedInSpatialStructure",
            fontsize=7, color=ACCENT, family="monospace", ha="center", va="bottom", zorder=3)
    for x, w, title in elems:
        cx = x + w / 2
        arrow(ax, (cx, y_trunk), (cx, y2), color=ACCENT)
        box(ax, x, y2, w, h2, title=title, align="center", title_size=8)

    # --- opening mechanism detail ---
    y3, h3 = y2 + h2 + 24, 46
    wall_x, wall_w = 8, 78
    open_x, open_w = 118, 84
    door_x, door_w = 234, 58

    box(ax, wall_x, y3, wall_w, h3, title="IfcWall",
        desc=["Mur porteur, geometrie", "pleine, sans decoupe"], title_size=9.5)
    box(ax, open_x, y3, open_w, h3, title="IfcOpeningElement",
        desc=["Volume mediateur du vide,", "non affiche seul"], edge=ACCENT, lw=1.7, title_size=9.5)
    box(ax, door_x, y3, door_w, h3, title="IfcDoor",
        desc=["Feuille de porte,", "inseree dans le vide"], title_size=9.5)

    cy3 = y3 + h3 / 2
    arrow(ax, (wall_x + wall_w, cy3), (open_x, cy3), label="IfcRelVoidsElement", color=ACCENT, lw=1.5)
    arrow(ax, (door_x, cy3), (open_x + open_w, cy3), label="IfcRelFillsElement", color=ACCENT, lw=1.5)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
FIGURES = {
    "schema_1_pipeline": fig_pipeline,
    "schema_2_architecture_spatiallm": fig_spatiallm_architecture,
    "schema_3_conversion_ifc_format": fig_ifc_format,
    "schema_4_conversion_ifc_semantique": fig_ifc_semantic,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="schemas_output", help="Dossier de sortie (defaut: schemas_output)")
    parser.add_argument("--formats", default="pdf,svg,png", help="Formats separes par des virgules (defaut: pdf,svg,png)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    for name, builder in FIGURES.items():
        print(f"Generation : {name}")
        fig = builder()
        save(fig, args.outdir, name, formats)


if __name__ == "__main__":
    main()
