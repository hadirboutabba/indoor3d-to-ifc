"""
Interface Streamlit — Pipeline complet PLY → IFC.

Workflow :
  1. Upload .ply
  2. Preprocessing    (nettoyage + alignement)
  3. Inference        (SpatialLM → layout TXT brut)
  4. Post-processing  (fusion murs colinéaires + affinage bbox)
  5. IFC              (TXT raffiné → fichier BIM)
  + Visualisation 3D maquette embarquée + viewer Rerun

Lancement :
    cd /mnt/c/Users/hboutabb/SpatialLM
    streamlit run pipeline_gui.py
"""

import os
import subprocess
import tempfile
import time
import math
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =============================================================================
# CONFIGURATION
# =============================================================================
SPATIALLM_DIR  = "/mnt/c/Users/hboutabb/SpatialLM"
RERUN_WEB_PORT = 9090

MODELS = {
    "SpatialLM1.1 — Llama 1B  (recommandé, précis)":  "/mnt/c/Users/hboutabb/SpatialLM1.1-Llama-1B",
    "SpatialLM1.1 — Qwen 0.5B (rapide, léger)":       "/mnt/c/Users/hboutabb/SpatialLM1.1-Qwen-0.5B",
}


# =============================================================================
# 3D LAYOUT MAQUETTE (Plotly) — réutilisé depuis spatiallm_gui.py
# =============================================================================

def parse_layout_for_3d(layout_text):
    walls, doors, windows, bboxes = [], [], [], []
    wall_map = {}
    for line in layout_text.strip().splitlines():
        m = re.match(r'(wall_\w+)=Wall\(([^)]+)\)', line)
        if m:
            vals = list(map(float, m.group(2).split(',')))
            w = {'x1': vals[0], 'y1': vals[1], 'z1': vals[2],
                 'x2': vals[3], 'y2': vals[4], 'z2': vals[5], 'height': vals[6]}
            walls.append(w)
            wall_map[m.group(1)] = w
            continue
        m = re.match(r'door_\w+=Door\(([^,]+),([^)]+)\)', line)
        if m:
            vals = list(map(float, m.group(2).split(',')))
            doors.append({'wall_ref': m.group(1).strip(),
                          'cx': vals[0], 'cy': vals[1], 'cz': vals[2],
                          'width': vals[3], 'height': vals[4]})
            continue
        m = re.match(r'window_\w+=Window\(([^,]+),([^)]+)\)', line)
        if m:
            vals = list(map(float, m.group(2).split(',')))
            windows.append({'wall_ref': m.group(1).strip(),
                            'cx': vals[0], 'cy': vals[1], 'cz': vals[2],
                            'width': vals[3], 'height': vals[4]})
            continue
        m = re.match(r'bbox_\w+=Bbox\(([^,]+),([^)]+)\)', line)
        if m:
            vals = list(map(float, m.group(2).split(',')))
            bboxes.append({'label': m.group(1),
                           'cx': vals[0], 'cy': vals[1], 'cz': vals[2],
                           'angle': vals[3],
                           'dx': vals[4], 'dy': vals[5], 'dz': vals[6]})
    return walls, wall_map, doors, windows, bboxes


def _wall_dir(wall):
    dx = wall['x2'] - wall['x1']
    dy = wall['y2'] - wall['y1']
    length = math.sqrt(dx * dx + dy * dy) or 1.0
    return dx / length, dy / length


def _quad_mesh(fig, xs4, ys4, zs4, color, opacity, lighting=None):
    kw = dict(flatshading=True, showlegend=False, hoverinfo='skip')
    if lighting:
        kw['lighting'] = lighting
    fig.add_trace(go.Mesh3d(
        x=xs4, y=ys4, z=zs4,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color, opacity=opacity, **kw,
    ))


def _quad_wire(fig, xs4, ys4, zs4, color, width=1.5):
    fig.add_trace(go.Scatter3d(
        x=list(xs4) + [xs4[0]], y=list(ys4) + [ys4[0]], z=list(zs4) + [zs4[0]],
        mode='lines', line=dict(color=color, width=width),
        showlegend=False, hoverinfo='skip',
    ))


_OBJ_PALETTE = [
    ('#e07828', '#a05010'), ('#2eaa44', '#1a7730'),
    ('#cc2222', '#991111'), ('#8833cc', '#661199'),
    ('#cc8800', '#996600'), ('#cc2299', '#991177'),
    ('#0099bb', '#006688'), ('#aa22cc', '#771199'),
]


def build_3d_figure(walls, wall_map, doors, windows, bboxes):
    fig = go.Figure()
    if not walls:
        return fig

    zb_global = min(min(w['z1'], w['z2']) for w in walls)
    zt_global = max(min(w['z1'], w['z2']) + w['height'] for w in walls)

    fxs = [w['x1'] for w in walls] + [w['x2'] for w in walls]
    fys = [w['y1'] for w in walls] + [w['y2'] for w in walls]
    fzs = [zb_global] * len(fxs)
    fig.add_trace(go.Mesh3d(x=fxs, y=fys, z=fzs, alphahull=0,
        color='#7a6a58', opacity=0.90, showlegend=False, hoverinfo='skip',
        flatshading=True, lighting=dict(ambient=0.95, diffuse=0.2)))
    fig.add_trace(go.Mesh3d(x=fxs, y=fys, z=[zt_global]*len(fxs), alphahull=0,
        color='#f0ece4', opacity=0.12, showlegend=False, hoverinfo='skip'))

    wall_light = dict(ambient=0.65, diffuse=0.85, specular=0.08, roughness=0.8)
    for wall in walls:
        x1, y1, z1 = wall['x1'], wall['y1'], wall['z1']
        x2, y2, z2 = wall['x2'], wall['y2'], wall['z2']
        zb = min(z1, z2); zt = zb + wall['height']
        _quad_mesh(fig, [x1,x2,x2,x1], [y1,y2,y2,y1], [zb,zb,zt,zt], '#ede5d8', 0.90, wall_light)
        _quad_wire(fig, [x1,x2,x2,x1], [y1,y2,y2,y1], [zb,zb,zt,zt], '#b8ad9e', 1.2)

    door_light = dict(ambient=0.6, diffuse=0.9, specular=0.05)
    for door in doors:
        cx, cy, cz = door['cx'], door['cy'], door['cz']
        w, h = door['width'], door['height']
        ddx, ddy = _wall_dir(wall_map[door['wall_ref']]) if door['wall_ref'] in wall_map else (1.0, 0.0)
        hw = w / 2
        _quad_mesh(fig, [cx-hw*ddx, cx+hw*ddx, cx+hw*ddx, cx-hw*ddx],
                        [cy-hw*ddy, cy+hw*ddy, cy+hw*ddy, cy-hw*ddy],
                        [cz-h/2, cz-h/2, cz+h/2, cz+h/2], '#8b6343', 0.82, door_light)
        _quad_wire(fig, [cx-hw*ddx, cx+hw*ddx, cx+hw*ddx, cx-hw*ddx],
                        [cy-hw*ddy, cy+hw*ddy, cy+hw*ddy, cy-hw*ddy],
                        [cz-h/2, cz-h/2, cz+h/2, cz+h/2], '#5c3d20', 2.0)

    win_light = dict(ambient=0.5, diffuse=0.7, specular=0.5, roughness=0.1)
    for win in windows:
        cx, cy, cz = win['cx'], win['cy'], win['cz']
        w, h = win['width'], win['height']
        wdx, wdy = _wall_dir(wall_map[win['wall_ref']]) if win['wall_ref'] in wall_map else (1.0, 0.0)
        hw = w / 2
        _quad_mesh(fig, [cx-hw*wdx, cx+hw*wdx, cx+hw*wdx, cx-hw*wdx],
                        [cy-hw*wdy, cy+hw*wdy, cy+hw*wdy, cy-hw*wdy],
                        [cz-h/2, cz-h/2, cz+h/2, cz+h/2], '#87ceeb', 0.45, win_light)
        _quad_wire(fig, [cx-hw*wdx, cx+hw*wdx, cx+hw*wdx, cx-hw*wdx],
                        [cy-hw*wdy, cy+hw*wdy, cy+hw*wdy, cy-hw*wdy],
                        [cz-h/2, cz-h/2, cz+h/2, cz+h/2], '#5599cc', 2.0)

    BOX_I = [0,0, 1,1, 0,0, 2,2, 0,0, 4,4]
    BOX_J = [2,4, 3,5, 1,4, 3,6, 1,2, 5,6]
    BOX_K = [6,6, 7,7, 5,5, 7,7, 3,3, 7,7]
    BOX_EDGES = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    label_colors, seen, cidx = {}, set(), 0
    obj_light = dict(ambient=0.6, diffuse=0.85, specular=0.1)

    for bbox in bboxes:
        label = bbox['label']
        if label not in label_colors:
            label_colors[label] = _OBJ_PALETTE[cidx % len(_OBJ_PALETTE)]
            cidx += 1
        fill_col, edge_col = label_colors[label]
        cx, cy, cz = bbox['cx'], bbox['cy'], bbox['cz']
        ang = bbox['angle']
        hdx, hdy, hdz = bbox['dx']/2, bbox['dy']/2, bbox['dz']/2
        ca, sa = math.cos(ang), math.sin(ang)
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    lx, ly = sx*hdx, sy*hdy
                    corners.append((cx + ca*lx - sa*ly, cy + sa*lx + ca*ly, cz + sz*hdz))
        cxs = [c[0] for c in corners]
        cys = [c[1] for c in corners]
        czs = [c[2] for c in corners]
        fig.add_trace(go.Mesh3d(x=cxs, y=cys, z=czs, i=BOX_I, j=BOX_J, k=BOX_K,
            color=fill_col, opacity=0.32, showlegend=False, hoverinfo='skip',
            flatshading=True, lighting=obj_light))
        ex, ey, ez = [], [], []
        for ai, bi in BOX_EDGES:
            ex += [corners[ai][0], corners[bi][0], None]
            ey += [corners[ai][1], corners[bi][1], None]
            ez += [corners[ai][2], corners[bi][2], None]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines',
            line=dict(color=edge_col, width=2), name=label.replace('_',' '),
            legendgroup=label, showlegend=label not in seen, hoverinfo='name'))
        seen.add(label)
        fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[cz+hdz+0.06], mode='text',
            text=[label.replace('_',' ')],
            textfont=dict(size=9, color='rgba(255,255,255,0.85)'),
            showlegend=False, hoverinfo='skip'))

    all_x = [w['x1'] for w in walls] + [w['x2'] for w in walls]
    all_y = [w['y1'] for w in walls] + [w['y2'] for w in walls]
    span = max(max(all_x)-min(all_x), max(all_y)-min(all_y), 1.0)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', gridcolor='#2d2d2d', backgroundcolor='#181818',
                       showbackground=True, zerolinecolor='#3a3a3a'),
            yaxis=dict(title='Y (m)', gridcolor='#2d2d2d', backgroundcolor='#181818',
                       showbackground=True, zerolinecolor='#3a3a3a'),
            zaxis=dict(title='Z (m)', gridcolor='#2d2d2d', backgroundcolor='#121212',
                       showbackground=True, zerolinecolor='#3a3a3a'),
            bgcolor='#141414', aspectmode='data',
            camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=-0.1),
                        eye=dict(x=1.6*span/4, y=-1.6*span/4, z=1.1*span/4)),
        ),
        paper_bgcolor='#0e1117', font=dict(color='#dddddd', size=11),
        margin=dict(l=0, r=0, t=10, b=0), height=560,
        legend=dict(x=0, y=1, bgcolor='rgba(15,15,15,0.88)',
                    bordercolor='#444', borderwidth=1,
                    font=dict(color='#cccccc', size=10)),
    )
    return fig


# =============================================================================
# HELPERS
# =============================================================================

def run_command(cmd, label):
    """Lance une commande shell et streame le log dans Streamlit."""
    log_box = st.empty()
    full_log = ""
    start = time.time()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=SPATIALLM_DIR,
    )
    for line in process.stdout:
        full_log += line
        log_box.code(full_log[-3000:], language="bash")
    process.wait()
    elapsed = time.time() - start
    if process.returncode == 0:
        st.success(f"✅ {label} terminé en {elapsed:.1f}s")
    else:
        st.error(f"❌ {label} a échoué (code {process.returncode})")
    return process.returncode == 0, elapsed


def layout_stats(text):
    return {
        "Murs":     text.count("=Wall("),
        "Portes":   text.count("=Door("),
        "Fenêtres": text.count("=Window("),
        "Objets":   text.count("=Bbox("),
    }


# =============================================================================
# PAGE
# =============================================================================

st.set_page_config(page_title="PLY → IFC Pipeline", page_icon="🏗️", layout="wide")

st.title("🏗️ Pipeline PLY → IFC")
st.caption("Upload un .ply → Preprocess → SpatialLM → Post-process → IFC")


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")

    # ── 1. Preprocessing ──
    st.subheader("1 · Preprocessing")
    preprocess_mode = st.radio(
        "Mode",
        ["🔧 Complet", "📐 Alignement uniquement", "⏭️ Ignorer"],
        index=0,
        help=(
            "**Complet** — dénoise + alignement Z-up/Manhattan + échelle\n\n"
            "**Alignement uniquement** — garde tous les points, aligne + échelle\n\n"
            "**Ignorer** — scan déjà propre et aligné"
        ),
    )
    if preprocess_mode == "🔧 Complet":
        pp_clean_mode = st.selectbox("Nettoyage", ["conservative","moderate","aggressive"], index=1)
        pp_keep_largest = st.checkbox("Garder le plus gros cluster (DBSCAN)", value=True)
        pp_normalize = st.checkbox("Normaliser les couleurs RGB", value=True)
        pp_target_h  = st.number_input("Hauteur cible (m)", 2.0, 5.0, 2.5, 0.1)
        pp_no_align  = st.checkbox("Skip alignement automatique", value=False)
        pp_no_denoise = False
    elif preprocess_mode == "📐 Alignement uniquement":
        st.info("Tous les points conservés — alignement + échelle uniquement.")
        pp_clean_mode  = "moderate"
        pp_keep_largest = False
        pp_normalize   = st.checkbox("Normaliser les couleurs RGB", value=True)
        pp_target_h    = st.number_input("Hauteur cible (m)", 2.0, 5.0, 2.5, 0.1)
        pp_no_align    = False
        pp_no_denoise  = True
    else:
        st.info("Preprocessing ignoré.")
        pp_clean_mode = "moderate"; pp_keep_largest = False
        pp_normalize  = False; pp_target_h = 2.5
        pp_no_align   = False; pp_no_denoise = False

    st.divider()

    # ── 2. Inference ──
    st.subheader("2 · Inférence SpatialLM")
    model_label = st.selectbox("Modèle", list(MODELS.keys()), index=0)
    model_path  = MODELS[model_label]
    detect_type = st.selectbox("Détection", ["all","arch","object"],
                               help="all=tout · arch=murs/portes/fenêtres · object=meubles")
    seed = st.number_input("Seed", value=42, step=1)

    st.divider()

    # ── 3. Post-processing ──
    st.subheader("3 · Post-processing")
    skip_postprocess = st.checkbox("Ignorer le post-processing", value=False)
    if not skip_postprocess:
        with st.expander("Paramètres avancés", expanded=False):
            pp_angle  = st.slider("Seuil angle fusion murs (°)",  1.0, 30.0, 10.0, 1.0,
                                  help="Deux murs sont colinéaires si leur angle diffère de moins de ce seuil")
            pp_dist   = st.slider("Seuil distance (m)",           0.05, 1.0, 0.35, 0.05,
                                  help="Distance perpendiculaire max entre les axes de deux murs à fusionner")
            pp_gap    = st.slider("Seuil espace (m)",             0.1,  2.0, 0.60, 0.05,
                                  help="Espace max entre les extrémités projetées pour autoriser la fusion")
            pp_minpts = st.number_input("Points min. pour raffinage bbox", 10, 500, 50, 10,
                                        help="Bbox avec moins de points cloud à l'intérieur ne seront pas réaffinées")
    else:
        pp_angle = 10.0; pp_dist = 0.35; pp_gap = 0.60; pp_minpts = 50

    st.divider()

    # ── 4. IFC ──
    st.subheader("4 · Génération IFC")
    skip_ifc = st.checkbox("Ignorer la génération IFC", value=False)


# =============================================================================
# MAIN AREA
# =============================================================================

col_upload, col_metrics = st.columns([2, 1])
with col_upload:
    st.subheader("📁 Importation")
    uploaded = st.file_uploader("Glisse-dépose ton fichier .ply", type=["ply"])
with col_metrics:
    st.metric("Modèle", "Llama 1B" if "Llama" in model_label else "Qwen 0.5B")
    st.metric("Preprocessing", {"🔧 Complet": pp_clean_mode,
                                 "📐 Alignement uniquement": "align only",
                                 "⏭️ Ignorer": "skipped"}[preprocess_mode])
    st.metric("Post-processing", "ignoré" if skip_postprocess else "activé")
    st.metric("IFC", "ignoré" if skip_ifc else "activé")


# ── Pipeline ───────────────────────────────────────────────────────────────
if uploaded:
    if "tmpdir" not in st.session_state:
        st.session_state["tmpdir"] = tempfile.mkdtemp(prefix="pipeline_ifc_")

    tmpdir = st.session_state["tmpdir"]
    base = Path(uploaded.name).stem

    raw_path      = os.path.join(tmpdir, uploaded.name)
    clean_path    = os.path.join(tmpdir, f"{base}_clean.ply")
    raw_layout    = os.path.join(tmpdir, f"{base}_layout.txt")
    refined_layout= os.path.join(tmpdir, f"{base}_refined_layout.txt")
    ifc_path      = os.path.join(tmpdir, f"{base}.ifc")
    rrd_path      = os.path.join(tmpdir, f"{base}.rrd")

    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info(f"📌 `{uploaded.name}` reçu ({uploaded.size/1024/1024:.1f} MB)")

    _btn_lbl = {
        "🔧 Complet":              "🚀 Lancer le pipeline complet (4 étapes)",
        "📐 Alignement uniquement": "🚀 Lancer : Alignement + Inférence + Post-process + IFC",
        "⏭️ Ignorer":              "🚀 Lancer : Inférence + Post-process + IFC",
    }
    if st.button(_btn_lbl[preprocess_mode], type="primary", use_container_width=True):
        timings = {}

        # ── ÉTAPE 1 : Preprocessing ─────────────────────────────────────────
        if preprocess_mode == "⏭️ Ignorer":
            with st.expander("📦 Étape 1/4 : Preprocessing — ignoré", expanded=False):
                st.info("Preprocessing ignoré.")
            infer_ply = raw_path
        else:
            step_lbl = ("📐 Étape 1/4 : Alignement + Échelle"
                        if preprocess_mode == "📐 Alignement uniquement"
                        else "📦 Étape 1/4 : Preprocessing")
            with st.expander(step_lbl, expanded=True):
                cmd = ["python", "preprocess_for_spatiallm.py",
                       "-i", raw_path, "-o", clean_path,
                       "--mode", pp_clean_mode,
                       "--target_height", str(pp_target_h)]
                if pp_no_denoise:     cmd.append("--no_denoise")
                if pp_keep_largest:   cmd.append("--keep_largest_cluster")
                if pp_normalize:      cmd.append("--normalize_colors")
                if pp_no_align:       cmd.append("--no_align")
                ok, t = run_command(cmd, "Preprocessing")
                if not ok: st.stop()
                timings["Preprocessing"] = t
            infer_ply = clean_path

        # ── ÉTAPE 2 : Inférence SpatialLM ───────────────────────────────────
        with st.expander("🧠 Étape 2/4 : Inférence SpatialLM", expanded=True):
            cmd = ["python", "inference.py",
                   "-p", infer_ply, "-o", raw_layout,
                   "--model_path", model_path,
                   "--detect_type", detect_type,
                   "--seed", str(seed)]
            ok, t = run_command(cmd, "Inférence SpatialLM")
            if not ok: st.stop()
            timings["Inférence"] = t

        # ── ÉTAPE 2b : Visualisation Rerun (optionnel, non-bloquant) ─────────
        with st.expander("🎨 Génération viewer .rrd", expanded=False):
            cmd_rrd = ["python", "visualize.py",
                       "-p", infer_ply, "-l", raw_layout, "--save", rrd_path]
            ok_rrd, t_rrd = run_command(cmd_rrd, "Visualisation RRD")
            if ok_rrd:
                timings["Visualisation RRD"] = t_rrd

        # ── ÉTAPE 3 : Post-processing ────────────────────────────────────────
        if skip_postprocess:
            with st.expander("🔧 Étape 3/4 : Post-processing — ignoré", expanded=False):
                st.info("Post-processing ignoré — layout brut utilisé.")
            final_layout = raw_layout
        else:
            with st.expander("🔧 Étape 3/4 : Post-processing", expanded=True):
                cmd = ["python", "postprocess.py",
                       "--layout", raw_layout,
                       "--point_cloud", infer_ply,
                       "--output", refined_layout,
                       "--angle_thresh", str(pp_angle),
                       "--dist_thresh",  str(pp_dist),
                       "--gap_thresh",   str(pp_gap),
                       "--min_points",   str(pp_minpts)]
                ok, t = run_command(cmd, "Post-processing")
                if not ok: st.stop()
                timings["Post-processing"] = t
            final_layout = refined_layout

        # ── ÉTAPE 4 : Génération IFC ─────────────────────────────────────────
        if skip_ifc:
            with st.expander("🏗️ Étape 4/4 : Génération IFC — ignorée", expanded=False):
                st.info("Génération IFC ignorée.")
        else:
            with st.expander("🏗️ Étape 4/4 : Génération IFC", expanded=True):
                cmd = ["python", "spatiallm_to_ifc.py",
                       "--input", final_layout, "--output", ifc_path]
                ok, t = run_command(cmd, "Génération IFC")
                if not ok: st.stop()
                timings["IFC"] = t

        # Persistance
        st.session_state.update({
            "infer_ply":    infer_ply,
            "raw_layout":   raw_layout,
            "final_layout": final_layout,
            "ifc_path":     ifc_path,
            "rrd_path":     rrd_path,
            "base":         base,
            "timings":      timings,
            "pipeline_done": True,
            "skip_ifc":     skip_ifc,
        })
        st.balloons()


# =============================================================================
# RÉSULTATS
# =============================================================================
if st.session_state.get("pipeline_done"):
    infer_ply    = st.session_state["infer_ply"]
    raw_layout   = st.session_state["raw_layout"]
    final_layout = st.session_state["final_layout"]
    ifc_path     = st.session_state["ifc_path"]
    rrd_path     = st.session_state["rrd_path"]
    base         = st.session_state["base"]
    timings      = st.session_state.get("timings", {})

    st.markdown("---")
    st.header("🎉 Résultats")

    # ── Métriques layout ────────────────────────────────────────────────────
    layout_file = final_layout if os.path.exists(final_layout) else raw_layout
    if os.path.exists(layout_file):
        with open(layout_file) as f:
            layout_text = f.read()
        stats = layout_stats(layout_text)

        st.subheader("📋 Layout détecté")
        cols = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, v)

        # Comparaison brut vs raffiné si post-processing actif
        if (os.path.exists(raw_layout) and os.path.exists(refined_layout)
                and raw_layout != refined_layout):
            with open(raw_layout) as f:
                raw_text = f.read()
            raw_stats = layout_stats(raw_text)
            with st.expander("Comparaison brut vs raffiné", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("**Brut (SpatialLM)**")
                    for k, v in raw_stats.items():
                        st.write(f"{k}: **{v}**")
                with c2:
                    st.caption("**Raffiné (post-processing)**")
                    for k, v in stats.items():
                        diff = v - raw_stats.get(k, v)
                        arrow = f" ({'+' if diff>0 else ''}{diff})" if diff != 0 else ""
                        st.write(f"{k}: **{v}**{arrow}")

        with st.expander("Voir le code du layout", expanded=False):
            st.code(layout_text, language="python")

    # ── Temps ───────────────────────────────────────────────────────────────
    if timings:
        with st.expander("⏱️ Temps par étape", expanded=False):
            for k, v in timings.items():
                st.write(f"**{k}** : {v:.1f}s")
            st.write(f"**Total** : {sum(timings.values()):.1f}s")

    # ── Téléchargements ─────────────────────────────────────────────────────
    st.subheader("📥 Téléchargements")
    dl_cols = st.columns(4)
    files = [
        (infer_ply,    f"{base}_clean.ply",            "⬇️ PLY nettoyé",      "application/octet-stream"),
        (raw_layout,   f"{base}_layout.txt",           "⬇️ Layout brut",       "text/plain"),
        (final_layout, f"{base}_refined_layout.txt",   "⬇️ Layout raffiné",    "text/plain"),
        (ifc_path,     f"{base}.ifc",                  "⬇️ Fichier IFC",       "application/octet-stream"),
    ]
    for col, (path, fname, label, mime) in zip(dl_cols, files):
        with col:
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    st.download_button(label, fh.read(), file_name=fname,
                                       mime=mime, use_container_width=True)
            else:
                st.button(label, disabled=True, use_container_width=True)

    # ── Visualisation ───────────────────────────────────────────────────────
    st.subheader("🎨 Visualisation 3D")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
    with btn_col1:
        if st.button("▶️ Lancer viewer RRD", use_container_width=True):
            st.session_state["viewer_running"] = True
    with btn_col2:
        if st.button("⏹️ Arrêter viewer", use_container_width=True):
            subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL)
            st.session_state["viewer_running"] = False
            st.rerun()
    with btn_col3:
        st.caption(f"Port Rerun : `{RERUN_WEB_PORT}`")

    rrd_col, maquette_col = st.columns(2)

    with rrd_col:
        st.caption("🔴 **Rerun** — point cloud + détections")
        if st.session_state.get("viewer_running"):
            try:
                check = subprocess.run(
                    ["pgrep", "-f", f"rerun.*{RERUN_WEB_PORT}"], capture_output=True)
                if check.returncode != 0:
                    subprocess.Popen(["rerun", "--web-viewer",
                                      "--web-viewer-port", str(RERUN_WEB_PORT),
                                      "--port", str(RERUN_WEB_PORT + 1), rrd_path],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(3)
                viewer_url = (f"http://localhost:{RERUN_WEB_PORT}"
                              f"/?url=ws://localhost:{RERUN_WEB_PORT + 1}")
                components.iframe(viewer_url, height=560)
                st.caption(f"[Ouvrir en plein écran](http://localhost:{RERUN_WEB_PORT})")
            except Exception as e:
                st.error(f"Erreur viewer : {e}")
        else:
            st.info("👉 Clique sur **Lancer viewer RRD** pour afficher le point cloud")

    with maquette_col:
        st.caption("📐 **Maquette 3D** — layout interactif")
        if not HAS_PLOTLY:
            st.warning("`pip install plotly` requis pour la maquette")
        elif os.path.exists(layout_file):
            try:
                _walls, _wmap, _doors, _wins, _bboxes = parse_layout_for_3d(layout_text)
                _fig = build_3d_figure(_walls, _wmap, _doors, _wins, _bboxes)
                st.plotly_chart(_fig, use_container_width=True)
                parts = []
                if _walls:  parts.append(f"🟦 {len(_walls)} murs")
                if _doors:  parts.append(f"🟩 {len(_doors)} portes")
                if _wins:   parts.append(f"🟦 {len(_wins)} fenêtres")
                if _bboxes: parts.append(f"🟧 {len(_bboxes)} objets")
                st.caption("  ·  ".join(parts))
            except Exception as e:
                st.error(f"Erreur maquette : {e}")
        else:
            st.info("Layout non disponible.")

elif not uploaded:
    st.info("👆 Upload un fichier .ply pour démarrer")

    with st.expander("ℹ️ Comment ça marche", expanded=False):
        st.markdown("""
        Ce pipeline enchaîne automatiquement 4 étapes :

        1. **Preprocessing** — nettoyage du bruit, alignement Z-up + Manhattan, mise à l'échelle métrique
        2. **Inférence SpatialLM** — détection des murs, portes, fenêtres et meubles
        3. **Post-processing** — fusion des murs colinéaires, réaffinage des bounding boxes sur le nuage réel
        4. **Génération IFC** — production d'un fichier BIM standard (IFC4) avec dalle de sol automatique

        **Résultats :**
        - Maquette 3D interactive (Plotly) directement dans l'interface
        - Viewer Rerun embarqué pour visualiser le nuage de points + détections
        - Téléchargements : PLY nettoyé · Layout TXT brut · Layout TXT raffiné · **Fichier IFC**
        """)
