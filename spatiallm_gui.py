"""
Interface Streamlit pour SpatialLM — version simplifiée.

Workflow :
  1. Upload .ply
  2. Pipeline automatique (preprocess + inference + visualize)
  3. Téléchargement des résultats via boutons (vont dans Downloads du navigateur)
  4. Visualisation 3D embarquée

Setup (dans WSL Ubuntu) :
    pip install streamlit rerun-sdk

Lancement :
    cd /mnt/c/Users/hboutabb/SpatialLM
    streamlit run spatiallm_gui.py

Ouvre http://localhost:8501 dans ton navigateur Windows.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import re
import math

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# =============================================================================
# 3D LAYOUT MAQUETTE (Plotly)
# =============================================================================

def parse_layout_for_3d(layout_text):
    walls, doors, windows, bboxes = [], [], [], []
    wall_map = {}

    for line in layout_text.strip().splitlines():
        m = re.match(r'(wall_\d+)=Wall\(([^)]+)\)', line)
        if m:
            vals = list(map(float, m.group(2).split(',')))
            w = {'x1': vals[0], 'y1': vals[1], 'z1': vals[2],
                 'x2': vals[3], 'y2': vals[4], 'z2': vals[5], 'height': vals[6]}
            walls.append(w)
            wall_map[m.group(1)] = w
            continue

        m = re.match(r'door_\d+=Door\(([^,]+),([^)]+)\)', line)
        if m:
            vals = list(map(float, m.group(2).split(',')))
            doors.append({'wall_ref': m.group(1).strip(),
                          'cx': vals[0], 'cy': vals[1], 'cz': vals[2],
                          'width': vals[3], 'height': vals[4]})
            continue

        m = re.match(r'window_\d+=Window\(([^,]+),([^)]+)\)', line)
        if m:
            vals = list(map(float, m.group(2).split(',')))
            windows.append({'wall_ref': m.group(1).strip(),
                            'cx': vals[0], 'cy': vals[1], 'cz': vals[2],
                            'width': vals[3], 'height': vals[4]})
            continue

        m = re.match(r'bbox_\d+=Bbox\(([^,]+),([^)]+)\)', line)
        if m:
            vals = list(map(float, m.group(2).split(',')))
            bboxes.append({'label': m.group(1),
                           'cx': vals[0], 'cy': vals[1], 'cz': vals[2],
                           'angle': vals[3], 'dx': vals[4], 'dy': vals[5], 'dz': vals[6]})

    return walls, wall_map, doors, windows, bboxes


def _wall_dir(wall):
    dx = wall['x2'] - wall['x1']
    dy = wall['y2'] - wall['y1']
    length = math.sqrt(dx * dx + dy * dy) or 1.0
    return dx / length, dy / length


def _quad_mesh(fig, xs4, ys4, zs4, color, opacity, lighting=None):
    """Add a filled quad (2 triangles) + its wire outline to fig."""
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
        x=list(xs4) + [xs4[0]],
        y=list(ys4) + [ys4[0]],
        z=list(zs4) + [zs4[0]],
        mode='lines',
        line=dict(color=color, width=width),
        showlegend=False, hoverinfo='skip',
    ))


# Object colour palette: (fill, edge)
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

    # ── Global z extents ────────────────────────────────────────────────────
    zb_global = min(min(w['z1'], w['z2']) for w in walls)
    zt_global = max(min(w['z1'], w['z2']) + w['height'] for w in walls)

    # ── Floor ────────────────────────────────────────────────────────────────
    # Use all wall endpoints + alphahull to fill the room footprint
    fxs = [w['x1'] for w in walls] + [w['x2'] for w in walls]
    fys = [w['y1'] for w in walls] + [w['y2'] for w in walls]
    fzs = [zb_global] * len(fxs)
    fig.add_trace(go.Mesh3d(
        x=fxs, y=fys, z=fzs,
        alphahull=0,           # Delaunay fill of the room footprint
        color='#7a6a58', opacity=0.90,
        showlegend=False, hoverinfo='skip',
        flatshading=True,
        lighting=dict(ambient=0.95, diffuse=0.2),
    ))

    # ── Ceiling (faint) ──────────────────────────────────────────────────────
    fig.add_trace(go.Mesh3d(
        x=fxs, y=fys, z=[zt_global] * len(fxs),
        alphahull=0,
        color='#f0ece4', opacity=0.12,
        showlegend=False, hoverinfo='skip',
    ))

    # ── Walls ────────────────────────────────────────────────────────────────
    wall_light = dict(ambient=0.65, diffuse=0.85, specular=0.08, roughness=0.8)
    for wall in walls:
        x1, y1, z1 = wall['x1'], wall['y1'], wall['z1']
        x2, y2, z2 = wall['x2'], wall['y2'], wall['z2']
        zb = min(z1, z2)
        zt = zb + wall['height']
        xs4 = [x1, x2, x2, x1]
        ys4 = [y1, y2, y2, y1]
        zs4 = [zb, zb, zt, zt]
        _quad_mesh(fig, xs4, ys4, zs4, '#ede5d8', 0.90, wall_light)
        _quad_wire(fig, xs4, ys4, zs4, '#b8ad9e', 1.2)

    # ── Doors ────────────────────────────────────────────────────────────────
    door_light = dict(ambient=0.6, diffuse=0.9, specular=0.05)
    for door in doors:
        cx, cy, cz = door['cx'], door['cy'], door['cz']
        w, h = door['width'], door['height']
        ddx, ddy = _wall_dir(wall_map[door['wall_ref']]) if door['wall_ref'] in wall_map else (1.0, 0.0)
        hw = w / 2
        xs4 = [cx - hw*ddx, cx + hw*ddx, cx + hw*ddx, cx - hw*ddx]
        ys4 = [cy - hw*ddy, cy + hw*ddy, cy + hw*ddy, cy - hw*ddy]
        zs4 = [cz - h/2, cz - h/2, cz + h/2, cz + h/2]
        _quad_mesh(fig, xs4, ys4, zs4, '#8b6343', 0.82, door_light)
        _quad_wire(fig, xs4, ys4, zs4, '#5c3d20', 2.0)

    # ── Windows ──────────────────────────────────────────────────────────────
    win_light = dict(ambient=0.5, diffuse=0.7, specular=0.5, roughness=0.1)
    for win in windows:
        cx, cy, cz = win['cx'], win['cy'], win['cz']
        w, h = win['width'], win['height']
        wdx, wdy = _wall_dir(wall_map[win['wall_ref']]) if win['wall_ref'] in wall_map else (1.0, 0.0)
        hw = w / 2
        xs4 = [cx - hw*wdx, cx + hw*wdx, cx + hw*wdx, cx - hw*wdx]
        ys4 = [cy - hw*wdy, cy + hw*wdy, cy + hw*wdy, cy - hw*wdy]
        zs4 = [cz - h/2, cz - h/2, cz + h/2, cz + h/2]
        _quad_mesh(fig, xs4, ys4, zs4, '#87ceeb', 0.45, win_light)
        _quad_wire(fig, xs4, ys4, zs4, '#5599cc', 2.0)

    # ── Bounding boxes ───────────────────────────────────────────────────────
    # 12 triangles covering the 6 faces of a box (corner ordering: ±x, ±y, ±z)
    # i=0: (-,-,-) i=1: (-,-,+) i=2: (-,+,-) i=3: (-,+,+)
    # i=4: (+,-,-) i=5: (+,-,+) i=6: (+,+,-) i=7: (+,+,+)
    BOX_I = [0, 0,  1, 1,  0, 0,  2, 2,  0, 0,  4, 4]
    BOX_J = [2, 4,  3, 5,  1, 4,  3, 6,  1, 2,  5, 6]
    BOX_K = [6, 6,  7, 7,  5, 5,  7, 7,  3, 3,  7, 7]
    BOX_EDGES = [(0,1),(2,3),(4,5),(6,7),
                 (0,2),(1,3),(4,6),(5,7),
                 (0,4),(1,5),(2,6),(3,7)]

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
        hdx, hdy, hdz = bbox['dx'] / 2, bbox['dy'] / 2, bbox['dz'] / 2
        ca, sa = math.cos(ang), math.sin(ang)

        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    lx, ly = sx * hdx, sy * hdy
                    corners.append((cx + ca*lx - sa*ly,
                                    cy + sa*lx + ca*ly,
                                    cz + sz * hdz))

        cxs = [c[0] for c in corners]
        cys = [c[1] for c in corners]
        czs = [c[2] for c in corners]

        # Solid semi-transparent faces
        fig.add_trace(go.Mesh3d(
            x=cxs, y=cys, z=czs,
            i=BOX_I, j=BOX_J, k=BOX_K,
            color=fill_col, opacity=0.32,
            showlegend=False, hoverinfo='skip',
            flatshading=True, lighting=obj_light,
        ))

        # Wireframe edges
        ex, ey, ez = [], [], []
        for ai, bi in BOX_EDGES:
            ex += [corners[ai][0], corners[bi][0], None]
            ey += [corners[ai][1], corners[bi][1], None]
            ez += [corners[ai][2], corners[bi][2], None]
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color=edge_col, width=2),
            name=label.replace('_', ' '),
            legendgroup=label,
            showlegend=label not in seen,
            hoverinfo='name',
        ))
        seen.add(label)

        # Floating label above box
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz + hdz + 0.06],
            mode='text',
            text=[label.replace('_', ' ')],
            textfont=dict(size=9, color='rgba(255,255,255,0.85)'),
            showlegend=False, hoverinfo='skip',
        ))

    # ── Camera & scene ───────────────────────────────────────────────────────
    all_x = [w['x1'] for w in walls] + [w['x2'] for w in walls]
    all_y = [w['y1'] for w in walls] + [w['y2'] for w in walls]
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', gridcolor='#2d2d2d',
                       backgroundcolor='#181818', showbackground=True,
                       zerolinecolor='#3a3a3a'),
            yaxis=dict(title='Y (m)', gridcolor='#2d2d2d',
                       backgroundcolor='#181818', showbackground=True,
                       zerolinecolor='#3a3a3a'),
            zaxis=dict(title='Z (m)', gridcolor='#2d2d2d',
                       backgroundcolor='#121212', showbackground=True,
                       zerolinecolor='#3a3a3a'),
            bgcolor='#141414',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=1.6 * span / 4, y=-1.6 * span / 4, z=1.1 * span / 4),
            ),
        ),
        paper_bgcolor='#0e1117',
        font=dict(color='#dddddd', size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        height=580,
        legend=dict(
            x=0, y=1,
            bgcolor='rgba(15,15,15,0.88)',
            bordercolor='#444', borderwidth=1,
            font=dict(color='#cccccc', size=10),
        ),
    )
    return fig


# =============================================================================
# CONFIGURATION
# =============================================================================
SPATIALLM_DIR  = "/mnt/c/Users/hboutabb/SpatialLM"
MODEL_PATH     = "/mnt/c/Users/hboutabb/SpatialLM1.1-Llama-1B"
RERUN_WEB_PORT = 9090

# =============================================================================

st.set_page_config(
    page_title="SpatialLM Pipeline",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 SpatialLM — Pipeline Automatique")
st.caption("Upload un .ply → Preprocess → Inference → Visualisation 3D embarquée")

# ── Sidebar : paramètres ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")

    st.subheader("Preprocessing")
    mode = st.selectbox(
        "Mode de nettoyage",
        ["conservative", "moderate", "aggressive"],
        index=1,
        help="conservative = préserve détails fins | moderate = équilibré | aggressive = bruit fort"
    )
    keep_largest = st.checkbox(
        "Garder le plus gros cluster (DBSCAN)",
        value=True,
        help="Élimine les blobs flottants isolés"
    )
    normalize_colors = st.checkbox(
        "Normaliser les couleurs RGB",
        value=True,
        help="Aligne sur la distribution du SpatialLM-Testset"
    )
    target_height = st.number_input(
        "Hauteur de plafond cible (m)",
        min_value=2.0, max_value=5.0, value=2.5, step=0.1,
        help="Hauteur réelle du plafond de la pièce scannée"
    )

    st.subheader("Inference")
    detect_type = st.selectbox(
        "Type de détection",
        ["all", "arch", "object"],
        help="all = tout | arch = murs/portes/fenêtres | object = meubles uniquement"
    )
    seed = st.number_input("Seed (reproductibilité)", value=42, step=1)

    st.subheader("Avancé")
    no_align = st.checkbox(
        "Skip alignement automatique",
        value=False,
        help="À cocher si déjà aligné dans CloudCompare"
    )

# ── Zone principale ────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Importation")
    uploaded = st.file_uploader(
        "Glisse-dépose ton fichier .ply",
        type=["ply"]
    )

with col2:
    st.metric("Modèle", "SpatialLM 1.1 Llama-1B")
    st.metric("Mode", mode)
    st.metric("Detection", detect_type)


def run_command(cmd, label):
    """Lance une commande shell et streame le log dans Streamlit."""
    log_box = st.empty()
    full_log = ""
    start = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=SPATIALLM_DIR
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
    return process.returncode == 0


# ── Pipeline ───────────────────────────────────────────────────────────────
if uploaded:
    # Stocke l'upload dans un dossier temporaire (effacé en fin de session)
    if "tmpdir" not in st.session_state:
        st.session_state["tmpdir"] = tempfile.mkdtemp(prefix="spatiallm_")

    tmpdir = st.session_state["tmpdir"]
    base = Path(uploaded.name).stem

    raw_path    = os.path.join(tmpdir, uploaded.name)
    clean_path  = os.path.join(tmpdir, f"{base}_clean.ply")
    layout_path = os.path.join(tmpdir, f"{base}_layout.txt")
    rrd_path    = os.path.join(tmpdir, f"{base}.rrd")

    # Sauvegarder le fichier uploadé en local pour que les scripts puissent le lire
    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info(f"📌 Fichier reçu : `{uploaded.name}` ({uploaded.size / 1024 / 1024:.1f} MB)")

    if st.button("🚀 Lancer le pipeline complet", type="primary", use_container_width=True):

        # ─── ÉTAPE 1 : Preprocessing ────────────────────────────────────────
        with st.expander("📦 Étape 1/3 : Preprocessing", expanded=True):
            cmd_preprocess = [
                "python", "preprocess_for_spatiallm.py",
                "-i", raw_path,
                "-o", clean_path,
                "--mode", mode,
                "--target_height", str(target_height),
            ]
            if keep_largest:
                cmd_preprocess.append("--keep_largest_cluster")
            if normalize_colors:
                cmd_preprocess.append("--normalize_colors")
            if no_align:
                cmd_preprocess.append("--no_align")

            ok1 = run_command(cmd_preprocess, "Preprocessing")
            if not ok1:
                st.stop()

        # ─── ÉTAPE 2 : Inférence ────────────────────────────────────────────
        with st.expander("🧠 Étape 2/3 : Inférence SpatialLM", expanded=True):
            cmd_inference = [
                "python", "inference.py",
                "-p", clean_path,
                "-o", layout_path,
                "--model_path", MODEL_PATH,
                "--detect_type", detect_type,
                "--seed", str(seed),
            ]
            ok2 = run_command(cmd_inference, "Inference")
            if not ok2:
                st.stop()

        # ─── ÉTAPE 3 : Génération du .rrd ───────────────────────────────────
        with st.expander("🎨 Étape 3/3 : Génération visualisation .rrd", expanded=True):
            cmd_visualize = [
                "python", "visualize.py",
                "-p", clean_path,
                "-l", layout_path,
                "--save", rrd_path,
            ]
            ok3 = run_command(cmd_visualize, "Visualization")
            if not ok3:
                st.stop()

        # Sauvegarde dans le state pour persistance
        st.session_state["clean_path"]    = clean_path
        st.session_state["layout_path"]   = layout_path
        st.session_state["rrd_path"]      = rrd_path
        st.session_state["base"]          = base
        st.session_state["pipeline_done"] = True

        st.balloons()


# ── Affichage des résultats (persistant via session_state) ─────────────────
if st.session_state.get("pipeline_done"):
    layout_path = st.session_state["layout_path"]
    rrd_path    = st.session_state["rrd_path"]
    clean_path  = st.session_state["clean_path"]
    base        = st.session_state["base"]

    st.markdown("---")
    st.header("🎉 Résultats")

    # Layout détecté
    if os.path.exists(layout_path):
        with open(layout_path) as f:
            layout_content = f.read()

        st.subheader("📋 Layout détecté")

        n_walls = layout_content.count("=Wall(")
        n_doors = layout_content.count("=Door(")
        n_windows = layout_content.count("=Window(")
        n_bbox = layout_content.count("=Bbox(")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Murs", n_walls)
        c2.metric("Portes", n_doors)
        c3.metric("Fenêtres", n_windows)
        c4.metric("Objets", n_bbox)

        with st.expander("Voir le code du layout", expanded=False):
            st.code(layout_content, language="python")

    # ─── Téléchargements ────────────────────────────────────────────────────
    st.subheader("📥 Téléchargements")
    st.caption("Les fichiers seront téléchargés dans ton dossier **Downloads**.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if os.path.exists(clean_path):
            with open(clean_path, "rb") as f:
                st.download_button(
                    "⬇️ PLY nettoyé", f.read(),
                    file_name=f"{base}_clean.ply",
                    mime="application/octet-stream",
                    use_container_width=True
                )
    with col_b:
        if os.path.exists(layout_path):
            with open(layout_path, "rb") as f:
                st.download_button(
                    "⬇️ Layout TXT", f.read(),
                    file_name=f"{base}_layout.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    with col_c:
        if os.path.exists(rrd_path):
            with open(rrd_path, "rb") as f:
                st.download_button(
                    "⬇️ Fichier RRD", f.read(),
                    file_name=f"{base}.rrd",
                    mime="application/octet-stream",
                    use_container_width=True
                )

    # ─── Visualisation 3D ───────────────────────────────────────────────────
    st.subheader("🎨 Visualisation 3D")

    # Control bar (above both panels)
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
    with btn_col1:
        if st.button("▶️ Lancer le viewer RRD", use_container_width=True):
            st.session_state["viewer_running"] = True
    with btn_col2:
        if st.button("⏹️ Arrêter le viewer", use_container_width=True):
            subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL)
            st.session_state["viewer_running"] = False
            st.rerun()
    with btn_col3:
        st.caption(f"Port Rerun : `{RERUN_WEB_PORT}`")

    # Split panels
    rrd_col, maquette_col = st.columns(2)

    # ── Left : Rerun RRD viewer ───────────────────────────────────────────
    with rrd_col:
        st.caption("🔴 **Rerun** — point cloud + détections")
        if st.session_state.get("viewer_running"):
            try:
                check = subprocess.run(
                    ["pgrep", "-f", f"rerun.*{RERUN_WEB_PORT}"],
                    capture_output=True
                )
                if check.returncode != 0:
                    subprocess.Popen([
                        "rerun",
                        "--web-viewer",
                        "--web-viewer-port", str(RERUN_WEB_PORT),
                        "--port", str(RERUN_WEB_PORT + 1),
                        rrd_path,
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(3)

                viewer_url = (
                    f"http://localhost:{RERUN_WEB_PORT}"
                    f"/?url=ws://localhost:{RERUN_WEB_PORT + 1}"
                )
                components.iframe(viewer_url, height=580)
                st.caption(
                    f"Lien direct : "
                    f"[http://localhost:{RERUN_WEB_PORT}](http://localhost:{RERUN_WEB_PORT})"
                )
            except Exception as e:
                st.error(f"Erreur viewer : {e}")
        else:
            st.info("👉 Clique sur **Lancer le viewer RRD** ci-dessus pour afficher le point cloud")

    # ── Right : Plotly 3D maquette ────────────────────────────────────────
    with maquette_col:
        st.caption("📐 **Maquette 3D** — layout interactif")
        if not HAS_PLOTLY:
            st.warning("Plotly non installé — `pip install plotly`")
        elif os.path.exists(layout_path):
            try:
                with open(layout_path) as _f:
                    _lt = _f.read()
                _walls, _wall_map, _doors, _wins, _bboxes = parse_layout_for_3d(_lt)
                _fig = build_3d_figure(_walls, _wall_map, _doors, _wins, _bboxes)
                st.plotly_chart(_fig, use_container_width=True)

                _leg = []
                if _walls:   _leg.append(f"🟦 {len(_walls)} murs")
                if _doors:   _leg.append(f"🟩 {len(_doors)} portes")
                if _wins:    _leg.append(f"🟦 {len(_wins)} fenêtres")
                if _bboxes:  _leg.append(f"🟧 {len(_bboxes)} objets")
                st.caption("  ·  ".join(_leg))
            except Exception as e:
                st.error(f"Erreur maquette : {e}")
        else:
            st.info("Layout non disponible.")

elif not uploaded:
    st.info("👆 Upload un fichier .ply pour démarrer")

    with st.expander("ℹ️ Comment ça marche", expanded=False):
        st.markdown("""
        Cette interface enchaîne automatiquement les 3 étapes :

        1. **Preprocessing** — nettoyage, alignement Z-up, échelle métrique, normalisation
        2. **Inference** — détection des éléments par SpatialLM
        3. **Visualization** — génère le fichier `.rrd` + viewer 3D embarqué

        **Sauvegarde** : les fichiers sont temporaires côté serveur. Pour les conserver,
        utilise les boutons **⬇️ Téléchargements** — ils iront dans ton dossier Downloads.
        """)
