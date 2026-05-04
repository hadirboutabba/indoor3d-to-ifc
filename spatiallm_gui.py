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

    # ─── Visualisation 3D embarquée ─────────────────────────────────────────
    st.subheader("🎨 Visualisation 3D")

    viz_col1, viz_col2 = st.columns([3, 1])

    with viz_col2:
        st.markdown("**Contrôles**")
        if st.button("▶️ Lancer le viewer 3D", use_container_width=True):
            st.session_state["viewer_running"] = True

        if st.button("⏹️ Arrêter le viewer", use_container_width=True):
            subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL)
            st.session_state["viewer_running"] = False
            st.rerun()

        st.caption(f"Port web : `{RERUN_WEB_PORT}`")

    with viz_col1:
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

                viewer_url = f"http://localhost:{RERUN_WEB_PORT}/?url=ws://localhost:{RERUN_WEB_PORT + 1}"
                components.iframe(viewer_url, height=600)
                st.caption(
                    f"Si la 3D ne s'affiche pas, ouvre directement : "
                    f"[http://localhost:{RERUN_WEB_PORT}](http://localhost:{RERUN_WEB_PORT})"
                )
            except Exception as e:
                st.error(f"Erreur viewer : {e}")
        else:
            st.info("👉 Clique sur **Lancer le viewer 3D** à droite pour afficher la scène")

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
