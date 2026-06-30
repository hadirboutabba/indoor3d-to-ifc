"""
run_pipeline.py
---------------
Pipeline complet  PLY → IFC  en une seule commande.

Étapes :
  1. preprocess_for_spatiallm.py   Nettoyage du nuage de points
  2. inference.py                   SpatialLM  →  layout TXT brut
  3. postprocess.py                 Correction géométrique du TXT
  4. spatiallm_to_ifc.py            TXT raffiné  →  fichier IFC

Usage minimal :
    python run_pipeline.py --input raw_scan.ply --output scene.ifc

Avec options :
    python run_pipeline.py \\
        --input  raw_scan.ply \\
        --output scene.ifc \\
        --model  manycore-research/SpatialLM1.1-Qwen-0.5B \\
        --detect_type all \\
        --preprocess_mode moderate \\
        --workdir /tmp/pipeline_work \\
        --skip_preprocess \\
        --skip_postprocess

Environnements :
    Toutes les étapes utilisent sys.executable par défaut.
    Si l'inférence SpatialLM tourne dans un environnement conda séparé
    (avec torch/transformers), passez son interpréteur avec --inference_python :

        --inference_python /path/to/spatiallm-env/bin/python
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ── Utilitaires ───────────────────────────────────────────────────────

def step(title: str):
    """Affiche un bandeau de séparation pour chaque étape."""
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def run(cmd: list[str], label: str) -> float:
    """Lance une commande, mesure le temps, lève RuntimeError si échec."""
    print(f"\n$ {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"[ÉCHEC] {label} (code={result.returncode}) — "
            "le pipeline s'arrête."
        )
    print(f"\n[OK] {label}  ({elapsed:.1f}s)")
    return elapsed


# ── Point d'entrée ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Pipeline PLY → IFC avec SpatialLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Entrée / sortie ──
    p.add_argument("--input",  "-i", required=True,
                   help="Nuage de points brut (.ply)")
    p.add_argument("--output", "-o", default=None,
                   help="Fichier IFC de sortie (défaut: <stem>.ifc à côté de --input)")
    p.add_argument("--workdir", "-w", default=None,
                   help="Dossier pour les fichiers intermédiaires "
                        "(défaut: dossier de --input)")

    # ── Étape 1 : prétraitement ──
    p.add_argument("--skip_preprocess", action="store_true",
                   help="Passer le prétraitement (utilise --input tel quel)")
    p.add_argument("--preprocess_mode",
                   choices=["conservative", "moderate", "aggressive"],
                   default="moderate",
                   help="Intensité du nettoyage (défaut: moderate)")
    p.add_argument("--target_height", type=float, default=2.5,
                   help="Hauteur normalisée du nuage de points (défaut: 2.5 m)")

    # ── Étape 2 : inférence SpatialLM ──
    p.add_argument("--model", "-m",
                   default="manycore-research/SpatialLM1.1-Llama-1B",
                   help="Chemin ou nom HuggingFace du modèle SpatialLM")
    p.add_argument("--detect_type",
                   choices=["all", "arch", "object"], default="all",
                   help="Éléments à détecter (défaut: all)")
    # À ajouter dans le bloc argparse de run_pipeline.py
    p.add_argument("--seed", type=int, default=42,
               help="Seed pour la reproductibilité de l'inférence (défaut: 42)")
    p.add_argument("--inference_python", default=None,
                   help="Interpréteur Python alternatif pour l'inférence "
                        "(ex: /opt/conda/envs/spatiallm/bin/python). "
                        "Utile si SpatialLM est dans un env conda séparé.")

    # ── Étape 3 : post-processing ──
    p.add_argument("--skip_postprocess", action="store_true",
                   help="Passer l'étape de correction géométrique")
    p.add_argument("--angle_thresh", type=float, default=10.0,
                   help="Seuil angle (deg) pour fusion de murs (défaut: 10)")
    p.add_argument("--dist_thresh",  type=float, default=0.35,
                   help="Seuil distance (m) pour fusion de murs (défaut: 0.35)")
    p.add_argument("--gap_thresh",   type=float, default=0.6,
                   help="Seuil espace (m) pour fusion de murs (défaut: 0.6)")
    p.add_argument("--min_points",   type=int,   default=50,
                   help="Points min pour raffinage bbox (défaut: 50)")

    args = p.parse_args()

    # ── Résolution des chemins ──
    input_ply  = Path(args.input).resolve()
    stem       = input_ply.stem
    workdir    = Path(args.workdir).resolve() if args.workdir else input_ply.parent
    workdir.mkdir(parents=True, exist_ok=True)

    output_ifc = Path(args.output).resolve() if args.output \
                 else input_ply.parent / f"{stem}.ifc"

    clean_ply     = workdir / f"{stem}_clean.ply"
    raw_layout    = workdir / f"{stem}_raw_layout.txt"
    refined_layout = workdir / f"{stem}_refined_layout.txt"

    # Pythons à utiliser
    this_python      = sys.executable
    inference_python = args.inference_python or this_python

    # Répertoire du script (même dossier que run_pipeline.py)
    script_dir = Path(__file__).parent

    # ── Affichage du plan ──
    print("\n╔══════════════════════════════════════════════════╗")
    print("║      Pipeline PLY → IFC  (SpatialLM)            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Entrée          : {input_ply}")
    print(f"  Sortie IFC      : {output_ifc}")
    print(f"  Dossier travail : {workdir}")
    print(f"  Modèle          : {args.model}")
    print(f"  Étapes          : "
          f"{'[skip] ' if args.skip_preprocess else ''}prétraitement → "
          f"inférence → "
          f"{'[skip] ' if args.skip_postprocess else ''}post-processing → "
          f"IFC")

    timings = {}
    t_total = time.time()

    # ─────────────────────────────────────────────────────────────────
    #  ÉTAPE 1 — Prétraitement
    # ─────────────────────────────────────────────────────────────────
    step("ÉTAPE 1 / 4 — Prétraitement du nuage de points")
    if args.skip_preprocess:
        print("  → Prétraitement ignoré, utilisation directe de :", input_ply)
        clean_ply = input_ply
    else:
        timings["preprocess"] = run(
            [
                this_python,
                str(script_dir / "preprocess_for_spatiallm.py"),
                "--input",  str(input_ply),
                "--output", str(clean_ply),
                "--mode",   args.preprocess_mode,
                "--target_height", str(args.target_height),
            ],
            "Prétraitement",
        )

    # ─────────────────────────────────────────────────────────────────
    #  ÉTAPE 2 — Inférence SpatialLM
    # ─────────────────────────────────────────────────────────────────
    step("ÉTAPE 2 / 4 — Inférence SpatialLM")
    timings["inference"] = run(
    [
        inference_python,
        str(script_dir / "inference.py"),
        "--point_cloud", str(clean_ply),
        "--output",      str(raw_layout),
        "--model_path",  args.model,
        "--detect_type", args.detect_type,
        "--seed",        str(args.seed),   # ← ajout
    ],
    "Inférence SpatialLM",
)

    # ─────────────────────────────────────────────────────────────────
    #  ÉTAPE 3 — Post-processing
    # ─────────────────────────────────────────────────────────────────
    step("ÉTAPE 3 / 4 — Correction géométrique (post-processing)")
    if args.skip_postprocess:
        print("  → Post-processing ignoré, layout brut utilisé.")
        refined_layout = raw_layout
    else:
        timings["postprocess"] = run(
            [
                this_python,
                str(script_dir / "postprocess.py"),
                "--layout",      str(raw_layout),
                "--point_cloud", str(clean_ply),
                "--output",      str(refined_layout),
                "--angle_thresh", str(args.angle_thresh),
                "--dist_thresh",  str(args.dist_thresh),
                "--gap_thresh",   str(args.gap_thresh),
                "--min_points",   str(args.min_points),
            ],
            "Post-processing",
        )

    # ─────────────────────────────────────────────────────────────────
    #  ÉTAPE 4 — Génération IFC
    # ─────────────────────────────────────────────────────────────────
    step("ÉTAPE 4 / 4 — Génération du fichier IFC")
    timings["ifc"] = run(
        [
            this_python,
            str(script_dir / "spatiallm_to_ifc.py"),
            "--input",  str(refined_layout),
            "--output", str(output_ifc),
        ],
        "Génération IFC",
    )

    # ── Résumé final ──
    total = time.time() - t_total
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Pipeline terminé avec succès                   ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  IFC généré : {output_ifc}")
    print("\n  Temps par étape :")
    for k, v in timings.items():
        print(f"    {k:<15} {v:>7.1f}s")
    print(f"    {'TOTAL':<15} {total:>7.1f}s")
    print()

    # Fichiers intermédiaires
    print("  Fichiers intermédiaires conservés :")
    for f in [clean_ply, raw_layout, refined_layout]:
        if Path(f).exists():
            size = Path(f).stat().st_size
            print(f"    {f}  ({size:,} octets)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Pipeline interrompu par l'utilisateur.", file=sys.stderr)
        sys.exit(130)
