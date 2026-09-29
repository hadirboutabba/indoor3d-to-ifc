# Tester Flash Attention avec SpatialLM

Procédure pour activer et tester Flash Attention sur une machine qui le supporte.
Écrite le 2026-09-24, d'après l'état du dépôt (branche `feature/optimized`, commit `0cecf0e`).

## 1. À savoir avant de commencer

- **Flash Attention n'est utilisé que par l'encodeur Sonata** (nuage de points vers vecteurs), une seule fois par scan.
  Il n'intervient pas dans la génération du texte par le LLM (Llama ou Qwen).
- Il donne un gain de **mémoire GPU et de vitesse** sur l'encodeur. Il ne change pas ce que le modèle comprend.
  Il calcule en float16, donc le texte produit peut différer légèrement du mode standard (float32).
- **Le code désactive Flash en dur** : `enable_flash=False` aux lignes 319, 526 et 826 de
  `spatiallm/model/sonata_encoder.py` (le dépôt officiel manycore-research/SpatialLM a `True`).
- Il n'y a **aucune détection automatique** du matériel dans le code, malgré ce que disent le README et la DOCUMENTATION.
  Il faut installer `flash-attn` ET passer les trois `False` à `True`. L'un sans l'autre ne sert à rien :
  - installer seulement : Flash reste inactif ;
  - passer à `True` sans installer : plantage `Make sure flash_attn is installed.`
- **GPU requis : Ampere ou plus récent** (capacité de calcul 8.0 ou plus : RTX 30xx/40xx, A100...).
  Les cartes Turing (RTX 20xx, capacité 7.5) ne sont pas supportées.
- L'image Docker actuelle **n'a pas** `flash-attn` (base `ubuntu:22.04`, sans `nvcc`, `TORCH_CUDA_ARCH_LIST="7.5"`).

## 2. Vérifier le GPU

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Il faut `compute_cap` >= 8.0. Sinon, arrêtez-vous ici.

## 3. Test en local avec conda (recommandé)

### 3.1 Branche de test

```bash
git checkout -b test/flash-attn
```

### 3.2 Vérifier torch AVANT d'installer flash-attn

```bash
conda activate spatiallm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Attendu : `2.4.1+cu124 True` (version du `pyproject.toml` et du `poetry.lock`).
**Ne modifiez plus `torch` après l'installation de `flash-attn`** : les extensions compilées sont liées à la version exacte
de torch (voir la section 8).

### 3.3 Installer flash-attn

```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

- `pip` télécharge un binaire précompilé s'il en existe un pour votre torch. Sinon il compile (nvcc CUDA 12.4 requis, 20 à 40 min).
- Si la RAM est limitée : `MAX_JOBS=4 pip install flash-attn --no-build-isolation`.

Vérification :

```bash
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
```

### 3.4 Activer Flash dans le code

```bash
grep -c "enable_flash=False" spatiallm/model/sonata_encoder.py     # doit afficher 3
sed -i 's/enable_flash=False/enable_flash=True/' spatiallm/model/sonata_encoder.py
git diff --stat                                                    # 3 lignes modifiées, 1 fichier
```

### 3.5 Vérifier que Flash est réellement actif

```bash
python - <<'EOF'
from spatiallm.model.sonata_encoder import Sonata
m = Sonata(in_channels=6, enable_fourier_encode=True)
print("enable_flash =", m.enc.enc0.block0.attn.enable_flash)
EOF
```

Attendu : `enable_flash = True`.

## 4. Lancer le pipeline

Pour une comparaison propre, utilisez un `clean.ply` déjà prétraité (le prétraitement contient un RANSAC non seedé qui ajoute de la variation).

### 4.1 Inférence seule (recommandé pour comparer)

```bash
/usr/bin/time -v python inference.py -p data/clean.ply -o out_flash.txt --seed 42 2> time_flash.txt
grep -E "Elapsed|Maximum resident" time_flash.txt
```

Dans un second terminal, pour suivre la mémoire GPU :

```bash
watch -n 1 nvidia-smi
```

### 4.2 Pipeline complet (PLY brut vers IFC)

```bash
python run_pipeline.py --input data/scan.ply --output out_flash.ifc --seed 42
```

### 4.3 Interface Streamlit

```bash
streamlit run pipeline_gui.py
```

La GUI appelle `inference.py` (`pipeline_gui.py:427`) et utilise donc le code modifié. Champ "Seed" : 42.
Elle ne donne pas de mesure de mémoire propre, d'où la préférence pour 4.1.

## 5. Comparer avec et sans Flash

Même `clean.ply`, même seed, même machine.

```bash
# SANS Flash : remettre les False
sed -i 's/enable_flash=True/enable_flash=False/' spatiallm/model/sonata_encoder.py
/usr/bin/time -v python inference.py -p data/clean.ply -o out_noflash.txt --seed 42 2> time_noflash.txt
grep -E "Elapsed|Maximum resident" time_noflash.txt

# AVEC Flash : remettre les True
sed -i 's/enable_flash=False/enable_flash=True/' spatiallm/model/sonata_encoder.py

# Comparer
diff out_flash.txt out_noflash.txt | head
wc -l out_flash.txt out_noflash.txt
```

Comment lire :

| Observation | Signification |
|---|---|
| Mémoire GPU ou temps plus faibles avec Flash | Gain attendu. Sur l'encodeur seul l'écart peut être faible : la génération du LLM domine probablement le temps total (non mesuré). |
| Textes identiques | Bon signe. |
| Textes légèrement différents | Normal (float16). Pas une erreur. |
| Lignes répétées en fin de fichier | Boucle de décodage (voir section 9), sans lien direct avec Flash. |

Notez aussi les versions utilisées : `pip list | grep -i -E "^(torch|transformers|spconv|flash)"`.

## 6. Test dans l'image Docker (moins fiable)

L'image n'a pas `nvcc` : l'installation ne marche que si `pip` trouve un binaire précompilé pour Python 3.11, torch 2.4, CUDA 12
(non vérifié). Sinon elle échoue.

```bash
docker compose run --rm spatiallm bash
```

Dans le conteneur :

```bash
pip install ninja
pip install flash-attn --no-build-isolation
sed -i 's/enable_flash=False/enable_flash=True/' /app/spatiallm/model/sonata_encoder.py
python inference.py -p /app/data/clean.ply -o /app/output/test_flash.txt --seed 42
```

Tout est perdu à la sortie (`--rm`). Pour un support durable dans Docker, il faudra modifier le Dockerfile :
base `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` (celle de la branche `main`) et `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"`.
À faire seulement après un test local concluant.

## 7. Revenir à l'état d'origine

```bash
git checkout -- spatiallm/model/sonata_encoder.py
pip uninstall -y flash-attn        # seulement sur une machine qui ne le supporte pas
```

## 8. Problèmes connus

### `undefined symbol: _ZNK3c105Error4whatEv` dans `flash_attn_2_cuda`, ou `undefined symbol ... parseSchemaOrName` dans `torch_scatter`

Cause : les extensions compilées (`flash-attn`, `torch_scatter`) ne correspondent pas à la version de `torch` installée.
Rencontré le 2026-09-24 sur la machine WSL : `torch` avait été redescendu en 2.3.1+cu121 alors que `torch_scatter`
(`pt24cu124`) et `torchaudio` (2.4.1+cu124) étaient restés en version 2.4.

`transformers` importe `flash_attn` tout seul dès qu'il est installé, avant le code de SpatialLM : un `flash-attn` cassé
fait donc planter même le pipeline sans Flash.

Correctif :

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch, torch_scatter, spconv.pytorch; print(torch.__version__, torch.cuda.is_available())"
```

Attendu : `2.4.1+cu124 True`. Sur une machine sans Flash, gardez `flash-attn` désinstallé.

Prévention : `pip freeze > ~/env_spatiallm_ok.txt` quand tout marche, et `pip install --dry-run <paquet>` avant
d'installer un nouveau paquet, pour voir s'il veut changer `torch`.

### `ModuleNotFoundError: No module named 'timm.layers'`

Cause : `sonata_encoder.py:31` fait `from timm.layers import DropPath`, qui n'existe que dans `timm` >= 0.9.
L'environnement avait `timm 0.6.13` (rétrogradé avec `torch`).
Correctif appliqué le 2026-09-24 : `pip install "timm>=0.9.12"` (a installé `timm 1.0.30`, sans toucher à `torch`).

### Environnement pollué par d'autres projets (constaté le 2026-09-24)

Dans l'environnement conda `spatiallm` de la machine WSL, `pip` signalait des paquets qui n'appartiennent pas à SpatialLM :
`xformers 0.0.27` (impose `torch==2.3.1`) et `image-reward 1.5` (impose `timm==0.6.13`).
Ils ont probablement causé la rétrogradation de `torch` et `timm`, puis les erreurs ci-dessus (cause probable, non prouvée).
Après le correctif, `image-reward` est en conflit avec `timm 1.0.30` : s'il est encore utilisé, lui donner son propre environnement.
Bonne pratique : un environnement conda dédié à SpatialLM, et ne rien y installer d'autre.

### `AssertionError: Make sure flash_attn is installed.`

`enable_flash=True` mais `flash-attn` absent. Installez-le (3.3) ou remettez `False` (7).

### Échec de compilation de `flash-attn`

Vérifiez `nvcc --version` (12.4 attendu), la RAM (`MAX_JOBS=4`) et que le GPU est bien Ampere ou plus.

## 9. Autres constats liés à ce test (hors Flash)

- **Boucles de décodage** : `redaPLY_subsampled_clean_layout.txt` contient 68 lignes `Bbox(dining_table,...)` identiques
  et consécutives (lignes 115 à 182). C'est une boucle du décodage glouton, sans lien avec la vitesse. Ce fichier n'est pas
  produit par Flash.
- **Résultats différents sur le même fichier** : le nettoyage d'`inference.py` est déterministe (testé 8 fois sur `room.ply`).
  La variation vient donc de l'encodeur (`shuffle_orders=True`, calculs GPU) ou du LLM (bfloat16 + décodage glouton).
  Le prétraitement `preprocess_for_spatiallm.py` est lui non déterministe (RANSAC non seedé, écart de plusieurs dizaines
  de mètres mesuré sur `room.ply` d'un lancement à l'autre).
- Les arguments `--temperature`, `--top_k`, `--top_p` d'`inference.py` sont ignorés (`do_sample=False` est écrit en dur).
- Le commentaire du Dockerfile (ligne 50) cite `flash_attn_utils.py`, qui n'existe pas. Le README et la DOCUMENTATION
  décrivent une détection automatique qui n'existe pas dans le code.

## 10. Idée d'amélioration (non faite)

Remplacer les trois `False` par une option `--flash_attn {on,off,auto}` (défaut `off`) pour ne plus modifier le fichier à la main.
`auto` testerait la capacité du GPU (>= 8.0) et la présence de `flash_attn`.
