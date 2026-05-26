# Documentation — Pipeline PLY → IFC avec SpatialLM

## Vue d'ensemble

Ce projet implémente un pipeline complet de reconstruction BIM à partir d'un nuage de points 3D.  
Il enchaîne automatiquement quatre étapes pour produire un fichier IFC standard à partir d'un scan `.ply` brut.

```
scan brut (.ply)
      │
      ▼ preprocess_for_spatiallm.py
nuage nettoyé (_clean.ply)
      │
      ▼ inference.py
layout brut (_raw_layout.txt)
      │
      ▼ postprocess.py
layout raffiné (_refined_layout.txt)
      │
      ▼ spatiallm_to_ifc.py
fichier BIM (.ifc)
```

---

## Format TXT — Sortie SpatialLM

Tous les scripts partagent le même format de fichier texte produit par SpatialLM :

```
wall_0=Wall(x1, y1, z1, x2, y2, z2, height, thickness)
door_0=Door(parent_wall, cx, cy, cz, width, height)
window_0=Window(parent_wall, cx, cy, cz, width, height)
bbox_0=Bbox(label, cx, cy, cz, rotation_rad, length, width, height)
```

**Signification des paramètres :**

| Type | Paramètre | Unité | Description |
|------|-----------|-------|-------------|
| Wall | x1,y1,z1 | m | Point de départ dans le repère monde |
| Wall | x2,y2,z2 | m | Point d'arrivée |
| Wall | height | m | Hauteur du mur |
| Wall | thickness | m | Épaisseur (souvent 0.0 — SpatialLM ne la prédit pas toujours) |
| Door/Window | parent_wall | — | Nom du mur hôte (ex: `wall_1`) |
| Door/Window | cx,cy,cz | m | Centre de l'ouverture dans le repère monde |
| Door/Window | width,height | m | Dimensions de l'ouverture |
| Bbox | label | — | Catégorie de l'objet (bed, chair, desk…) |
| Bbox | cx,cy,cz | m | Centre de l'objet |
| Bbox | rotation_rad | rad | Rotation autour de Z (sens anti-horaire depuis X) |
| Bbox | length,width,height | m | Dimensions (length = plus longue dimension horizontale) |

**Exemple réel :**
```
wall_1=Wall(-0.436,-1.986,0.25, 1.789,-0.136,0.25, 2.14, 0.0)
window_0=Window(wall_1, 0.914,-0.786,1.55, 1.0, 1.02)
bbox_0=Bbox(chair,-0.386,-1.136,0.7, -0.785, 0.5625,0.515625,0.875)
```

---

## Scripts

### 1. `preprocess_for_spatiallm.py` — Nettoyage du nuage de points

**Rôle :** Prépare un `.ply` brut pour l'inférence SpatialLM.  
**Opérations :** débruitage (Statistical Outlier Removal), alignement Z-up + Manhattan, mise à l'échelle métrique, normalisation couleurs.

**Fichier d'origine — non modifié.**

```bash
python preprocess_for_spatiallm.py \
    -i raw_scan.ply \
    -o clean_scan.ply \
    --mode moderate \
    --target_height 2.5 \
    --keep_largest_cluster \
    --normalize_colors
```

| Argument | Défaut | Description |
|----------|--------|-------------|
| `-i` / `--input` | requis | Fichier `.ply` d'entrée |
| `-o` / `--output` | requis | Fichier `.ply` nettoyé |
| `--mode` | `moderate` | Intensité du nettoyage : `conservative`, `moderate`, `aggressive` |
| `--target_height` | `2.5` | Hauteur cible du plafond (m) pour la mise à l'échelle |
| `--no_denoise` | False | Ignorer le débruitage |
| `--no_align` | False | Ignorer l'alignement automatique |
| `--keep_largest_cluster` | False | Garder uniquement le plus grand cluster DBSCAN |
| `--normalize_colors` | False | Normaliser la distribution RGB |

---

### 2. `inference.py` — Inférence SpatialLM

**Rôle :** Analyse le nuage de points nettoyé et produit le layout TXT.  
**Modèle :** LLaMA 1B ou Qwen 0.5B + encodeur de points Sonata.

**Fichier d'origine — non modifié.**

```bash
python inference.py \
    -p clean_scan.ply \
    -o layout.txt \
    --model_path manycore-research/SpatialLM-Llama-1B \
    --detect_type all
```

| Argument | Défaut | Description |
|----------|--------|-------------|
| `-p` / `--point_cloud` | requis | Nuage de points `.ply` |
| `-o` / `--output` | requis | Fichier TXT de sortie |
| `-m` / `--model_path` | `SpatialLM-Llama-1B` | Chemin local ou nom HuggingFace |
| `-d` / `--detect_type` | `all` | `all` / `arch` (murs+portes+fenêtres) / `object` (bbox) |

**Modèles disponibles :**
- `manycore-research/SpatialLM1.1-Llama-1B` — précision maximale (~6 GB VRAM)
- `manycore-research/SpatialLM1.1-Qwen-0.5B` — rapide, moins de VRAM

---

### 3. `postprocess.py` — Correction géométrique *(nouveau)*

**Rôle :** Corrige les erreurs géométriques courantes du layout SpatialLM avant la génération IFC.  
**Dépendances :** numpy, scipy, open3d.

```bash
python postprocess.py \
    --layout raw_layout.txt \
    --point_cloud clean_scan.ply \
    --output refined_layout.txt
```

| Argument | Défaut | Description |
|----------|--------|-------------|
| `-l` / `--layout` | requis | Layout TXT brut (sortie SpatialLM) |
| `-p` / `--point_cloud` | requis | Nuage de points nettoyé |
| `-o` / `--output` | requis | Layout TXT corrigé |
| `--angle_thresh` | `10.0°` | Seuil d'angle pour détecter la colinéarité de deux murs |
| `--dist_thresh` | `0.35 m` | Distance perpendiculaire max pour fusionner deux murs |
| `--gap_thresh` | `0.60 m` | Espace max entre les projections de deux murs pour les fusionner |
| `--min_points` | `50` | Nombre minimum de points dans une bbox pour la réaffiner |

**Bibliothèques utilisées :**

| Bibliothèque | Usage dans ce script |
|---|---|
| `numpy` | Toutes les opérations vectorielles et matricielles (projections, rotations, coins de bbox) |
| `scipy.spatial.ConvexHull` | Enveloppe convexe 2D pour le rotating calipers (réaffinage bbox) |
| `scipy.spatial.Delaunay` | Test d'appartenance de points à l'intérieur d'une bbox 3D |
| `open3d` | Lecture du fichier `.ply` et accès aux points |

---

**Algorithme 1 — Détection de murs colinéaires**

SpatialLM divise parfois un mur réel en deux segments. La détection utilise trois tests géométriques en cascade — tous les trois doivent passer :

```
are_collinear(w1, w2) :

  Test 1 — Angle (mod 180° pour ignorer le sens)
    a1 = atan2(dy1, dx1) mod 180°
    a2 = atan2(dy2, dx2) mod 180°
    diff = min(|a1−a2|, 180−|a1−a2|)
    → diff < angle_thresh (défaut 10°)

  Test 2 — Distance perpendiculaire
    midpoint de w2 : mx = (x1+x2)/2, my = (y1+y2)/2
    d = |−dy1·(mx−x1) + dx1·(my−y1)| / ‖(dx1,dy1)‖
    → d < dist_thresh (défaut 0.35 m)

  Test 3 — Écart de projection sur l'axe commun
    Projeter les 4 extrémités sur l'axe unitaire de w1
    gap = max(0, début_segment_droit − fin_segment_gauche)
    → gap < gap_thresh (défaut 0.60 m)
```

**Algorithme 2 — Fusion de murs (merge)**

Si les trois tests passent, les deux murs sont remplacés par un seul couvrant la portée totale :

```
merge_walls(w1, w2) :
  Projeter les 4 extrémités sur l'axe unitaire de w1
  t_min = projection minimale → new_start
  t_max = projection maximale → new_end
  new_start = origin_w1 + t_min × unit_vec(w1)
  new_end   = origin_w1 + t_max × unit_vec(w1)
  hauteur   = max(h1, h2)
  épaisseur = max(t1, t2) ou 0.2 m
```

La fusion est **itérative** : on recommence jusqu'à ce qu'aucune nouvelle paire colinéaire ne soit trouvée.

---

**Algorithme 3 — Test points-dans-bbox 3D (Delaunay)**

Pour extraire les points du nuage réellement à l'intérieur d'une bounding box orientée :

```
bbox_corners_3d(b) :
  8 coins locaux (±l/2, ±w/2, ±h/2) rotés autour de Z par angle b.rotation
  + translatés au centre (cx, cy, cz)

points_in_convex_hull(corners, points) :
  hull = Delaunay(8 coins)
  mask = hull.find_simplex(points) >= 0
  → points[mask]
```

Delaunay est utilisé comme test d'inclusion dans un polytope convexe 3D arbitraire. Plus robuste qu'un AABB car il respecte la rotation de la bbox.

---

**Algorithme 4 — Rotating calipers (minimum bounding box 2D)**

Pour réaffiner les dimensions et l'orientation d'une bbox sur les points réels du nuage :

```
fit_min_bbox_2d(points) :
  1. Projeter les points en 2D (ignorer Z)
  2. ConvexHull(points 2D) → N arêtes du hull
  3. Pour chaque arête du hull :
       a. Définir un repère local (axe = direction de l'arête)
       b. Projeter tous les points dans ce repère
       c. Calculer le rectangle englobant (min/max sur chaque axe)
       d. Mesurer l'aire = largeur × longueur
  4. Garder l'arête qui minimise l'aire
  → angle optimal, nouveau centre (cx, cy), longueur, largeur
  Z et hauteur conservés depuis SpatialLM
```

Complexité O(n log n) dominée par le convex hull. C'est l'algorithme standard **minimum area bounding rectangle**.

---

### 4. `spatiallm_to_ifc.py` — Génération IFC *(modifié)*

**Rôle :** Convertit le layout TXT raffiné en fichier IFC4 standard.  
**Dépendances :** ifcopenshell, numpy, scipy.

```bash
python spatiallm_to_ifc.py \
    --input refined_layout.txt \
    --output scene.ifc
```

| Argument | Défaut | Description |
|----------|--------|-------------|
| `-i` / `--input` | requis | Layout TXT (brut ou raffiné) |
| `-o` / `--output` | `output.ifc` | Fichier IFC de sortie |

**Structure IFC générée :**

```
IfcProject
  └─ IfcSite
       └─ IfcBuilding
            └─ IfcBuildingStorey  ("Ground Floor")
                 ├─ IfcSlab           (dalle de sol — convex hull des murs)
                 ├─ IfcWall           (×N murs)
                 ├─ IfcDoor           (×N portes)
                 ├─ IfcWindow         (×N fenêtres)
                 └─ IfcFurnishingElement  (×N objets Bbox)
```

**Valeurs utilisées dans le IFC :**

| Élément | Attribut IFC | Source | Remarque |
|---------|-------------|--------|----------|
| Mur | longueur | calculée `‖(x2,y2)−(x1,y1)‖` | valeur réelle |
| Mur | hauteur | `height` du TXT | valeur réelle |
| Mur | épaisseur | `thickness` du TXT | 0.2 m si TXT = 0.0 |
| Mur | position | `(x1, y1, z1)` | valeur réelle |
| Mur | orientation | `atan2(y2−y1, x2−x1)` | valeur réelle |
| Porte | largeur | `width` du TXT | valeur réelle |
| Porte | hauteur | `height` du TXT | valeur réelle |
| Porte | profondeur | épaisseur du mur parent | valeur réelle |
| Porte | position (bas) | `(cx, cy, cz − height/2)` | valeur réelle |
| Porte | orientation | direction du mur parent | valeur réelle |
| Fenêtre | largeur | `width` du TXT | valeur réelle |
| Fenêtre | hauteur | `height` du TXT | valeur réelle |
| Fenêtre | profondeur | épaisseur du mur parent | valeur réelle |
| Fenêtre | position (bas) | `(cx, cy, cz − height/2)` | valeur réelle |
| Fenêtre | orientation | direction du mur parent | valeur réelle |
| Bbox | longueur/largeur/hauteur | `length`, `width`, `height` | valeurs réelles |
| Bbox | position (bas) | `(cx, cy, cz − height/2)` | valeur réelle |
| Bbox | rotation | `rotation` en radians | valeur réelle |
| Dalle | polygone | convex hull des extrémités de murs | calculé |
| Dalle | épaisseur | 0.10 m (fixe) | — |

**Bibliothèques utilisées :**

| Bibliothèque | Usage dans ce script |
|---|---|
| `ifcopenshell` | Création de toutes les entités IFC (géométrie, hiérarchie, relations, GUID) |
| `numpy` | Construction du tableau de points pour le convex hull de la dalle |
| `scipy.spatial.ConvexHull` | Polygone du périmètre de la dalle de sol |
| `math` | `atan2`, `cos`, `sin` pour les orientations de murs et d'objets |
| `uuid` | Génération des GUID IFC (identifiants uniques de chaque entité) |

---

**Algorithme 1 — Placement local IFC (IfcAxis2Placement3D)**

Chaque élément est positionné dans un repère local défini par trois vecteurs :

```
IfcAxis2Placement3D :
  origine : point 3D de l'élément
  axe Z   : toujours (0, 0, 1) — vertical mondial
  axe X   : direction principale de l'élément (horizontal)

Calcul de l'axe X selon le type :
  Mur     → θ = atan2(y2−y1, x2−x1)  →  (cos θ, sin θ, 0)
  Porte   → direction du mur parent    →  wall_dir_map[parent]
  Fenêtre → direction du mur parent    →  wall_dir_map[parent]
  Bbox    → rotation SpatialLM (rad)   →  (cos r, sin r, 0)
```

L'axe Y est déduit automatiquement par le produit vectoriel Z × X.

---

**Algorithme 2 — Géométrie par extrusion (IfcExtrudedAreaSolid)**

Tous les solides sont construits par la même méthode :

```
1. IfcRectangleProfileDef(largeur, profondeur)
     → profil rectangulaire 2D centré à l'origine locale
2. IfcExtrudedAreaSolid(profil, direction=(0,0,1), longueur=hauteur)
     → extrusion verticale du profil
3. IfcShapeRepresentation → IfcProductDefinitionShape
     → représentation géométrique attachée à l'entité IFC

Dimensions utilisées :
  Mur     : largeur = ‖(x2,y2)−(x1,y1)‖, profondeur = thickness, hauteur = height
  Porte   : largeur = width, profondeur = épaisseur mur parent, hauteur = height
  Fenêtre : largeur = width, profondeur = épaisseur mur parent, hauteur = height
  Bbox    : largeur = length, profondeur = width, hauteur = height
```

---

**Algorithme 3 — Dalle de sol (Convex hull + extrusion)**

```
1. Collecter tous les points XY des extrémités de murs :
     pts = [(w.x1, w.y1), (w.x2, w.y2)] pour chaque mur
2. ConvexHull(pts) → indices des vertices ordonnés
3. IfcPolyline([p0, p1, ..., pN, p0])  ← fermeture du polygone
4. IfcArbitraryClosedProfileDef(polyline) → profil polygonal
5. Extrusion de 10 cm vers le haut
6. Placement à z = z_min − 0.10 m  ← sous le plancher des murs
```

Le convex hull garantit que la dalle couvre exactement l'emprise de la pièce quelle que soit sa forme.

---

**Algorithme 4 — Direction et épaisseur des murs (lookups)**

Avant de traiter les portes et fenêtres, deux dictionnaires sont construits :

```python
wall_dir_map[w.name]       = (dx/‖d‖, dy/‖d‖)           # vecteur unitaire
wall_thickness_map[w.name] = w.thickness if > 0.001 else 0.2
```

Chaque porte/fenêtre interroge ces dictionnaires via son champ `parent_wall` pour obtenir l'orientation et l'épaisseur correctes de son mur hôte.

---

**Modifications apportées au fichier d'origine :**

1. **Dalle de sol automatique** — ajout d'un `IfcSlab` généré par convex hull des extrémités de murs (scipy `ConvexHull`) → `IfcArbitraryClosedProfileDef` extrudé de 10 cm.

2. **Orientation correcte des portes/fenêtres** — `wall_dir_map` construit depuis les données de murs, utilisé comme axe X local du placement. Avant le fix : axe X toujours `(1,0,0)` → portes perpendiculaires aux murs.

3. **Position verticale correcte des fenêtres** — placement à `cz − height/2` (bas de la fenêtre). Avant le fix : placement à `cz` interprété comme le bas → fenêtres décalées vers le haut.

4. **Profondeur réelle des portes/fenêtres** — `wall_thickness_map` utilisé à la place des valeurs codées en dur (`0.1 m` pour les portes, `0.05 m` pour les fenêtres).

**Limitation connue :** les portes et fenêtres sont des solides positionnés dans le mur, pas des ouvertures réelles. En IFC complet, il faudrait un `IfcOpeningElement` + `IfcRelFillsElement`. Ce niveau de détail nécessiterait une géométrie de découpe que SpatialLM ne fournit pas.

---

### 5. `run_pipeline.py` — Orchestrateur CLI *(nouveau)*

**Rôle :** Enchaîne les 4 étapes en une seule commande.

```bash
# Commande minimale
python run_pipeline.py --input raw_scan.ply --output scene.ifc

# Commande complète
python run_pipeline.py \
    --input  raw_scan.ply \
    --output scene.ifc \
    --workdir /tmp/pipeline \
    --model  manycore-research/SpatialLM1.1-Qwen-0.5B \
    --detect_type all \
    --preprocess_mode moderate \
    --target_height 2.5 \
    --skip_preprocess \
    --skip_postprocess \
    --inference_python /path/to/spatiallm-env/bin/python
```

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--input` / `-i` | requis | Fichier `.ply` brut |
| `--output` / `-o` | `<stem>.ifc` | Fichier IFC de sortie |
| `--workdir` / `-w` | dossier de `--input` | Dossier pour les fichiers intermédiaires |
| `--model` / `-m` | `SpatialLM-Llama-1B` | Modèle SpatialLM |
| `--detect_type` | `all` | Type de détection |
| `--preprocess_mode` | `moderate` | Mode de nettoyage |
| `--target_height` | `2.5` | Hauteur cible du plafond |
| `--skip_preprocess` | False | Utiliser le `.ply` tel quel |
| `--skip_postprocess` | False | Sauter la correction géométrique |
| `--inference_python` | `sys.executable` | Interpréteur Python alternatif pour l'inférence |
| `--angle_thresh` | `10.0°` | Seuil fusion murs |
| `--dist_thresh` | `0.35 m` | Seuil distance murs |
| `--gap_thresh` | `0.60 m` | Seuil espace murs |
| `--min_points` | `50` | Points min pour raffinage bbox |

**Fichiers intermédiaires produits :**

```
<workdir>/
  <stem>_clean.ply            ← nuage nettoyé
  <stem>_raw_layout.txt       ← layout brut SpatialLM
  <stem>_refined_layout.txt   ← layout corrigé
<output>                      ← fichier IFC final
```

**Note sur les environnements :** l'inférence SpatialLM nécessite PyTorch + Transformers.  
Si ces dépendances sont dans un conda séparé :
```bash
python run_pipeline.py --input scan.ply \
    --inference_python /opt/conda/envs/spatiallm/bin/python
```

---

### 6. `pipeline_gui.py` — Interface graphique *(nouveau)*

**Rôle :** Interface Streamlit pour exécuter le pipeline complet sans ligne de commande.

```bash
cd /mnt/c/Users/hboutabb/SpatialLM
streamlit run pipeline_gui.py
# → ouvrir http://localhost:8501 dans le navigateur Windows
```

**Fonctionnalités :**

- Upload du `.ply` par glisser-déposer
- Contrôle de chaque étape via la sidebar (avec option de les ignorer)
- Log en temps réel de chaque script pendant l'exécution
- Comparaison layout brut vs raffiné (comptage des éléments avec deltas)
- Maquette 3D interactive (Plotly) avec sol, murs, portes, fenêtres et objets
- Viewer Rerun embarqué pour le nuage de points + détections
- Téléchargements : PLY nettoyé, layout TXT brut, layout TXT raffiné, fichier IFC

**Paramètres disponibles dans la sidebar :**

| Section | Paramètre |
|---------|-----------|
| Preprocessing | Mode (Complet / Alignement uniquement / Ignorer), nettoyage, cluster, couleurs, hauteur cible |
| Inférence | Modèle (Llama 1B / Qwen 0.5B), type de détection, seed |
| Post-processing | Ignorer / activer, seuils (angle, distance, espace, points min) |
| IFC | Ignorer / activer |

---

### 7. `spatiallm_gui.py` — Interface graphique originale

**Fichier d'origine — non modifié.**  
Interface 3 étapes : Preprocessing → Inférence → Visualisation RRD.  
Remplacé fonctionnellement par `pipeline_gui.py` qui ajoute post-processing + IFC.

---

## Corrections et bugs résolus

### Bug 1 — Portes et fenêtres perpendiculaires aux murs

**Symptôme :** dans le viewer IFC, les portes et fenêtres étaient orientées à 90° par rapport à leur mur.

**Cause :** l'axe X du placement local était toujours `(1, 0, 0)` (direction globale X), indépendamment de l'orientation réelle du mur.

**Fix :**
```python
# Construction du dictionnaire direction de mur
wall_dir_map[w.name] = (dx/ln, dy/ln)  # vecteur normalisé

# Utilisation pour chaque porte/fenêtre
dir_x, dir_y = wall_dir_map.get(d.parent_wall, (1.0, 0.0))
placement = IfcAxis2Placement3D(
    origin=(cx, cy, cz),
    z_axis=(0,0,1),
    x_axis=(dir_x, dir_y, 0.0),   # ← direction réelle du mur
)
```

---

### Bug 2 — Fenêtres décalées vers le haut

**Symptôme :** les fenêtres dépassaient le haut des murs.

**Cause :** SpatialLM donne `cz` comme le **centre** de la fenêtre, mais le script plaçait le **bas** à `cz` sans correction.

**Fix :**
```python
# Avant
model.createIfcCartesianPoint((wn.cx, wn.cy, wn.cz))

# Après
model.createIfcCartesianPoint((wn.cx, wn.cy, wn.cz - wn.height / 2))
```

**Exemple :** `window_0` avec `cz=1.55m`, `height=1.02m`, mur top à `2.39m` :
- Avant fix : fenêtre de `1.55m` à `2.57m` → dépasse de 18 cm
- Après fix : fenêtre de `1.04m` à `2.06m` → dans le mur

---

### Bug 3 — Profondeur des portes/fenêtres incorrecte

**Symptôme :** portes et fenêtres trop fines (5–10 cm) au lieu de l'épaisseur réelle du mur.

**Cause :** profondeur codée en dur (`0.1 m` pour portes, `0.05 m` pour fenêtres).

**Fix :**
```python
wall_thickness_map[w.name] = w.thickness if w.thickness > 0.001 else 0.2

# Utilisation
wall_t = wall_thickness_map.get(d.parent_wall, 0.2)
profile = IfcRectangleProfileDef(width=d.width, depth=wall_t)
```

---

## Dépendances

| Bibliothèque | Version | Rôle |
|-------------|---------|------|
| `torch` + `transformers` | selon HF | Modèle SpatialLM (inférence) |
| `open3d` | ≥ 0.19 | Lecture PLY, débruitage, visualisation |
| `numpy` | ≥ 2.3 | Calculs géométriques |
| `scipy` | ≥ 1.16 | ConvexHull, Delaunay, rotating calipers |
| `ifcopenshell` | ≥ 0.8.5 | Génération IFC |
| `streamlit` | ≥ 1.3 | Interface graphique |
| `plotly` | ≥ 5.0 | Maquette 3D dans l'interface |
| `rerun-sdk` | ≥ 0.15 | Viewer nuage de points |

**Environnement recommandé :** Python 3.12 (open3d ne supporte pas Python 3.13+).

```bash
conda create -n spatiallm python=3.12
conda activate spatiallm
pip install -e .   # dans le dossier pystruct3d si utilisé
pip install ifcopenshell streamlit plotly rerun-sdk
```

---

## Chemins configurés

| Variable | Valeur |
|----------|--------|
| Dossier SpatialLM | `/mnt/c/Users/hboutabb/SpatialLM` |
| Dossier scans | `C:\Users\hboutabb\scanbureau\` |
| Modèle Llama 1B | `/mnt/c/Users/hboutabb/SpatialLM1.1-Llama-1B` |
| Modèle Qwen 0.5B | `/mnt/c/Users/hboutabb/SpatialLM1.1-Qwen-0.5B` |

---

## Exemple de session complète

```bash
# 1. Lancer l'interface graphique
cd /mnt/c/Users/hboutabb/SpatialLM
streamlit run pipeline_gui.py

# OU en ligne de commande :

# 1. Nettoyage
python preprocess_for_spatiallm.py \
    -i /mnt/c/Users/hboutabb/scanbureau/scan_brut.ply \
    -o /mnt/c/Users/hboutabb/scanbureau/scan_clean.ply \
    --mode moderate --target_height 2.5 --keep_largest_cluster

# 2. Inférence SpatialLM
python inference.py \
    -p /mnt/c/Users/hboutabb/scanbureau/scan_clean.ply \
    -o /mnt/c/Users/hboutabb/scanbureau/layout_raw.txt \
    --model_path /mnt/c/Users/hboutabb/SpatialLM1.1-Llama-1B

# 3. Correction géométrique
python postprocess.py \
    --layout  /mnt/c/Users/hboutabb/scanbureau/layout_raw.txt \
    --point_cloud /mnt/c/Users/hboutabb/scanbureau/scan_clean.ply \
    --output  /mnt/c/Users/hboutabb/scanbureau/layout_refined.txt

# 4. Génération IFC
python spatiallm_to_ifc.py \
    --input  /mnt/c/Users/hboutabb/scanbureau/layout_refined.txt \
    --output /mnt/c/Users/hboutabb/scanbureau/scene.ifc

# OU tout en une commande :
python run_pipeline.py \
    --input  /mnt/c/Users/hboutabb/scanbureau/scan_brut.ply \
    --output /mnt/c/Users/hboutabb/scanbureau/scene.ifc
```
