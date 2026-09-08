# Prompt à copier-coller dans un autre LLM (Gemini, ChatGPT, etc.)

Copie tout le bloc ci-dessous dans une nouvelle conversation. Il contient le contexte,
les règles de style, et le contenu exact des 4 schémas. Si le LLM cible n'arrive pas à
produire les 4 d'un coup, colle-le une première fois, puis redemande "genere maintenant
le schema 2" etc. dans la meme conversation pour qu'il garde le contexte et le style.

---

## CONTEXTE

Je rédige un rapport de projet de fin d'études (PFE) sur un pipeline qui convertit un
nuage de points 3D (scan d'interieur) en maquette numerique au format IFC (BIM), en
utilisant un modele de langage multimodal appele SpatialLM. J'ai besoin de 4 schemas
techniques pour illustrer ce pipeline dans mon rapport et ma soutenance.

## REGLES DE STYLE (obligatoires pour les 4 schemas)

- Style sobre et professionnel, comme une figure d'article scientifique ou de these
  d'ingenieur (IEEE, thesis, documentation technique). PAS un style "startup"/"slide
  marketing" : pas de degrades, pas d'ombres portees, pas d'icones ou d'emojis, pas de
  fond colore, pas de coins arrondis exageres, pas de palette multicolore decorative.
- Fond blanc. Traits et texte en noir/gris fonce. Une seule couleur d'accentuation
  (par exemple un bleu petrole ou bordeaux discret) reservee UNIQUEMENT aux fleches ou
  bordures qui representent le mecanisme cle explique par le schema (jamais utilisee
  pour decorer un element secondaire).
- Boites rectangulaires a coins legerement arrondis, bordure fine (1 a 1.5 pt), fond
  blanc, texte aligne a gauche (ou centre pour les boites courtes a une seule ligne).
- Police sans-serif standard (Arial/Helvetica) pour les titres de boites et les
  descriptions. Police a chasse fixe (monospace, type Courier/Consolas) UNIQUEMENT pour
  les noms de fichiers, noms de scripts, extraits de code, noms d'entites techniques
  (ex. IfcWall, wall_0=Wall(...)).
- Fleches simples avec une pointe standard (pas de fleches en forme de flamme ou
  d'eclair). Une etiquette courte au-dessus de la fleche quand elle represente un
  fichier ou une relation nommee.
- NE PAS mettre de titre du schema, de numero de figure, ni de phrase d'explication ou
  de legende dans l'image elle-meme : seulement les elements du diagramme (boites,
  fleches, textes techniques courts). Le titre et la legende seront ajoutes separement
  dans le corps du rapport.
- Le texte doit rester en francais (sauf les noms de fichiers, de scripts, et les noms
  d'entites/API qui sont du code et restent tels quels).
- Fournis chaque schema sous forme de code (SVG ou script Python/matplotlib) que je
  peux executer ou ouvrir moi-meme, plutot qu'une image bitmap generee, afin de garder
  un rendu vectoriel net et modifiable.

---

## SCHEMA 1 — Architecture generale du pipeline

Flux horizontal lineaire de 5 etapes puis un livrable final, connectees par des
fleches. La derniere boite (livrable) est visuellement distincte (bordure plus epaisse
ou couleur d'accent), les 5 premieres sont identiques en style.

Boites, dans l'ordre, chacune avec : un titre (gras), un nom de fichier/script en
monospace, et 1 a 2 lignes de description :

1. Titre : "Nuage de points brut" — monospace : `.ply` — description : "Scan 3D brut
   (LiDAR / photogrammetrie)"
2. Titre : "Pretraitement" — monospace : `preprocess_for_spatiallm.py` — description :
   "Filtrage statistique + DBSCAN + normalisation de hauteur (Open3D)"
3. Titre : "Inference SpatialLM" — monospace : `inference.py` — description :
   "SpatialLM1.1 (Llama-1B ou Qwen-0.5B)"
4. Titre : "Post-traitement" — monospace : `postprocess.py` — description : "Fusion
   des murs colineaires + reajustement des bounding boxes (numpy/scipy)"
5. Titre : "Generation IFC" — monospace : `spatiallm_to_ifc.py` — description :
   "ifcopenshell : dalle, murs, ouvertures, mobilier"
6. (livrable final, boite plus petite, bordure d'accent) Titre : "Modele IFC" —
   monospace : `.ifc`

Etiquettes des fleches (nom du fichier intermediaire reellement ecrit sur disque entre
deux etapes) :
- 1 → 2 : `scan.ply`
- 2 → 3 : `scan_clean.ply`
- 3 → 4 : `raw_layout.txt`
- 4 → 5 : `refined_layout.txt`
- 5 → 6 : pas d'etiquette (c'est le resultat final), fleche en couleur d'accent

---

## SCHEMA 2 — Architecture interne du modele SpatialLM

Objectif : montrer que les tokens issus du nuage de points ne sont pas juste
concatenes au texte : ils remplacent un marqueur special au milieu de la sequence de
tokens texte, avant d'etre traites par un LLM autoregressif classique.

Disposition : empilement vertical de 6 niveaux. Les niveaux 1 a 3 ont deux colonnes
paralleles (gauche = chemin "nuage de points", droite = chemin "texte/prompt") qui
fusionnent au niveau 4, puis une seule colonne pour les niveaux 5 et 6.

Niveau 1 (entrees), deux boites :
- Gauche : "Nuage de points pretraite" — description : "Coordonnees (x,y,z), couleur
  RGB, grille de voxels (grid_coord)" — monospace : `point_cloud : (n_points, n_features)`
- Droite : "Prompt texte" — contenu monospace litteral (3 lignes) :
  ```
  <point_start><point_pad><point_end>
  Detect walls, doors, windows, boxes.
  Reference code: {code_template}
  ```

Niveau 2 :
- Gauche : "Encodeur de nuage de points" — monospace (2 lignes) : "Sonata (point
  transformer)" / "ou SceneScript (sparse conv)"
- Droite : "Tokenizer" — description : "Decoupe le prompt en tokens du vocabulaire
  Llama / Qwen"

Niveau 3 :
- Gauche (bordure legerement plus epaisse) : "Projecteur (lineaire ou MLP)" —
  description : "Projette les features de points vers la dimension cachee du LLM"
- Droite : "Embeddings de tokens texte" — description : "Une ligne d'embedding par
  token du prompt"

Niveau 4 — une seule rangee large divisee en 3 segments cote a cote (c'est le coeur du
schema, a bien montrer comme une insertion/substitution) :
- Segment gauche (neutre, sans titre) : "Debut de la conversation"
- Segment central (bordure en couleur d'accent, plus epaisse) : titre "Tokens de
  points" — monospace : "n tokens, remplacent <point_pad>"
- Segment droit (neutre, sans titre) : "Consigne + template de code (texte)"

Niveau 5 (une seule boite pleine largeur) : "Modele de langage (LLM)" — monospace :
`Llama-1B ou Qwen-0.5B` — description : "Decodeur transformer, attention causale"

Niveau 6, deux boites horizontales :
- Gauche : "Generation token par token" — description : "lm_head + decodage glouton
  (do_sample = False)"
- Droite (bordure d'accent) : "Code de layout" — monospace (3 lignes) :
  `wall_0=Wall(...)` / `door_0=Door(...)` / `bbox_0=Bbox(...)`

Fleches : verticales simples entre niveaux 1→2→3 (une par colonne) ; du niveau 3 vers
le niveau 4, la fleche venant du Projecteur (colonne gauche) est en couleur d'accent
et pointe vers le segment central ; la fleche venant des Embeddings texte (colonne
droite) pointe vers le segment droit. Une seule fleche du niveau 4 (centre) vers le
niveau 5. Une fleche du niveau 5 vers "Generation token par token", puis une fleche en
couleur d'accent de "Generation token par token" vers "Code de layout".

---

## SCHEMA 3 — Du texte SpatialLM aux entites IFC

Tableau a 3 colonnes ("Sortie texte (.txt)" / "Structures de parsing" / "Entites du
modele IFC") x 4 lignes (mur, porte, fenetre, objet), plus une fleche de mecanisme
supplementaire.

Ligne 1 — mur :
- Col 1 (monospace) : `wall_0=Wall(x1,y1,z1, x2,y2,z2, h, epaisseur)`
- Col 2 : titre "WallData" — description : "Extremites, hauteur, epaisseur"
- Col 3 : titre "IfcWall" — description : "Solide extrude, place et oriente dans
  l'etage"

Ligne 2 — porte (le texte "wall_0" dans la colonne 1 doit etre visuellement mis en
evidence, par exemple en gras ou en couleur d'accent, car c'est la reference cle) :
- Col 1 (monospace) : `door_0=Door(wall_0, cx,cy,cz, largeur, h)`
- Col 2 : titre "DoorData" — monospace : `parent_wall = wall_0`
- Col 3 : titre "IfcOpeningElement + IfcDoor" — description : "Vide perce dans
  IfcWall, porte inseree dans le vide"

Ligne 3 — fenetre (meme structure que la ligne porte) :
- Col 1 (monospace) : `window_0=Window(wall_2, cx,cy,cz, largeur, h)`
- Col 2 : titre "WindowData" — monospace : `parent_wall = wall_2`
- Col 3 : titre "IfcOpeningElement + IfcWindow" — description : "Meme mecanisme de
  vide que pour une porte"

Ligne 4 — objet/mobilier :
- Col 1 (monospace) : `bbox_0=Bbox(classe, cx,cy,cz, rotation, longueur,largeur,hauteur)`
- Col 2 : titre "BboxData" — description : "Classe, pose, dimensions"
- Col 3 : titre "IfcFurnishingElement" — description : "Mobilier / objets detectes"

Fleche de mecanisme (en plus des fleches horizontales classiques colonne 1→2→3) : une
courte fleche verticale en couleur d'accent (pointillee), reliant la boite "DoorData"
(ligne 2, colonne 2) a la boite "WallData" (ligne 1, colonne 2) juste au-dessus,
etiquetee `wall_dir_map / wall_thickness_map`. Cette fleche montre que la porte doit
relire les donnees du mur parent pour connaitre sa direction et son epaisseur.

---

## SCHEMA 4 — Hierarchie spatiale IFC et vides semantiques

Deux zones empilees verticalement, separees par un espace clair (pas de texte "A"/"B",
juste la separation visuelle) :

**Zone haute — hierarchie spatiale**, chaine horizontale de 4 boites reliees par des
fleches simples : "IfcProject" (monospace : `SpatialLM Project`) → "IfcSite"
(monospace : `Origine (0,0,0)`) → "IfcBuilding" → "IfcBuildingStorey" (monospace :
`Ground Floor`).

Depuis la derniere boite, un motif d'eventail vers le bas en couleur d'accent : une
ligne verticale descend puis devient horizontale (etiquetee
`IfcRelContainedInSpatialStructure`), puis se ramifie en 4 fleches verticales vers une
rangee de boites : "IfcSlab", "IfcWall (n)", "IfcDoor / IfcWindow", "Furnishing (n)".

**Zone basse — detail du percage d'une baie**, 3 boites en flux horizontal :
1. "IfcWall" — description : "Mur porteur, geometrie pleine, sans decoupe"
2. (boite centrale, bordure en couleur d'accent, plus large) "IfcOpeningElement" —
   description : "Volume mediateur du vide, non affiche seul"
3. "IfcDoor" — description : "Feuille de porte, inseree dans le vide"

Important : les deux fleches doivent pointer VERS la boite centrale depuis les deux
cotes opposes (pas une fleche continue de gauche a droite) :
- De "IfcWall" vers "IfcOpeningElement" (pointe vers la droite), etiquetee
  `IfcRelVoidsElement`
- De "IfcDoor" vers "IfcOpeningElement" (pointe vers la gauche), etiquetee
  `IfcRelFillsElement`

Les deux fleches convergent donc vers la boite du milieu depuis des directions
opposees : cela montre que l'ouverture est une entite mediatrice reliee separement au
mur et a la porte, et non une simple decoupe booleenne directe.
