# Contrat d'équipe

- Mots : liste unique dans config.py (15 mots ASL)
- Longueur de séquence : T = 30 frames
- 144 valeurs par frame : main gauche 21 + main droite 21 + 6 points de pose, chacun en x/y/z
- Tableaux : X [N, 30, 144], y [N], groups [N] (ID participant)
- Split TOUJOURS par participant, jamais aléatoire
- features.py est le seul endroit du prétraitement (entraînement ET app)
- Datasets jamais commités (partage via Kaggle/Drive)

# Git
- Jamais de push direct sur main
- Une branche par tâche : feature/<role>-<sujet> (ex : feature/data-preprocessing)
- Pull request obligatoire, 1 relecture minimum
- Toute modification de features.py est relue par le PM
