# bank_rec_matcher

Une application complète pour extraire des informations des factures à l'aide de l'IA et les associer automatiquement aux transactions bancaires.

## Fonctionnalités
- Analyse de factures par IA : Extraction automatique des informations des factures (commerçant, date, montant, etc.) à l'aide du modèle Pixtral via l'API Mistral.
- Matching intelligent : Association automatique des factures avec les transactions bancaires en utilisant plusieurs critères (montant, date, nom du vendeur).
- Enrichissement des relevés bancaires : Ajout des références aux factures dans vos relevés bancaires CSV.
- Interface utilisateur intuitive : Application web conviviale développée avec Streamlit.

  ## Prérequis
- Python 3.7 ou supérieur.
- Compte Mistral AI avec clé API (pour l'analyse des factures).
- Les bibliothèques Python listées dans requirements.txt

  ## Installations
  1. Cloner ce dépot ou télécgargez les fichiers
  2. Installez les dépendences :
 
```bash
pip install -r requirements.txt
```
3. Configurez votre clé API Mistral 
- Créez un fichier .env à la racine du projet
- Ajoutez votre clé API :
```
MISTRAL_API_KEY=votre_clé_api_ici
```

## Structures du projets

```
├── app.py                      # Application Streamlit principale
├── receipt_analyzer.py         # Classe pour l'analyse des factures 
├── receipt_matcher.py          # Classe pour le matching avec les relevés
├── requirements.txt            # Dépendances Python
├── .env                        # Fichier pour la clé API (à créer)
├── uploads/                    # Dossier pour les fichiers téléchargés
│   ├── receipts/               # Factures téléchargées
│   ├── bank_statements/        # Relevés bancaires téléchargés
│   └── prompts/                # Fichiers de prompt
└── output/                     # Dossier pour les résultats
    ├── receipts/               # Résultats d'analyse des factures
    └── matching/               # Résultats du matching
```

##Utilisation
- lancement de l'application :
```bash
streamlit run app.py
```

- Workflow d'utilisation :
1. Préparation des fichiers
* Préparez vos factures en format image (JPG, JPEG, PNG)
* Assurez-vous que vos relevés bancaires sont au format CSV avec au minimum les colonnes : date, amount, vendor

  2. Téléchargement des fichiers
*Téléchargez vos factures via la barre latérale
*Téléchargez vos relevés bancaires CSV

  3. Analyse des factures
*Personnalisez le prompt d'extraction si nécessaire
*Lancez l'analyse des factures

  4. Matching avec relevé bancaires
*Ajustez les paramètres de matching selon vos besoins
*Lancez le processus de matching

  5. Visualisation et exportation des résultats
*Consultez les factures matchées et non matchées
*Téléchargez les résultats au format CSV
*Obtenez vos relevés bancaires enrichis

## Paramètres de matching personnalisables 
* Écart de jours maximum : Nombre de jours d'écart toléré entre la date de la facture et celle de la transaction bancaire.
* Seuil de similarité des noms : Pourcentage minimal de similarité entre les noms des vendeurs pour considérer qu'ils sont identiques.
* Tolérance stricte pour les montants : Différence acceptable pour considérer deux montants comme très proches.
* Tolérance large pour les montants : Différence maximale acceptable pour considérer deux montants comme potentiellement liés.

## Limitation actuel :
* L'analyse des factures dépend de la qualité des images et de la précision du prompt.
* Les performances peuvent varier selon le type et la complexité des factures.
* L'application est optimisée pour les formats de relevés bancaires standards, des ajustements peuvent être nécessaires pour certains formats spécifiques.

# onseils pour de meilleurs résultats :
*Utilisez des images de factures claires et bien cadrées.
*Personnalisez le prompt d'extraction en fonction du type de vos factures.
*Ajustez les paramètres de matching selon vos besoins spécifiques.
*Pour les relevés bancaires, assurez-vous que les dates sont dans un format standard (YYYY-MM-DD).
