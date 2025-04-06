import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import time
import shutil
from receipt_analyzer import ReceiptAnalyzer
from receipt_matcher import ReceiptMatcher

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyseur de Factures et Matching Bancaire",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Vérification de la clé API
def get_api_key():
    """Récupère la clé API de diverses sources."""
    # Essayer d'abord les secrets Streamlit (pour déploiement cloud)
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except Exception as e:
        st.warning(f"Impossible de récupérer le secret MISTRAL_API_KEY: {e}")
    
    # Essayer les variables d'environnement
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key:
        return api_key
    
    # Essayer le fichier .env s'il existe
    env_path = Path(".env")
    if env_path.exists():
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("MISTRAL_API_KEY="):
                        return line.split("=")[1].strip()
        except Exception as e:
            st.warning(f"Erreur lors de la lecture du fichier .env: {e}")
    
    return ""

# Vérification au démarrage de l'application
api_key = get_api_key()
if not api_key:
    st.error("⚠️ Aucune clé API Mistral n'a été trouvée. Veuillez configurer un secret 'MISTRAL_API_KEY' dans les paramètres de Streamlit Cloud ou créer un fichier .env.")

# RESTE DU CODE INCHANGÉ
# Ajoutez ci-dessous le reste de votre code app.py, avec la seule modification 
# de passer l'api_key à l'initialisation de ReceiptAnalyzer

# Par exemple, quand vous initialisez l'analyseur dans process_receipts(), utilisez:
# analyzer = ReceiptAnalyzer(
#     prompt_path=prompt_path,
#     receipts_dir="uploads/receipts",
#     output_dir="output/receipts",
#     consolidated_output="all_receipts.json",
#     api_key=get_api_key()  # Utiliser la fonction pour obtenir la clé API
# )
