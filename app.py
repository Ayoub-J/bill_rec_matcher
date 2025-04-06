import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import time
import shutil
from logic.receipt_analyzer import ReceiptAnalyzer
from logic.receipt_matcher import ReceiptMatcher

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyseur de Factures et Matching Bancaire",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem !important;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.5rem !important;
        color: #0D47A1;
        margin: 1.5rem 0 0.75rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    .error-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialiser les variables de session si elles n'existent pas déjà
if 'receipts_uploaded' not in st.session_state:
    st.session_state.receipts_uploaded = False
if 'bank_statements_uploaded' not in st.session_state:
    st.session_state.bank_statements_uploaded = False
if 'receipts_analyzed' not in st.session_state:
    st.session_state.receipts_analyzed = False
if 'matching_completed' not in st.session_state:
    st.session_state.matching_completed = False
if 'matching_results' not in st.session_state:
    st.session_state.matching_results = None
if 'analysis_log' not in st.session_state:
    st.session_state.analysis_log = []
if 'matching_log' not in st.session_state:
    st.session_state.matching_log = []

# Fonction pour créer les dossiers nécessaires
def ensure_directories():
    directories = ["uploads/receipts", "uploads/bank_statements", "uploads/prompts", "output/receipts", "output/matching"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Créer un fichier .env par défaut s'il n'existe pas
    env_path = Path(".env")
    if not env_path.exists():
        st.exception("Votre Clé API est manquante, veuillez contacter votre administrateur")
        # with open(env_path, "w") as f:
        #     f.write("MISTRAL_API_KEY=votre_clé_api_ici\n")

# Fonction pour ajouter un message au log
def add_to_log(log_type, message):
    if log_type == "analysis":
        st.session_state.analysis_log.append(message)
    elif log_type == "matching":
        st.session_state.matching_log.append(message)

# Fonction pour effacer les fichiers d'un dossier
def clear_directory(directory):
    for item in Path(directory).glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

# Fonction pour traiter les factures
def process_receipts(prompt_content):
    try:
        # Charger la clé API du fichier .env
        api_key = ""
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("MISTRAL_API_KEY="):
                        api_key = line.split("=")[1].strip()
                        break
            
            if not api_key or api_key == "votre_clé_api_ici":
                add_to_log("analysis", "❌ Aucune clé API valide trouvée dans le fichier .env")
                return False, 0
        except Exception as e:
            add_to_log("analysis", f"❌ Erreur lors de la lecture du fichier .env: {str(e)}")
            return False, 0
        
        # Enregistrer le contenu du prompt dans un fichier
        prompt_path = "uploads/prompts/prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_content)
        
        add_to_log("analysis", "Démarrage de l'analyse des factures...")
        
        # Initialiser l'analyseur de reçus
        analyzer = ReceiptAnalyzer(
            prompt_path=prompt_path,
            receipts_dir="uploads/receipts",
            output_dir="output/receipts",
            consolidated_output="all_receipts.json"
        )
        
        # Effectuer l'analyse
        results = analyzer.batch_process()
        
        add_to_log("analysis", f"✅ Analyse terminée : {len(results)} factures traitées")
        
        st.session_state.receipts_analyzed = True
        
        return True, len(results)
    except Exception as e:
        add_to_log("analysis", f"❌ Erreur lors de l'analyse : {str(e)}")
        return False, 0

# Fonction pour exécuter le matching
def run_matching(matching_params):
    try:
        add_to_log("matching", "Démarrage du processus de matching...")
        
        # Initialiser le matcher avec les paramètres fournis
        matcher = ReceiptMatcher(
            receipts_json_path="output/receipts/all_receipts.json",
            bank_statements_dir="uploads/bank_statements",
            output_dir="output/matching",
            days_delta=matching_params["days_delta"],
            amount_tolerance_tier1=matching_params["amount_tolerance_tier1"],
            amount_tolerance_tier2=matching_params["amount_tolerance_tier2"],
            similarity_threshold=matching_params["similarity_threshold"]
        )
        
        # Exécuter le processus complet
        results = matcher.run_complete_process()
        
        if results["success"]:
            add_to_log("matching", f"✅ Matching terminé : {results['matching_count']}/{results['matching_total']} factures matchées")
            add_to_log("matching", f"✅ {len(results['enriched_files'])} relevés bancaires enrichis")
            
            # Charger les résultats du matching pour affichage
            if results["matching_json"]:
                with open(results["matching_json"], "r", encoding="utf-8") as f:
                    matching_data = json.load(f)
                st.session_state.matching_results = matching_data
                st.session_state.matching_completed = True
                return True, results
            else:
                add_to_log("matching", "⚠️ Aucun fichier de résultats généré")
                return False, None
        else:
            add_to_log("matching", f"❌ Erreur : {results.get('error', 'Erreur inconnue')}")
            return False, None
    except Exception as e:
        add_to_log("matching", f"❌ Erreur lors du matching : {str(e)}")
        return False, None

# Créer les dossiers nécessaires au démarrage
ensure_directories()

# Affichage du titre de l'application
st.markdown("<h1 class='main-title'>Analyseur de Factures et Matching Bancaire</h1>", unsafe_allow_html=True)

# Barre latérale
with st.sidebar:
    st.markdown("<h2 class='section-title'>Téléchargement des données</h2>", unsafe_allow_html=True)
    
    # Section téléchargement des factures
    st.markdown("### 📄 Factures")
    uploaded_receipts = st.file_uploader("Télécharger des factures", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    
    if uploaded_receipts:
        clear_btn_receipts = st.button("Effacer les factures")
        if clear_btn_receipts:
            clear_directory("uploads/receipts")
            st.session_state.receipts_uploaded = False
            st.session_state.receipts_analyzed = False
            st.success("Les factures ont été effacées.")
            st.rerun()
        
        for uploaded_file in uploaded_receipts:
            # Sauvegarder le fichier dans le dossier uploads/receipts
            file_path = Path("uploads/receipts") / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.session_state.receipts_uploaded = True
        st.success(f"{len(uploaded_receipts)} factures téléchargées")
    
    # Section téléchargement des relevés bancaires
    st.markdown("### 🏦 Relevés Bancaires")
    uploaded_statements = st.file_uploader("Télécharger des relevés bancaires", accept_multiple_files=True, type=["csv"])
    
    if uploaded_statements:
        clear_btn_statements = st.button("Effacer les relevés")
        if clear_btn_statements:
            clear_directory("uploads/bank_statements")
            st.session_state.bank_statements_uploaded = False
            st.session_state.matching_completed = False
            st.success("Les relevés bancaires ont été effacés.")
            st.rerun()
        
        for uploaded_file in uploaded_statements:
            # Sauvegarder le fichier dans le dossier uploads/bank_statements
            file_path = Path("uploads/bank_statements") / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.session_state.bank_statements_uploaded = True
        st.success(f"{len(uploaded_statements)} relevés bancaires téléchargés")
    
    # Affichage du statut
    st.markdown("### 📊 Statut")
    st.info(f"Factures: {'✅ Téléchargées' if st.session_state.receipts_uploaded else '❌ Non téléchargées'}")
    st.info(f"Relevés bancaires: {'✅ Téléchargés' if st.session_state.bank_statements_uploaded else '❌ Non téléchargés'}")
    st.info(f"Analyse des factures: {'✅ Terminée' if st.session_state.receipts_analyzed else '❌ Non effectuée'}")
    st.info(f"Matching: {'✅ Terminé' if st.session_state.matching_completed else '❌ Non effectué'}")

# Zone principale
tabs = st.tabs(["Analyse des Factures", "Matching", "Résultats", "Logs"])

# Onglet Analyse des Factures
with tabs[0]:
    st.markdown("<h2 class='section-title'>Analyse des Factures</h2>", unsafe_allow_html=True)
    
    if not st.session_state.receipts_uploaded:
        st.markdown("<div class='info-box'>Téléchargez d'abord des factures dans la barre latérale.</div>", unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔑 Configuration API")
            st.info("La clé API Mistral est chargée depuis le fichier .env")
            
            # Charger la clé API depuis .env pour l'utilisation interne
            api_key = ""
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("MISTRAL_API_KEY="):
                            api_key = line.split("=")[1].strip()
                            break
                
                if not api_key or api_key == "votre_clé_api_ici":
                    st.warning("⚠️ Aucune clé API valide trouvée dans le fichier .env. Veuillez éditer ce fichier directement.")
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier .env: {e}")
                st.warning("Assurez-vous que le fichier .env existe avec la variable MISTRAL_API_KEY correctement définie.")
        
        with col2:
            st.markdown("### 📝 Prompt")
            default_prompt = """Analysez cette facture et extrayez les informations suivantes au format JSON:

{
    "merchant": {
        "name": "Nom du commerçant",
        "address": "Adresse si disponible"
    },
    "date": "Date de la transaction (YYYY-MM-DD)",
    "total": "Montant total",
    "items": [
        {
            "description": "Description de l'article",
            "quantity": "Quantité",
            "unit_price": "Prix unitaire",
            "total_price": "Prix total pour cet article"
        }
    ],
    "payment_method": "Méthode de paiement si disponible"
}"""
            
            # Charger le prompt depuis un fichier s'il existe
            prompt_path = Path("uploads/prompts/prompt.txt")
            if prompt_path.exists():
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        default_prompt = f.read()
                except:
                    pass
            
            prompt_content = st.text_area("Prompt pour l'analyse", value=default_prompt, height=300, help="Instructions pour l'extraction des informations des factures")
        
        # Bouton d'analyse
        analyze_button = st.button("🔍 Analyser les Factures")
        
        if analyze_button:
            # Vérifier que le fichier .env contient une clé API valide
            api_key_valid = False
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("MISTRAL_API_KEY="):
                            key = line.split("=")[1].strip()
                            if key and key != "votre_clé_api_ici":
                                api_key_valid = True
                                break
            except:
                pass
            
            if not api_key_valid:
                st.error("Aucune clé API valide trouvée dans le fichier .env. Veuillez éditer ce fichier directement.")
            else:
                # Afficher un spinner pendant l'analyse
                with st.spinner("Analyse des factures en cours..."):
                    success, count = process_receipts(prompt_content)
                
                if success:
                    st.markdown(f"<div class='success-box'>✅ Analyse terminée avec succès ! {count} factures ont été analysées.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='error-box'>❌ Erreur lors de l'analyse des factures. Consultez les logs pour plus de détails.</div>", unsafe_allow_html=True)

# Onglet Matching
with tabs[1]:
    st.markdown("<h2 class='section-title'>Matching des Factures avec les Relevés Bancaires</h2>", unsafe_allow_html=True)
    
    if not st.session_state.receipts_analyzed:
        st.markdown("<div class='info-box'>Veuillez d'abord analyser les factures dans l'onglet précédent.</div>", unsafe_allow_html=True)
    elif not st.session_state.bank_statements_uploaded:
        st.markdown("<div class='info-box'>Téléchargez des relevés bancaires dans la barre latérale.</div>", unsafe_allow_html=True)
    else:
        st.markdown("### ⚙️ Paramètres de Matching")
        
        col1, col2 = st.columns(2)
        
        with col1:
            days_delta = st.slider("Écart de jours maximum", min_value=0, max_value=10, value=3, help="Nombre de jours maximum d'écart entre la date de la facture et celle du relevé bancaire")
            similarity_threshold = st.slider("Seuil de similarité des noms", min_value=50, max_value=100, value=85, help="Seuil minimum pour considérer deux noms de vendeurs comme similaires (en %)")
        
        with col2:
            amount_tolerance_tier1 = st.slider("Tolérance stricte pour les montants", min_value=0.0, max_value=0.10, value=0.05, step=0.01, format="%.2f", help="Différence acceptée pour considérer deux montants comme très proches (en %)")
            amount_tolerance_tier2 = st.slider("Tolérance large pour les montants", min_value=0.0, max_value=0.20, value=0.10, step=0.01, format="%.2f", help="Différence maximale acceptée pour considérer deux montants comme potentiellement liés (en %)")
        
        # Bouton de matching
        matching_button = st.button("🔍 Lancer le Matching")
        
        if matching_button:
            # Paramètres de matching
            matching_params = {
                "days_delta": days_delta,
                "amount_tolerance_tier1": amount_tolerance_tier1,
                "amount_tolerance_tier2": amount_tolerance_tier2,
                "similarity_threshold": similarity_threshold
            }
            
            # Afficher un spinner pendant le matching
            with st.spinner("Matching en cours..."):
                success, results = run_matching(matching_params)
            
            if success:
                matched_count = results["matching_count"]
                total_count = results["matching_total"]
                match_percentage = (matched_count / total_count * 100) if total_count > 0 else 0
                
                st.markdown(f"<div class='success-box'>✅ Matching terminé avec succès ! {matched_count}/{total_count} factures matchées ({match_percentage:.1f}%)</div>", unsafe_allow_html=True)
                
                # Afficher le nombre de relevés enrichis
                enriched_files = len(results.get("enriched_files", []))
                if enriched_files > 0:
                    st.markdown(f"<div class='success-box'>✅ {enriched_files} relevés bancaires ont été enrichis avec les informations des factures.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error-box'>❌ Erreur lors du matching. Consultez les logs pour plus de détails.</div>", unsafe_allow_html=True)

# Onglet Résultats
with tabs[2]:
    st.markdown("<h2 class='section-title'>Résultats du Matching</h2>", unsafe_allow_html=True)
    
    if not st.session_state.matching_completed:
        st.markdown("<div class='info-box'>Aucun résultat de matching disponible. Veuillez d'abord effectuer le matching dans l'onglet précédent.</div>", unsafe_allow_html=True)
    else:
        # Convertir les résultats en DataFrame pour affichage
        matching_data = st.session_state.matching_results
        
        if matching_data:
            # Séparer les résultats en matchés et non matchés
            matched_results = [r for r in matching_data if r.get("matched", False)]
            unmatched_results = [r for r in matching_data if not r.get("matched", False)]
            
            # Créer des DataFrames
            if matched_results:
                matched_df = pd.DataFrame(matched_results)
                
                # Sélectionner et renommer les colonnes pour une meilleure lisibilité
                display_columns = {
                    "receipt_filename": "Nom du fichier",
                    "receipt_total": "Montant facture",
                    "receipt_date": "Date facture",
                    "vendor_receipt": "Vendeur facture",
                    "bank_amount": "Montant banque",
                    "bank_date": "Date banque",
                    "bank_vendor": "Vendeur banque"
                }
                # Ne pas afficher les scores et similarités comme demandé
                
                matched_display_df = matched_df[[col for col in display_columns.keys() if col in matched_df.columns]]
                matched_display_df = matched_display_df.rename(columns=display_columns)
                
                # Onglets pour les résultats matchés et non matchés
                results_tabs = st.tabs(["Matchés", "Non Matchés"])
                
                with results_tabs[0]:
                    st.markdown(f"### ✅ Factures matchées ({len(matched_results)})")
                    
                    # Fonction pour afficher l'image lorsqu'on clique sur le nom du fichier
                    def show_receipt_image(receipt_filename):
                        # Chercher l'image correspondante dans le dossier des factures
                        receipt_dir = Path("uploads/receipts")
                        possible_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
                        
                        # Essayer de trouver le fichier avec différentes extensions
                        image_path = None
                        for ext in possible_extensions:
                            potential_path = receipt_dir / f"{receipt_filename}{ext}"
                            if potential_path.exists():
                                image_path = potential_path
                                break
                        
                        if image_path and image_path.exists():
                            return st.image(str(image_path), caption=f"Facture: {receipt_filename}")
                        else:
                            return st.warning(f"Image de la facture '{receipt_filename}' introuvable.")
                    
                    # Créer une version interactive du DataFrame pour pouvoir cliquer sur les noms de fichiers
                    st.dataframe(matched_display_df, use_container_width=True)
                    
                    # Sélection du fichier pour afficher l'image
                    selected_receipt = st.selectbox(
                        "Sélectionnez une facture pour voir l'image:",
                        options=[result["receipt_filename"] for result in matched_results],
                        key="matched_receipt_selector"
                    )
                    
                    if selected_receipt:
                        show_receipt_image(selected_receipt)
                    
                    # Option pour télécharger les résultats matchés
                    csv = matched_display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger les résultats matchés (CSV)",
                        data=csv,
                        file_name="resultats_matches.csv",
                        mime="text/csv"
                    )
                
                with results_tabs[1]:
                    if unmatched_results:
                        st.markdown(f"### ❌ Factures non matchées ({len(unmatched_results)})")
                        
                        unmatched_df = pd.DataFrame(unmatched_results)
                        unmatched_display_cols = {
                            "receipt_filename": "Nom du fichier",
                            "receipt_total": "Montant",
                            "receipt_date": "Date",
                            "vendor_receipt": "Vendeur",
                            "reason": "Raison"
                        }
                        
                        unmatched_display_df = unmatched_df[[col for col in unmatched_display_cols.keys() if col in unmatched_df.columns]]
                        unmatched_display_df = unmatched_display_df.rename(columns=unmatched_display_cols)
                        
                        st.dataframe(unmatched_display_df, use_container_width=True)
                        
                        # Sélection du fichier non matché pour afficher l'image
                        selected_unmatched = st.selectbox(
                            "Sélectionnez une facture non matchée pour voir l'image:",
                            options=[result["receipt_filename"] for result in unmatched_results],
                            key="unmatched_receipt_selector"
                        )
                        
                        if selected_unmatched:
                            # Réutilisation de la même fonction définie plus haut pour afficher l'image
                            receipt_dir = Path("uploads/receipts")
                            possible_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
                            
                            # Essayer de trouver le fichier avec différentes extensions
                            image_path = None
                            for ext in possible_extensions:
                                potential_path = receipt_dir / f"{selected_unmatched}{ext}"
                                if potential_path.exists():
                                    image_path = potential_path
                                    break
                            
                            if image_path and image_path.exists():
                                st.image(str(image_path), caption=f"Facture: {selected_unmatched}")
                            else:
                                st.warning(f"Image de la facture '{selected_unmatched}' introuvable.")
                        
                        # Option pour télécharger les résultats non matchés
                        csv_unmatched = unmatched_display_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger les factures non matchées (CSV)",
                            data=csv_unmatched,
                            file_name="factures_non_matchees.csv",
                            mime="text/csv",
                            key="download_unmatched_csv"
                        )
                    else:
                        st.markdown("<div class='success-box'>Toutes les factures ont été matchées !</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warning-box'>Aucune facture matchée trouvée dans les résultats.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warning-box'>Les résultats du matching sont vides ou dans un format inattendu.</div>", unsafe_allow_html=True)
        
        # Section pour télécharger les fichiers de sortie
        st.markdown("### 📂 Fichiers de sortie")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Fichiers de matching")
            matching_files = list(Path("output/matching").glob("*.json")) + list(Path("output/matching").glob("*.csv"))
            
            if matching_files:
                for i, file_path in enumerate(matching_files):
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"📥 {file_path.name}",
                            data=f,
                            file_name=file_path.name,
                            mime="application/octet-stream",
                            key=f"matching_file_{i}"  # Clé unique pour chaque bouton
                        )
            else:
                st.info("Aucun fichier de matching disponible")
        
        with col2:
            st.markdown("#### Relevés bancaires enrichis")
            enriched_files = list(Path("output/matching").glob("*_enriched*.csv"))
            
            if enriched_files:
                for i, file_path in enumerate(enriched_files):
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"📥 {file_path.name}",
                            data=f,
                            file_name=file_path.name,
                            mime="application/octet-stream",
                            key=f"enriched_file_{i}"  # Clé unique pour chaque bouton
                        )
            else:
                st.info("Aucun relevé bancaire enrichi disponible")

# Onglet Logs
with tabs[3]:
    st.markdown("<h2 class='section-title'>Logs</h2>", unsafe_allow_html=True)
    
    # Onglets pour les différents types de logs
    log_tabs = st.tabs(["Analyse des Factures", "Matching"])
    
    with log_tabs[0]:
        if st.session_state.analysis_log:
            for log_entry in st.session_state.analysis_log:
                if "✅" in log_entry:
                    st.markdown(f"<div class='success-box'>{log_entry}</div>", unsafe_allow_html=True)
                elif "⚠️" in log_entry:
                    st.markdown(f"<div class='warning-box'>{log_entry}</div>", unsafe_allow_html=True)
                elif "❌" in log_entry:
                    st.markdown(f"<div class='error-box'>{log_entry}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='info-box'>{log_entry}</div>", unsafe_allow_html=True)
        else:
            st.info("Aucun log d'analyse disponible")
    
    with log_tabs[1]:
        if st.session_state.matching_log:
            for log_entry in st.session_state.matching_log:
                if "✅" in log_entry:
                    st.markdown(f"<div class='success-box'>{log_entry}</div>", unsafe_allow_html=True)
                elif "⚠️" in log_entry:
                    st.markdown(f"<div class='warning-box'>{log_entry}</div>", unsafe_allow_html=True)
                elif "❌" in log_entry:
                    st.markdown(f"<div class='error-box'>{log_entry}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='info-box'>{log_entry}</div>", unsafe_allow_html=True)
        else:
            st.info("Aucun log de matching disponible")

# Pied de page
st.markdown("---")
st.markdown("📊 **Analyseur de Factures et Matching Bancaire** | Développé avec Streamlit")
