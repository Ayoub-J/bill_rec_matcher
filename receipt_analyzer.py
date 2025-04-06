import os
import base64
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Modifier l'importation de Mistral pour être compatible avec différentes versions
try:
    from mistralai.client import MistralClient as Mistral
except ImportError:
    try:
        from mistralai.client.mistral_client import MistralClient as Mistral
    except ImportError:
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("Impossible d'importer la classe Mistral. Vérifiez votre installation.")

# Import conditionnel de streamlit pour les déploiements Cloud
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class ReceiptAnalyzer:
    """
    Classe pour analyser les reçus/factures en utilisant le modèle Pixtral via l'API Mistral.
    Cette classe combine les fonctionnalités d'extraction et de traitement par lot.
    """

    DEFAULT_MODEL = "pixtral-12b-2409"
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    DEFAULT_RETRY_DELAY = 10  # secondes
    DEFAULT_REQUEST_PAUSE = 2  # secondes
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BATCH_SIZE = 20
    DEFAULT_DELAY_BETWEEN_BATCHES = 10  # secondes

    def __init__(
        self,
        prompt_path: str = "prompt.txt",
        model: str = DEFAULT_MODEL,
        receipts_dir: str = "receipts",
        output_dir: str = "output", 
        env_path: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        delay_between_batches: int = DEFAULT_DELAY_BETWEEN_BATCHES,
        consolidated_output: str = "all_results.json",
        api_key: Optional[str] = None
    ):
        """
        Initialisation de l'analyseur de reçus.
        
        Args:
            prompt_path: Chemin vers le fichier prompt (par défaut: "prompt.txt")
            model: Modèle Mistral à utiliser
            receipts_dir: Dossier des reçus (crée un dossier 'receipts' par défaut)
            output_dir: Dossier de sortie pour les résultats JSON (crée un dossier 'output' par défaut)
            env_path: Chemin du fichier .env (par défaut, recherche dans le dossier courant)
            batch_size: Nombre d'images à traiter par lot
            delay_between_batches: Délai en secondes entre chaque lot
            consolidated_output: Nom du fichier pour la sortie consolidée de tous les résultats
            api_key: Clé API Mistral fournie directement (prioritaire sur les autres méthodes)
        """
        # Configuration du logging
        self._setup_logging()
        
        # Vérification de la clé API (plusieurs sources possibles)
        self.api_key = api_key or self._get_api_key(env_path)
        if not self.api_key:
            raise ValueError("La clé API Mistral est manquante. Fournissez-la via api_key, le secret Streamlit ou un fichier .env")
        
        # Initialisation du client Mistral
        self.client = Mistral(api_key=self.api_key)
        self.model = model
        
        # Configuration des chemins
        self.prompt_path = prompt_path
        
        # Configuration des paramètres de traitement par lot
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        
        # Configuration de la sortie consolidée
        self.consolidated_output = consolidated_output
        
        # Création des dossiers nécessaires
        self.receipts_dir = self._ensure_directory(receipts_dir)
        self.output_dir = self._ensure_directory(output_dir)
        
        # Dictionnaire pour stocker tous les résultats
        self.all_results = []
        
        # Charger les résultats existants s'ils existent
        self._load_existing_consolidated_results()
        
        self.logger.info(f"ReceiptAnalyzer initialisé avec: model={model}, "
                         f"receipts_dir={self.receipts_dir}, output_dir={self.output_dir}")

    def _get_api_key(self, env_path: Optional[str] = None) -> str:
        """
        Obtient la clé API Mistral de diverses sources.
        
        Args:
            env_path: Chemin vers le fichier .env
            
        Returns:
            La clé API Mistral ou une chaîne vide si non trouvée
        """
        # 1. Essayer d'abord les secrets Streamlit (pour déploiement cloud)
        if STREAMLIT_AVAILABLE:
            try:
                return st.secrets["MISTRAL_API_KEY"]
            except:
                self.logger.warning("Pas de secret Streamlit 'MISTRAL_API_KEY' trouvé")
        
        # 2. Essayer les variables d'environnement
        api_key = os.environ.get("MISTRAL_API_KEY")
        if api_key:
            return api_key
        
        # 3. Charger depuis un fichier .env
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key:
            return api_key
        
        # 4. Essayer de lire directement le fichier .env si présent
        env_path = Path(".env")
        if env_path.exists():
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("MISTRAL_API_KEY="):
                            return line.split("=")[1].strip()
            except Exception as e:
                self.logger.error(f"Erreur lors de la lecture du fichier .env: {e}")
        
        return ""

    def _setup_logging(self) -> None:
        """Configure le système de journalisation."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('ReceiptAnalyzer')

    def _ensure_directory(self, directory: str) -> str:
        """
        S'assure que le répertoire existe, le crée si nécessaire.
        
        Args:
            directory: Chemin du répertoire à vérifier/créer
            
        Returns:
            Le chemin absolu du répertoire
        """
        path = Path(directory).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    def _load_existing_consolidated_results(self) -> None:
        """
        Charge les résultats existants du fichier consolidé s'il existe.
        """
        consolidated_path = Path(self.output_dir) / self.consolidated_output
        
        if consolidated_path.exists():
            try:
                with open(consolidated_path, "r", encoding="utf-8") as f:
                    self.all_results = json.load(f)
                    
                self.logger.info(f"Chargement de {len(self.all_results)} résultats existants depuis {consolidated_path}")
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement des résultats existants: {e}")
                self.all_results = []

    def _load_prompt(self, path: str) -> str:
        """
        Charge le prompt depuis un fichier.
        
        Args:
            path: Chemin vers le fichier de prompt
            
        Returns:
            Le contenu du prompt
        """
        try:
            prompt_path = Path(path)
            if not prompt_path.exists():
                self.logger.warning(f"Fichier prompt non trouvé: {path}")
                return "Veuillez analyser cette facture et extraire les détails importants au format JSON."
            
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du prompt: {e}")
            return "Veuillez analyser cette facture et extraire les détails importants au format JSON."

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode une image en base64.
        
        Args:
            image_path: Chemin de l'image à encoder
            
        Returns:
            L'image encodée en base64
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_receipt(self, image_path: str, filename: str, max_retries: int = DEFAULT_MAX_RETRIES) -> Dict[str, Any]:
        """
        Analyse un reçu avec gestion des tentatives en cas d'erreur.
        
        Args:
            image_path: Chemin de l'image à analyser
            filename: Nom du fichier pour le rapport
            max_retries: Nombre maximum de tentatives en cas d'erreur 429
            
        Returns:
            Résultats d'analyse sous forme de dictionnaire
        """
        for attempt in range(max_retries):
            try:
                result = self._perform_analysis(image_path, filename)
                
                # Ajouter le résultat au dictionnaire global et sauvegarder
                self._add_to_consolidated_results(result)
                
                return result
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    self.logger.warning(f"⏳ Limite atteinte (429). Attente {self.DEFAULT_RETRY_DELAY} secondes "
                                       f"avant nouvelle tentative... ({attempt + 1}/{max_retries})")
                    time.sleep(self.DEFAULT_RETRY_DELAY)
                else:
                    self.logger.error(f"Erreur d'analyse pour {filename}: {e}")
                    raise

    def _perform_analysis(self, image_path: str, filename: str) -> Dict[str, Any]:
        """
        Exécute l'analyse d'un reçu.
        
        Args:
            image_path: Chemin de l'image à analyser
            filename: Nom du fichier pour la référence
            
        Returns:
            Résultats d'analyse sous forme de dictionnaire
        """
        # Encoder l'image en base64
        base64_image = self._encode_image_to_base64(image_path)
        
        # Charger le prompt depuis le fichier
        prompt = self._load_prompt(self.prompt_path)
        
        # Préparer le prompt avec le nom du fichier
        prompt_complet = f"{prompt}\nThe filename is: {filename}"
        
        # Préparer la requête
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_complet},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ]
        
        # Appeler l'API
        try:
            # Version 1: Nouvelle version de l'API
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        except AttributeError:
            try:
                # Version 2: Ancienne version de l'API
                response = self.client.chat_completions(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
            except:
                # Version 3: Encore plus ancienne version possible
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
        
        # Ajouter des métadonnées
        result["_metadata"] = {
            "filename": filename,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model
        }
        
        return result
    
    def _add_to_consolidated_results(self, result: Dict[str, Any]) -> None:
        """
        Ajoute un résultat au fichier consolidé et sauvegarde.
        
        Args:
            result: Résultat à ajouter
        """
        # Vérifier si le résultat existe déjà (en se basant sur le nom de fichier)
        filename = result.get("_metadata", {}).get("filename", "")
        
        # Chercher et remplacer si le résultat pour ce fichier existe déjà
        found = False
        for i, existing_result in enumerate(self.all_results):
            if existing_result.get("_metadata", {}).get("filename", "") == filename:
                self.all_results[i] = result
                found = True
                break
        
        # Sinon, l'ajouter
        if not found:
            self.all_results.append(result)
        
        # Sauvegarder le fichier consolidé
        self._save_consolidated_results()

    def _save_consolidated_results(self) -> str:
        """
        Sauvegarde tous les résultats dans le fichier consolidé.
        
        Returns:
            Chemin du fichier de sortie
        """
        output_path = Path(self.output_dir) / self.consolidated_output
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
            
        return str(output_path)

    def batch_process(
        self, 
        limit: Optional[int] = None,
        output_filename: str = "resultats.json"
    ) -> List[Dict[str, Any]]:
        """
        Traite un lot d'images de reçus.
        
        Args:
            limit: Limite le nombre d'images à traiter (None pour traiter toutes les images)
            output_filename: Nom du fichier de sortie pour les résultats
            
        Returns:
            Liste des résultats d'analyse
        """
        # Trouver tous les fichiers avec les extensions supportées
        image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(list(Path(self.receipts_dir).glob(f"*{ext}")))
            image_files.extend(list(Path(self.receipts_dir).glob(f"*{ext.upper()}")))
        
        # Trier les fichiers pour un traitement cohérent
        image_files.sort()
        
        # Appliquer la limite si spécifiée
        if limit and limit > 0:
            image_files = image_files[:limit]
        
        total_files = len(image_files)
        self.logger.info(f"🔍 Traitement de {total_files} images dans le dossier `{self.receipts_dir}`...")
        
        results = []
        
        # Traitement par lots
        for batch_idx in range(0, len(image_files), self.batch_size):
            batch = image_files[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"🔄 Traitement du batch {batch_num}/{total_batches}...")
            
            batch_results = []
            for idx, image_path in enumerate(batch, start=batch_idx + 1):
                filename = image_path.name
                self.logger.info(f"📄 {idx}/{total_files}. Traitement de : {filename}...")
                
                try:
                    # Analyser le reçu (le résultat est déjà ajouté au fichier consolidé)
                    result = self.analyze_receipt(str(image_path), filename)
                    results.append(result)
                    batch_results.append(result)
                    
                    # Pause pour éviter de surcharger l'API
                    time.sleep(self.DEFAULT_REQUEST_PAUSE)
                    
                except Exception as e:
                    self.logger.error(f"⚠️ Erreur sur {filename} : {e}")
            
            # Sauvegarder les résultats intermédiaires après chaque lot
            self._save_batch_results(batch_results, batch_num, total_batches)
            
            # Pause entre les lots
            if batch_num < total_batches:
                self.logger.info(f"⏳ Pause de {self.delay_between_batches} secondes entre les batches...")
                time.sleep(self.delay_between_batches)
        
        # Sauvegarde dans le fichier de sortie spécifié (en plus du fichier consolidé)
        output_path = self._save_results(results, output_filename)
        
        # Logger des informations sur les deux fichiers
        consolidated_path = Path(self.output_dir) / self.consolidated_output
        self.logger.info(f"✅ Traitement terminé.")
        self.logger.info(f"📄 Résultats de cette session dans: `{output_path}`")
        self.logger.info(f"📄 Tous les résultats consolidés dans: `{consolidated_path}`")
        
        return results
    
    def _save_batch_results(self, batch_results: List[Dict[str, Any]], batch_num: int, total_batches: int) -> str:
        """
        Sauvegarde les résultats d'un lot.
        
        Args:
            batch_results: Liste des résultats du lot à sauvegarder
            batch_num: Numéro du lot actuel
            total_batches: Nombre total de lots
            
        Returns:
            Chemin du fichier de sortie
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"resultats_batch_{batch_num}_de_{total_batches}_{timestamp}.json"
        output_path = Path(self.output_dir) / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"💾 Sauvegarde du batch effectuée: {output_path}")
        return str(output_path)

    def _save_results(self, results: List[Dict[str, Any]], filename: str) -> str:
        """
        Sauvegarde les résultats dans un fichier JSON.
        
        Args:
            results: Liste des résultats à sauvegarder
            filename: Nom du fichier de sortie
            
        Returns:
            Chemin du fichier de sortie
        """
        output_path = Path(self.output_dir) / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return str(output_path)
    
    def process_single_image(self, image_path: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Traite une seule image et sauvegarde le résultat.
        
        Args:
            image_path: Chemin de l'image à analyser
            output_filename: Nom du fichier de sortie (généré automatiquement si None)
            
        Returns:
            Résultat de l'analyse
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"L'image {image_path} n'existe pas")
        
        filename = image_path.name
        self.logger.info(f"Traitement de l'image unique: {filename}")
        
        # Analyser (le résultat est automatiquement ajouté au fichier consolidé)
        result = self.analyze_receipt(str(image_path), filename)
        
        # Déterminer le nom du fichier de sortie individuel
        if not output_filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"analyse_{image_path.stem}_{timestamp}.json"
        
        # Sauvegarder le résultat individuel
        output_path = Path(self.output_dir) / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Chemin du fichier consolidé
        consolidated_path = Path(self.output_dir) / self.consolidated_output
        
        self.logger.info(f"Analyse terminée.")
        self.logger.info(f"📄 Résultat individuel enregistré dans: `{output_path}`")
        self.logger.info(f"📄 Tous les résultats consolidés dans: `{consolidated_path}`")
        
        return result


if __name__ == "__main__":
    # Exemple d'utilisation de base
    analyzer = ReceiptAnalyzer()
    analyzer.batch_process()
