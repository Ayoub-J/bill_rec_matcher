import json
import pandas as pd
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from rapidfuzz import fuzz
import logging


class ReceiptMatcher:
    """
    Classe pour faire correspondre des reçus analysés avec des relevés bancaires
    et enrichir les relevés avec les informations des reçus correspondants.
    """

    def __init__(
        self,
        receipts_json_path: str = "output/resultats.json",
        bank_statements_dir: str = "bank_statements",
        output_dir: str = "output",
        matching_output_prefix: str = "matchings",
        enriched_output_prefix: str = "enriched",
        days_delta: int = 3,
        amount_tolerance_tier1: float = 0.05,
        amount_tolerance_tier2: float = 0.10,
        similarity_threshold: int = 85
    ):
        """
        Initialise le matcher de reçus et relevés bancaires.
        
        Args:
            receipts_json_path: Chemin vers le fichier JSON contenant les reçus analysés
            bank_statements_dir: Dossier contenant les relevés bancaires CSV
            output_dir: Dossier de sortie pour les résultats
            matching_output_prefix: Préfixe pour les fichiers de résultats du matching
            enriched_output_prefix: Préfixe pour les fichiers de relevés enrichis
            days_delta: Nombre de jours maximum d'écart accepté entre la date de facture et du relevé
            amount_tolerance_tier1: Tolérance de niveau 1 pour les différences de montant (% ou valeur absolue)
            amount_tolerance_tier2: Tolérance de niveau 2 pour les différences de montant (% ou valeur absolue)
            similarity_threshold: Seuil minimum pour la similarité des noms de vendeurs (0-100)
        """
        # Configuration du logging
        self._setup_logging()
        
        # Chemins et paramètres
        self.receipts_json_path = Path(receipts_json_path)
        self.bank_statements_dir = Path(bank_statements_dir)
        self.output_dir = Path(output_dir)
        
        # Préfixes pour les fichiers de sortie
        self.matching_output_prefix = matching_output_prefix
        self.enriched_output_prefix = enriched_output_prefix
        
        # Paramètres de matching
        self.days_delta = days_delta
        self.amount_tolerance_tier1 = amount_tolerance_tier1
        self.amount_tolerance_tier2 = amount_tolerance_tier2
        self.similarity_threshold = similarity_threshold
        
        # Données
        self.receipts = []
        self.bank_statements = pd.DataFrame()
        self.matchings = []
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ReceiptMatcher initialisé avec les paramètres suivants:")
        self.logger.info(f" - Fichier de reçus: {self.receipts_json_path}")
        self.logger.info(f" - Dossier de relevés: {self.bank_statements_dir}")
        self.logger.info(f" - Dossier de sortie: {self.output_dir}")
        self.logger.info(f" - Delta jours: {self.days_delta}")
        self.logger.info(f" - Tolérances montant: {self.amount_tolerance_tier1}/{self.amount_tolerance_tier2}")
        self.logger.info(f" - Seuil similarité: {self.similarity_threshold}%")

    def _setup_logging(self) -> None:
        """Configure le système de journalisation."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('ReceiptMatcher')

    def load_receipts(self, json_path: Optional[str] = None) -> None:
        """
        Charge les reçus depuis le fichier JSON.
        
        Args:
            json_path: Chemin facultatif vers le fichier JSON (utilise celui défini dans le constructeur par défaut)
        """
        path = Path(json_path) if json_path else self.receipts_json_path
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.receipts = json.load(f)
            
            self.logger.info(f"✅ {len(self.receipts)} reçus chargés depuis {path}")
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement des reçus depuis {path}: {e}")
            self.receipts = []

    def load_bank_statements(self, directory: Optional[str] = None) -> None:
        """
        Charge tous les relevés bancaires CSV du dossier spécifié.
        
        Args:
            directory: Dossier facultatif contenant les relevés (utilise celui défini dans le constructeur par défaut)
        """
        dir_path = Path(directory) if directory else self.bank_statements_dir
        
        if not dir_path.exists():
            self.logger.error(f"❌ Le dossier de relevés {dir_path} n'existe pas")
            return
        
        csv_paths = list(dir_path.glob("*.csv"))
        if not csv_paths:
            self.logger.warning(f"⚠️ Aucun fichier CSV trouvé dans {dir_path}")
            return
        
        dataframes = []
        for csv_path in csv_paths:
            try:
                # Essayer d'abord avec la virgule comme séparateur
                df = pd.read_csv(csv_path)
                # Si cela donne une seule colonne, essayer avec point-virgule
                if df.shape[1] == 1:
                    df = pd.read_csv(csv_path, sep=";")
                
                dataframes.append(df)
                self.logger.info(f"✅ Relevé chargé: {csv_path.name} ({len(df)} lignes)")
            except Exception as e:
                self.logger.error(f"❌ Erreur lors de la lecture de {csv_path}: {e}")
        
        if dataframes:
            self.bank_statements = pd.concat(dataframes, ignore_index=True)
            
            # Conversion des dates et nettoyage des noms de vendeurs
            self.bank_statements["date"] = pd.to_datetime(self.bank_statements["date"], errors="coerce")
            
            # Ajouter une colonne avec les noms de vendeurs nettoyés et en majuscules
            if "vendor" in self.bank_statements.columns:
                self.bank_statements["vendor_clean"] = self.bank_statements["vendor"].astype(str).str.strip().str.upper()
            
            self.logger.info(f"✅ Total: {len(self.bank_statements)} lignes bancaires chargées depuis {len(dataframes)} fichiers")
        else:
            self.bank_statements = pd.DataFrame()
            self.logger.warning("⚠️ Aucun relevé bancaire chargé")

    def perform_matching(self) -> List[Dict[str, Any]]:
        """
        Effectue le matching entre les reçus et les relevés bancaires.
        
        Returns:
            Liste des matchings trouvés
        """
        if not self.receipts:
            self.logger.error("❌ Aucun reçu chargé. Impossible de faire le matching.")
            return []
        
        if self.bank_statements.empty:
            self.logger.error("❌ Aucun relevé bancaire chargé. Impossible de faire le matching.")
            return []
        
        self.matchings = []
        
        self.logger.info(f"🔍 Début du matching pour {len(self.receipts)} reçus...")
        
        for receipt in self.receipts:
            # Extraire les informations du reçu
            filename = self._extract_receipt_filename(receipt)
            total = self._extract_receipt_total(receipt)
            date_str = self._extract_receipt_date(receipt)
            vendor_name = self._extract_receipt_vendor_name(receipt)
            
            # Vérifier que les informations essentielles sont présentes
            if not all([filename, total is not None, date_str]):
                self.logger.warning(f"⚠️ Reçu incomplet ignoré: {filename or 'sans nom'}")
                self._add_unmatched_receipt(receipt, "Informations incomplètes")
                continue
            
            # Convertir la date en objet datetime
            try:
                date_receipt = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                self.logger.warning(f"⚠️ Format de date invalide ({date_str}) pour le reçu: {filename}")
                self._add_unmatched_receipt(receipt, "Format de date invalide")
                continue
            
            # Rechercher le meilleur match
            best_match = self._find_best_match(filename, total, date_receipt, vendor_name)
            
            if best_match:
                self.matchings.append(best_match)
            else:
                self._add_unmatched_receipt(receipt, "Aucun match satisfaisant trouvé")
        
        self.logger.info(f"✅ Matching terminé: {sum(1 for m in self.matchings if m['matched'])}/{len(self.matchings)} reçus matchés")
        
        return self.matchings

    def _extract_receipt_filename(self, receipt: Dict[str, Any]) -> str:
        """
        Extrait le nom du fichier d'un reçu.
        
        Args:
            receipt: Dictionnaire contenant les données du reçu
            
        Returns:
            Nom du fichier du reçu
        """
        # Essayer d'abord dans les metadata
        metadata = receipt.get("_metadata", {})
        if metadata and "filename" in metadata:
            # Retirer l'extension si présente
            filename = metadata["filename"]
            return Path(filename).stem
        
        # Sinon chercher un champ 'filename'
        if "filename" in receipt:
            return Path(receipt["filename"]).stem
        
        return ""

    def _extract_receipt_total(self, receipt: Dict[str, Any]) -> Optional[float]:
        """
        Extrait le montant total d'un reçu.
        
        Args:
            receipt: Dictionnaire contenant les données du reçu
            
        Returns:
            Montant total du reçu ou None si non trouvé
        """
        # Plusieurs structures possibles selon le prompt utilisé
        # Essayer différents chemins pour trouver le total
        try:
            # Chemin direct
            if "total" in receipt:
                return float(receipt["total"])
            
            # Dans un sous-dictionnaire 'payment'
            if "payment" in receipt and "total" in receipt["payment"]:
                return float(receipt["payment"]["total"])
            
            # Dans un sous-dictionnaire 'transaction'
            if "transaction" in receipt and "total" in receipt["transaction"]:
                return float(receipt["transaction"]["total"])
            
            # Dans le montant total
            if "total_amount" in receipt:
                return float(receipt["total_amount"])
            
            return None
        except (ValueError, TypeError):
            return None

    def _extract_receipt_date(self, receipt: Dict[str, Any]) -> str:
        """
        Extrait la date d'un reçu.
        
        Args:
            receipt: Dictionnaire contenant les données du reçu
            
        Returns:
            Date du reçu au format YYYY-MM-DD ou chaîne vide si non trouvée
        """
        # Plusieurs structures possibles selon le prompt utilisé
        # Essayer différents chemins pour trouver la date
        
        # Chemin direct
        if "date" in receipt:
            return receipt["date"]
        
        # Dans un sous-dictionnaire 'transaction'
        if "transaction" in receipt and "date" in receipt["transaction"]:
            return receipt["transaction"]["date"]
        
        # Sous le nom 'purchase_date'
        if "purchase_date" in receipt:
            return receipt["purchase_date"]
        
        return ""

    def _extract_receipt_vendor_name(self, receipt: Dict[str, Any]) -> str:
        """
        Extrait le nom du vendeur d'un reçu.
        
        Args:
            receipt: Dictionnaire contenant les données du reçu
            
        Returns:
            Nom du vendeur en majuscules ou chaîne vide si non trouvé
        """
        # Plusieurs structures possibles selon le prompt utilisé
        vendor_name = ""
        
        # Chemin direct
        if "merchant" in receipt:
            if isinstance(receipt["merchant"], dict) and "name" in receipt["merchant"]:
                vendor_name = receipt["merchant"]["name"]
            else:
                vendor_name = str(receipt["merchant"])
        
        # Sous le nom 'vendor'
        elif "vendor" in receipt:
            if isinstance(receipt["vendor"], dict) and "name" in receipt["vendor"]:
                vendor_name = receipt["vendor"]["name"]
            else:
                vendor_name = str(receipt["vendor"])
        
        # Sous le nom 'store'
        elif "store" in receipt:
            vendor_name = str(receipt["store"])
        
        # Sous le nom 'merchant_name'
        elif "merchant_name" in receipt:
            vendor_name = str(receipt["merchant_name"])
        
        return vendor_name.strip().upper()

    def _find_best_match(
        self, 
        filename: str,
        total: float, 
        date_receipt: datetime, 
        vendor_receipt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Trouve le meilleur match pour un reçu dans les relevés bancaires.
        
        Args:
            filename: Nom du fichier du reçu
            total: Montant total du reçu
            date_receipt: Date du reçu
            vendor_receipt: Nom du vendeur du reçu
            
        Returns:
            Dictionnaire contenant les détails du meilleur match ou None si aucun match satisfaisant
        """
        best_score = 0
        best_match = None
        
        for _, row in self.bank_statements.iterrows():
            # Extraire et convertir le montant
            try:
                bank_amount = float(str(row["amount"]).replace(",", "."))
            except (ValueError, TypeError):
                continue  # Ignorer les lignes avec des montants invalides
            
            # Vérifier la date
            bank_date = row["date"]
            if pd.isnull(bank_date):
                continue  # Ignorer les lignes sans date
            
            # Récupérer le vendeur
            vendor_bank = str(row.get("vendor_clean", row.get("vendor", ""))).upper()
            
            # Calculer l'écart de montant
            amount_diff = abs(bank_amount - total)
            
            # Première validation: le montant doit être suffisamment proche
            if amount_diff <= self.amount_tolerance_tier2 * total:
                score = 0
                criteria = []
                
                # Score sur le montant
                if amount_diff == 0:
                    score += 50
                    criteria.append("Montant exact")
                elif amount_diff <= self.amount_tolerance_tier1 * total:
                    score += 35
                    criteria.append("Montant proche")
                else:
                    score += 20
                    criteria.append("Montant acceptable")
                
                # Score sur la date
                if pd.notnull(bank_date):
                    date_diff = abs((bank_date - date_receipt).days)
                    if date_diff <= self.days_delta:
                        score += 30
                        criteria.append(f"Date proche ({date_diff} jours)")
                
                # Score sur le nom du vendeur
                if vendor_receipt and vendor_bank:
                    similarity = fuzz.token_sort_ratio(vendor_receipt, vendor_bank)
                    if similarity >= self.similarity_threshold:
                        score += 20
                        criteria.append(f"Nom similaire ({similarity}%)")
                
                # Mettre à jour le meilleur match si ce score est meilleur
                if score > best_score:
                    best_score = score
                    best_match = {
                        "receipt_filename": filename,
                        "matched": True,
                        "match_score": score,
                        "match_criteria": criteria,
                        "receipt_total": total,
                        "receipt_date": date_receipt.strftime("%Y-%m-%d"),
                        "vendor_receipt": vendor_receipt,
                        "bank_amount": bank_amount,
                        "bank_date": bank_date.strftime("%Y-%m-%d") if pd.notnull(bank_date) else "",
                        "bank_vendor": vendor_bank,
                        "similarity_score": similarity if 'similarity' in locals() else 0
                    }
        
        return best_match

    def _add_unmatched_receipt(self, receipt: Dict[str, Any], reason: str) -> None:
        """
        Ajoute un reçu non matché à la liste des matchings.
        
        Args:
            receipt: Dictionnaire contenant les données du reçu
            reason: Raison pour laquelle le reçu n'a pas été matché
        """
        filename = self._extract_receipt_filename(receipt)
        total = self._extract_receipt_total(receipt)
        date_str = self._extract_receipt_date(receipt)
        vendor_name = self._extract_receipt_vendor_name(receipt)
        
        self.matchings.append({
            "receipt_filename": filename,
            "matched": False,
            "match_score": 0,
            "receipt_total": total,
            "receipt_date": date_str,
            "vendor_receipt": vendor_name,
            "reason": reason
        })

    def save_matching_results(self, suffix: str = "") -> Tuple[str, str]:
        """
        Sauvegarde les résultats du matching dans des fichiers JSON et CSV.
        
        Args:
            suffix: Suffixe à ajouter au nom des fichiers de sortie
            
        Returns:
            Tuple contenant les chemins des fichiers JSON et CSV
        """
        if not self.matchings:
            self.logger.warning("⚠️ Aucun résultat de matching à sauvegarder")
            return "", ""
        
        # Construire les noms de fichiers
        base_name = f"{self.matching_output_prefix}{suffix}"
        json_path = self.output_dir / f"{base_name}.json"
        csv_path = self.output_dir / f"{base_name}.csv"
        
        # Sauvegarder au format JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.matchings, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✅ Résultats de matching sauvegardés dans {json_path}")
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la sauvegarde JSON: {e}")
            return "", ""
        
        # Sauvegarder au format CSV
        try:
            pd.DataFrame(self.matchings).to_csv(csv_path, index=False)
            self.logger.info(f"✅ Résultats de matching sauvegardés dans {csv_path}")
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la sauvegarde CSV: {e}")
            return str(json_path), ""
        
        return str(json_path), str(csv_path)

    def enrich_bank_statements(self, matching_json_path: Optional[str] = None) -> List[str]:
        """
        Enrichit les relevés bancaires avec les informations des reçus correspondants.
        
        Args:
            matching_json_path: Chemin facultatif vers un fichier JSON de matching
                (utilise les résultats du dernier matching si non spécifié)
            
        Returns:
            Liste des chemins des fichiers CSV enrichis
        """
        # Charger les matchings si un chemin est spécifié
        if matching_json_path:
            try:
                with open(matching_json_path, "r", encoding="utf-8") as f:
                    self.matchings = json.load(f)
                self.logger.info(f"✅ {len(self.matchings)} matchings chargés depuis {matching_json_path}")
            except Exception as e:
                self.logger.error(f"❌ Erreur lors du chargement des matchings: {e}")
                return []
        
        # Vérifier que nous avons des matchings
        if not self.matchings:
            self.logger.error("❌ Aucun matching disponible pour enrichir les relevés")
            return []
        
        # Vérifier que nous avons des relevés bancaires
        if self.bank_statements.empty:
            self.logger.error("❌ Aucun relevé bancaire chargé pour l'enrichissement")
            return []
        
        # Créer un index des matchings pour un accès rapide
        match_index = {}
        for match in self.matchings:
            if match.get("matched", False):
                # Créer une clé basée sur le montant, la date et le vendeur
                key = (
                    round(float(match["bank_amount"]), 2),
                    match["bank_date"],
                    match["bank_vendor"].strip().upper()
                )
                match_index[key] = match["receipt_filename"] + ".json"
        
        self.logger.info(f"📊 {len(match_index)} matchings indexés pour l'enrichissement")
        
        # Traiter chaque fichier CSV original séparément
        enriched_files = []
        
        # Obtenir la liste des fichiers CSV dans le dossier
        csv_paths = list(self.bank_statements_dir.glob("*.csv"))
        
        for csv_path in csv_paths:
            # Charger le CSV original
            try:
                df = pd.read_csv(csv_path)
                if df.shape[1] == 1:  # Fichier mal séparé
                    df = pd.read_csv(csv_path, sep=";")
            except Exception as e:
                self.logger.error(f"❌ Erreur lors de la lecture de {csv_path}: {e}")
                continue
            
            # Convertir les dates
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            
            # Ajouter une colonne pour les noms de vendeurs nettoyés
            df["vendor_clean"] = df["vendor"].astype(str).str.strip().str.upper()
            
            # Ajouter une colonne pour la source (reçu correspondant)
            df["source"] = ""
            
            # Rechercher les correspondances pour chaque ligne
            for idx, row in df.iterrows():
                date = row["date"]
                if pd.isnull(date):
                    continue
                
                # Convertir le montant
                try:
                    amount = round(float(str(row["amount"]).replace(",", ".")), 2)
                except ValueError:
                    continue
                
                vendor = row["vendor_clean"]
                
                # Chercher une correspondance en tenant compte d'un delta de jours
                for delta in range(-self.days_delta, self.days_delta + 1):
                    candidate_date = (date + timedelta(days=delta)).strftime("%Y-%m-%d")
                    key = (amount, candidate_date, vendor)
                    
                    if key in match_index:
                        df.at[idx, "source"] = match_index[key]
                        break
            
            # Supprimer la colonne temporaire
            df.drop(columns=["vendor_clean"], inplace=True)
            
            # Enregistrer le fichier enrichi
            output_filename = f"{csv_path.stem}_{self.enriched_output_prefix}.csv"
            output_path = self.output_dir / output_filename
            
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"✅ Relevé enrichi: {csv_path.name} → {output_filename}")
            enriched_files.append(str(output_path))
        
        if enriched_files:
            self.logger.info(f"✅ {len(enriched_files)} relevés bancaires enrichis avec succès")
        else:
            self.logger.warning("⚠️ Aucun relevé bancaire n'a été enrichi")
        
        return enriched_files

    def run_complete_process(self, suffix: str = "") -> Dict[str, Any]:
        """
        Exécute le processus complet: matching et enrichissement des relevés.
        
        Args:
            suffix: Suffixe à ajouter aux noms des fichiers de sortie
            
        Returns:
            Dictionnaire contenant les résultats du processus
        """
        self.logger.info("🚀 Démarrage du processus complet...")
        
        # Charger les données
        self.load_receipts()
        self.load_bank_statements()
        
        # Vérifier que les données sont disponibles
        if not self.receipts or self.bank_statements.empty:
            self.logger.error("❌ Impossible de continuer sans données")
            return {
                "success": False,
                "error": "Données manquantes",
                "matching_count": 0,
                "enriched_files": []
            }
        
        # Effectuer le matching
        self.perform_matching()
        
        # Sauvegarder les résultats du matching
        json_path, csv_path = self.save_matching_results(suffix)
        
        # Enrichir les relevés bancaires
        enriched_files = self.enrich_bank_statements()
        
        # Compter les matchings réussis
        matched_count = sum(1 for m in self.matchings if m.get("matched", False))
        
        self.logger.info(f"✅ Processus complet terminé: {matched_count}/{len(self.matchings)} reçus matchés")
        
        return {
            "success": True,
            "matching_json": json_path,
            "matching_csv": csv_path,
            "matching_count": matched_count,
            "matching_total": len(self.matchings),
            "enriched_files": enriched_files
        }


if __name__ == "__main__":
    # Exemple d'utilisation
    matcher = ReceiptMatcher(
        receipts_json_path="resultats.json",
        bank_statements_dir="bank_statements",
        output_dir="output",
        matching_output_prefix="matchings",
        enriched_output_prefix="enriched",
        days_delta=3,
        amount_tolerance_tier1=0.05,
        amount_tolerance_tier2=0.10,
        similarity_threshold=85
    )
    
    # Exécuter le processus complet
    results = matcher.run_complete_process("_smart")
    
    if results["success"]:
        print("\n" + "="*50)
        print(f"✅ Résultats de matching dans:")
        print(f"   - {results['matching_json']}")
        print(f"   - {results['matching_csv']}")
        print(f"✅ {results['matching_count']}/{results['matching_total']} reçus matchés")
        print(f"✅ {len(results['enriched_files'])} relevés bancaires enrichis")
        print("="*50)
    else:
        print(f"\n❌ Erreur: {results['error']}")