import os
import json
import glob
import datetime
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from ollama import Client

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
INTERVIEW_FOLDER = os.path.join(BASE_DIR, "data")          # même dossier que llm_chain
REPORT_FOLDER    = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Phases connues (doit correspondre à llm_chain.py)
KNOWN_PHASES = [
    "INTRO_ETUDES",
    "MAPPING_PROJETS_OFFRE",
    "VALIDATION_HARD_SKILLS",
    "SOFT_SKILLS_RH",
    "QUESTIONS_CANDIDAT",
    "CONCLUSION",
]

PHASE_LABELS = {
    "INTRO_ETUDES":           "Présentation & Parcours",
    "MAPPING_PROJETS_OFFRE":  "Pertinence des Projets",
    "VALIDATION_HARD_SKILLS": "Compétences Techniques",
    "SOFT_SKILLS_RH":         "Soft Skills & Culture Fit",
    "QUESTIONS_CANDIDAT":     "Curiosité & Motivation",
    "CONCLUSION":             "Conclusion",
}

# Barème de scoring par phase (total = 100)
PHASE_MAX_SCORES = {
    "INTRO_ETUDES":           10,
    "MAPPING_PROJETS_OFFRE":  20,
    "VALIDATION_HARD_SKILLS": 40,
    "SOFT_SKILLS_RH":         20,
    "QUESTIONS_CANDIDAT":     10,
}

# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================
class InterviewEvaluator:

    def __init__(self, model_name: str = "qwen2.5"):
        self.llm_client = Client()
        self.model      = model_name

    # -------------------------------------------------------------------------
    # CHARGEMENT : auto-détection du dernier fichier interview_*.txt
    # -------------------------------------------------------------------------
    def load_latest_interview(self, specific_file: str = None) -> tuple[str, str]:
        """
        Charge le fichier d'entretien.
        - Si specific_file est fourni → charge ce fichier.
        - Sinon → charge le plus récent interview_*.txt dans INTERVIEW_FOLDER.

        Retourne (contenu_texte, chemin_fichier).
        """
        if specific_file:
            file_path = specific_file
        else:
            pattern = os.path.join(INTERVIEW_FOLDER, "interview_*.txt")
            files   = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(
                    f"❌ Aucun fichier interview_*.txt trouvé dans : {INTERVIEW_FOLDER}"
                )
            # Tri par date de modification → prend le plus récent
            file_path = max(files, key=os.path.getmtime)

        print(f"📂  Chargement : {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content, file_path

    # -------------------------------------------------------------------------
    # PARSING : extrait les échanges par phase depuis le fichier .txt
    # -------------------------------------------------------------------------
    def parse_interview(self, raw_text: str) -> dict:
        """
        Parse le fichier .txt et retourne un dict :
        {
            "meta": {"langue": ..., "duree": ..., "date": ...},
            "phases": {"INTRO_ETUDES": [...lignes...], ...},
            "all_candidate_lines": [...]
        }
        """
        meta   = {"langue": "N/A", "duree": "N/A", "date": "N/A"}
        phases = {p: [] for p in KNOWN_PHASES}
        all_candidate_lines = []

        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Extraction des métadonnées de l'en-tête
            if line.startswith("Date    :"):
                meta["date"] = line.split(":", 1)[1].strip()
            elif line.startswith("Langue  :") or line.startswith("ENTRETIEN RH —"):
                if "—" in line:
                    meta["langue"] = line.split("—", 1)[1].strip()
                else:
                    meta["langue"] = line.split(":", 1)[1].strip()
            elif line.startswith("Durée   :"):
                meta["duree"] = line.split(":", 1)[1].strip()

            # Extraction des lignes d'échange  [timestamp] [PHASE] Speaker: texte ...
            if line.startswith("[") and "] [" in line:
                try:
                    # Phase
                    phase_part = line.split("] [", 1)[1].split("]", 1)[0].strip()
                    rest       = line.split("] [", 1)[1].split("]", 1)[1].strip()

                    # Speaker + texte
                    if ": " in rest:
                        speaker, text_part = rest.split(": ", 1)
                        speaker = speaker.strip()
                        # Nettoie le suffixe "| Emotion: X | Action: Y"
                        text_clean = text_part.split("| Emotion:")[0].strip()

                        if phase_part in phases:
                            phases[phase_part].append({
                                "speaker": speaker,
                                "text":    text_clean,
                                "raw":     line,
                            })

                        if "Candidate" in speaker or "Candidat" in speaker:
                            all_candidate_lines.append(text_clean)
                except (IndexError, ValueError):
                    continue

        return {
            "meta":                 meta,
            "phases":               phases,
            "all_candidate_lines":  all_candidate_lines,
        }

    # -------------------------------------------------------------------------
    # NLU ASSESSMENT via LLM
    # -------------------------------------------------------------------------
    def run_nlu_assessment(self, parsed: dict, brain_scores: dict = None) -> dict:
        """
        Envoie l'entretien parsé au LLM et retourne le rapport JSON.
        """
        print(f"🧠  Analyse NLU en cours avec le modèle {self.model}...")

        # Construit un résumé par phase pour le prompt
        phase_summaries = []
        for phase, entries in parsed["phases"].items():
            if entries:
                label = PHASE_LABELS.get(phase, phase)
                texts = "\n".join(
                    f"  [{e['speaker']}]: {e['text']}" for e in entries
                )
                phase_summaries.append(f"### {label} ({phase})\n{texts}")

        interview_summary = "\n\n".join(phase_summaries) if phase_summaries else "(vide)"

        # Barème formaté pour le prompt
        bareme = "\n".join(
            f"  - {PHASE_LABELS[p]}: /{PHASE_MAX_SCORES[p]} pts"
            for p in PHASE_MAX_SCORES
        )

        nlu_prompt = f"""
Tu es un Expert RH senior chargé d'évaluer un candidat après un entretien.

LANGUE DE L'ENTRETIEN : {parsed['meta']['langue']}
DATE                  : {parsed['meta']['date']}
DURÉE                 : {parsed['meta']['duree']}

BARÈME DE NOTATION (total /100) :
{bareme}

ENTRETIEN PAR PHASES :
{interview_summary}

TÂCHE :
Génère un rapport d'évaluation JSON STRICT en respectant EXACTEMENT ce format.
Chaque score doit être un entier dans la limite du barème.
Le verdict doit être l'un de : "EMBAUCHE" | "A REVOIR" | "REJET"

FORMAT JSON ATTENDU (retourne UNIQUEMENT ce JSON, sans texte avant/après) :
{{
    "meta": {{
        "date_entretien": "{parsed['meta']['date']}",
        "langue":         "{parsed['meta']['langue']}",
        "duree":          "{parsed['meta']['duree']}",
        "date_rapport":   "{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    }},
    "scores_par_phase": {{
        "presentation_parcours":    {{"obtenu": <int /10>,  "max": 10,  "commentaire": "<str>"}},
        "pertinence_projets":       {{"obtenu": <int /20>,  "max": 20,  "commentaire": "<str>"}},
        "competences_techniques":   {{"obtenu": <int /40>,  "max": 40,  "commentaire": "<str>"}},
        "soft_skills_culture_fit":  {{"obtenu": <int /20>,  "max": 20,  "commentaire": "<str>"}},
        "curiosite_motivation":     {{"obtenu": <int /10>,  "max": 10,  "commentaire": "<str>"}}
    }},
    "score_technique":    <int 0-100>,
    "score_communication": <int 0-100>,
    "score_global":        <int 0-100>,
    "competences_detectees": ["<str>", "..."],
    "lacunes_identifiees":   ["<str>", "..."],
    "analyse_motivation":    "<str>",
    "analyse_soft_skills":   "<str>",
    "points_forts":          ["<str>", "..."],
    "points_amelioration":   ["<str>", "..."],
    "verdict_final":         "EMBAUCHE | A REVOIR | REJET",
    "recommandation_detail": "<str>"
}}
        """.strip()

        try:
            response = self.llm_client.generate(
                model=self.model,
                prompt=nlu_prompt,
                format="json",
                options={"temperature": 0.2},
            )
            # Fix : objet Ollama, pas un dict
            raw = response.response if hasattr(response, "response") else response["response"]

            # Nettoyage des éventuels backticks
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

            report = json.loads(raw)
            print("✅  Analyse NLU terminée.")
            return report

        except json.JSONDecodeError as e:
            print(f"❌  JSON invalide retourné par le LLM : {e}")
            return self._fallback_report(parsed, brain_scores)
        except Exception as e:
            print(f"Erreur LLM : {e}")
            return self._fallback_report(parsed, brain_scores)

    # -------------------------------------------------------------------------
    # RAPPORT DE SECOURS si le LLM échoue
    # -------------------------------------------------------------------------
    def _fallback_report(self, parsed: dict, brain_scores: dict = None) -> dict:
        """
        Rapport de secours si le LLM echoue.
        Utilise brain_scores (collectes tour par tour) pour calculer
        un score reel au lieu de tout mettre a 0.
        """
        brain_scores = brain_scores or {}

        phase_map = {
            "INTRO_ETUDES":           ("presentation_parcours",   10),
            "MAPPING_PROJETS_OFFRE":  ("pertinence_projets",      20),
            "VALIDATION_HARD_SKILLS": ("competences_techniques",  40),
            "SOFT_SKILLS_RH":         ("soft_skills_culture_fit", 20),
            "QUESTIONS_CANDIDAT":     ("curiosite_motivation",    10),
        }

        scores_par_phase = {}
        score_total = 0

        for phase_key, (label, max_pts) in phase_map.items():
            raw = brain_scores.get(phase_key)  # valeur 1-10 ou None
            obtenu = round((raw / 10) * max_pts) if raw is not None else 0
            score_total += obtenu
            scores_par_phase[label] = {
                "obtenu":      obtenu,
                "max":         max_pts,
                "commentaire": "Score estime depuis les evaluations par tour.",
            }

        pct = round((score_total / 100) * 100)
        if pct >= 65:
            verdict = "EMBAUCHE"
        elif pct >= 45:
            verdict = "A REVOIR"
        else:
            verdict = "REJET"

        return {
            "meta": {
                "date_entretien": parsed["meta"]["date"],
                "langue":         parsed["meta"]["langue"],
                "duree":          parsed["meta"]["duree"],
                "date_rapport":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "scores_par_phase":      scores_par_phase,
            "score_technique":       pct,
            "score_communication":   0,
            "score_global":          pct,
            "competences_detectees": [],
            "lacunes_identifiees":   ["Analyse NLU indisponible — scores estimes depuis l entretien."],
            "analyse_motivation":    "N/A",
            "analyse_soft_skills":   "N/A",
            "points_forts":          [],
            "points_amelioration":   [],
            "verdict_final":         verdict,
            "recommandation_detail": f"Score global estime : {pct}/100. Revue manuelle recommandee.",
        }

    # -------------------------------------------------------------------------
    # AFFICHAGE CONSOLE
    # -------------------------------------------------------------------------
    def display_results(self, report: dict):
        print("\n" + "═" * 62)
        print("   🏆  RÉSULTATS DE L'ÉVALUATION NLU")
        print("═" * 62)

        meta = report.get("meta", {})
        print(f"  📅 Date entretien  : {meta.get('date_entretien', 'N/A')}")
        print(f"  🌐 Langue          : {meta.get('langue', 'N/A')}")
        print(f"  ⏱️  Durée           : {meta.get('duree', 'N/A')}")
        print()

        # Scores par phase
        phases = report.get("scores_par_phase", {})
        if phases:
            print("  📊 SCORES PAR PHASE :")
            for key, val in phases.items():
                obtenu = val.get("obtenu", 0)
                maxi   = val.get("max", 10)
                pct    = round((obtenu / maxi) * 100) if maxi else 0
                bar    = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                label  = key.replace("_", " ").title()
                print(f"    {label:<30} {bar}  {obtenu}/{maxi}")
                commentaire = val.get("commentaire", "")
                if commentaire:
                    print(f"    {'':30}  → {commentaire}")
            print()

        print(f"  ⭐ Score Technique   : {report.get('score_technique', 0)}/100")
        print(f"  💬 Score Communication: {report.get('score_communication', 0)}/100")
        print(f"  🎯 Score Global      : {report.get('score_global', 0)}/100")
        print()

        # Compétences
        competences = report.get("competences_detectees", [])
        if competences:
            print("  ✅ COMPÉTENCES DÉTECTÉES :")
            for c in competences:
                print(f"    • {c}")
            print()

        # Lacunes
        lacunes = report.get("lacunes_identifiees", [])
        if lacunes:
            print("  ⚠️  LACUNES IDENTIFIÉES :")
            for l in lacunes:
                print(f"    • {l}")
            print()

        # Points forts
        forts = report.get("points_forts", [])
        if forts:
            print("  💪 POINTS FORTS :")
            for p in forts:
                print(f"    • {p}")
            print()

        # Points à améliorer
        amelio = report.get("points_amelioration", [])
        if amelio:
            print("  📈 POINTS À AMÉLIORER :")
            for p in amelio:
                print(f"    • {p}")
            print()

        # Motivation & Soft skills
        motivation = report.get("analyse_motivation", "")
        if motivation:
            print(f"  🎯 MOTIVATION : {motivation}")
            print()

        soft = report.get("analyse_soft_skills", "")
        if soft:
            print(f"  🤝 SOFT SKILLS : {soft}")
            print()

        # Verdict
        verdict = report.get("verdict_final", "N/A")
        verdict_icon = {"EMBAUCHE": "✅", "A REVOIR": "⚠️ ", "REJET": "❌"}.get(verdict, "❓")
        print(f"  {verdict_icon} VERDICT FINAL : {verdict}")
        reco = report.get("recommandation_detail", "")
        if reco:
            print(f"     → {reco}")
        print("═" * 62)

    # -------------------------------------------------------------------------
    # SAUVEGARDE JSON
    # -------------------------------------------------------------------------
    def save_json_report(self, report: dict, source_file: str = "") -> str:
        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        langue      = report.get("meta", {}).get("langue", "unknown").replace(" ", "_")
        output_path = os.path.join(REPORT_FOLDER, f"evaluation_{langue}_{timestamp}.json")

        # Ajoute le chemin source dans les métadonnées
        if source_file:
            report.setdefault("meta", {})["source_interview"] = source_file

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print(f"💾  Rapport JSON sauvegardé : {output_path}")
        return output_path


# =============================================================================
# EXÉCUTION
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Évaluateur NLU d'entretien RH")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Chemin vers un fichier interview_*.txt spécifique (optionnel)."
             " Par défaut : dernier fichier généré."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen2.5",
        help="Modèle Ollama à utiliser (défaut: qwen2.5)"
    )
    args = parser.parse_args()

    evaluator = InterviewEvaluator(model_name=args.model)

    try:
        # 1. Chargement
        raw_text, file_path = evaluator.load_latest_interview(
            specific_file=args.file
        )

        # 2. Parsing structuré
        parsed = evaluator.parse_interview(raw_text)
        print(f"✅  Entretien parsé — langue: {parsed['meta']['langue']} | "
              f"durée: {parsed['meta']['duree']}")
        print(f"   Répliques candidat trouvées : {len(parsed['all_candidate_lines'])}")

        # 3. Analyse NLU
        final_report = evaluator.run_nlu_assessment(parsed)

        # 4. Affichage + sauvegarde
        if final_report:
            evaluator.display_results(final_report)
            evaluator.save_json_report(final_report, source_file=file_path)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"❌  Erreur fatale : {e}")
        raise