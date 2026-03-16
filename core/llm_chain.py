import os
import gc
import json
import shutil
import datetime
import time
from typing import List, Optional, AsyncGenerator

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from ollama import Client, AsyncClient


CHROMA_PATH = "data/chroma_db"
UPLOADS_DIR = "data"
# LOG_FILE est maintenant un attribut d instance (self.log_file) cree dans __init__

MAX_HISTORY_TURNS = 6

SUPPORTED_EXTS = {
    "cv":           {".pdf", ".docx"},
    "job_offer":    {".pdf", ".docx"},
    "company_info": {".pdf", ".docx", ".txt"},
}

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

DEFAULT_RESPONSE = {
    "candidate_sentiment": "neutre",
    "text":                "Pouvez-vous reformuler votre reponse ?",
    "next_step":           False,
    "score":               None,
    "interview_ended":     False,
}


class HRInteractiveBrain:

    def __init__(self, target_lang: str, duration_minutes: int):

        OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.llm_client       = Client(host=OLLAMA_HOST)
        self.llm_async_client = AsyncClient(host=OLLAMA_HOST)

        self.target_lang      = target_lang
        self.duration_minutes = duration_minutes

        self.chat_history: List[dict] = []
        self.full_log: List[dict]     = []

        self.start_time: Optional[float] = None

        self.steps = [
            "INTRO_ETUDES",
            "MAPPING_PROJETS_OFFRE",
            "VALIDATION_HARD_SKILLS",
            "SOFT_SKILLS_RH",
            "QUESTIONS_CANDIDAT",
            "CONCLUSION",
        ]

        self.current_step_index = 0
        self.db: Optional[Chroma] = None

        self.ingested_docs = {
            "cv":           [],
            "job_offer":    [],
            "company_info": [],
        }

        self.scores: dict = {step: None for step in self.steps}

        # Ameliorations 1-10
        self.conclusion_done: bool = False
        self.conclusion_turn_count: int = 0   # nb d echanges effectues en phase CONCLUSION
        self.last_question: str = ""
        self.total_turns: int = 0
        self.MIN_WORDS_THRESHOLD: int = 8
        self.dimension_scores: dict = {
            "communication": [],
            "technique":     [],
            "motivation":    [],
        }

        # ── STRUCTURE 2/3 TECHNIQUE + 1/3 RH ────────────────────────────
        # 2/3 du temps : INTRO + PROJETS + COMPETENCES TECHNIQUES
        # 1/3 du temps : SOFT SKILLS + QUESTIONS CANDIDAT + CONCLUSION
        # Valide pour toutes les durees : 5min, 10min, 25min, 60min, 90min

        _t_tech = round(duration_minutes * 2 / 3)   # bloc technique
        _t_rh   = duration_minutes - _t_tech          # bloc RH

        # Budget temps par phase — sans minimum fixe pour respecter la duree reelle.
        # Pour les tres courtes durees (<10min), on divise proportionnellement
        # sans plancher, sauf CONCLUSION qui garde 1min minimum.
        # Les phases PROJETS et TECH sont plafonnees a leur valeur reelle
        # pour ne pas depasser la duree totale.
        if duration_minutes >= 10:
            self._phase_time_budget: dict = {
                "INTRO_ETUDES":           max(1, round(_t_tech * 0.20)),
                "MAPPING_PROJETS_OFFRE":  max(2, round(_t_tech * 0.40)),
                "VALIDATION_HARD_SKILLS": max(2, round(_t_tech * 0.40)),
                "SOFT_SKILLS_RH":         max(1, round(_t_rh   * 0.35)),
                "QUESTIONS_CANDIDAT":     max(1, round(_t_rh   * 0.45)),
                "CONCLUSION":             max(1, round(_t_rh   * 0.20)),
            }
        else:
            # Courte duree (<10min) : distribution strictement proportionnelle
            self._phase_time_budget: dict = {
                "INTRO_ETUDES":           max(1, round(_t_tech * 0.20)),
                "MAPPING_PROJETS_OFFRE":  max(1, round(_t_tech * 0.40)),
                "VALIDATION_HARD_SKILLS": max(1, round(_t_tech * 0.40)),
                "SOFT_SKILLS_RH":         max(1, round(_t_rh   * 0.35)),
                "QUESTIONS_CANDIDAT":     max(1, round(_t_rh   * 0.45)),
                "CONCLUSION":             max(1, round(_t_rh   * 0.20)),
            }

        # Questions max par phase selon la duree
        # INTRO=1 toujours | CONCLUSION=1 toujours
        _factor = min(5, max(1, round(duration_minutes / 15)))
        self.MAX_Q_PER_PHASE: dict = {
            "INTRO_ETUDES":           1,
            "MAPPING_PROJETS_OFFRE":  max(1, _factor),
            "VALIDATION_HARD_SKILLS": max(1, _factor + 1),
            "SOFT_SKILLS_RH":         max(1, _factor),
            "QUESTIONS_CANDIDAT":     max(1, _factor + 1),
            "CONCLUSION":             1,
        }

        # MAX_TURNS >= total_questions + 2 et coherent avec la duree
        _total_q = sum(self.MAX_Q_PER_PHASE.values())
        self.MAX_TURNS: int = max(_total_q + 2, min(80, int(duration_minutes / 1.5)))

        self._phase_start_time: dict = {}
        self.questions_per_phase: dict = {step: 0 for step in self.steps}

        # Fichier log unique par session
        import os as _os
        _os.makedirs("data", exist_ok=True)
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file: str = f"data/interview_{_ts}.txt"

        self._reset_db()
        self.db = self._initialize_db()

    # ─────────────────────────────────────────
    # Chroma DB
    # ─────────────────────────────────────────

    def _reset_db(self):
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        print("Nouvelle session RAG")

    def _initialize_db(self):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=EMBEDDINGS,
        )

    # ─────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────

    def save_to_log(self, speaker: str, text: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        phase     = self.steps[self.current_step_index]
        line      = f"[{timestamp}] [{phase}] {speaker}: {text}\n"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line)

        self.full_log.append({
            "timestamp": timestamp,
            "phase":     phase,
            "speaker":   speaker,
            "text":      text,
        })

    # ─────────────────────────────────────────
    # Timer
    # ─────────────────────────────────────────

    def get_time_remaining(self) -> float:
        if self.start_time is None:
            return self.duration_minutes
        elapsed = (time.time() - self.start_time) / 60
        return max(0, round(self.duration_minutes - elapsed, 1))

    # ─────────────────────────────────────────
    # Document ingestion
    # ─────────────────────────────────────────

    def ingest_document(self, file_path: str, doc_type: str):
        if not os.path.exists(file_path):
            return

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        else:
            return

        documents = loader.load()
        splitter  = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks    = splitter.split_documents(documents)

        for doc in chunks:
            doc.metadata["type"] = doc_type

        if chunks:
            self.db.add_documents(chunks)
            self.ingested_docs[doc_type].append(os.path.basename(file_path))
            print(f"OK {doc_type} indexe")

    # ─────────────────────────────────────────
    # Greeting
    # ─────────────────────────────────────────

    def get_initial_greeting(self) -> str:
        self.start_time = time.time()

        with open(self.log_file, "a", encoding="utf-8") as _f:
            _f.write(f"Date    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _f.write(f"Langue  : {self.target_lang}\n")
            _f.write(f"Duree   : {self.duration_minutes} minutes\n\n")

        greeting = {
            "Francais": f"Bonjour, merci d etre present pour cet entretien de {self.duration_minutes} minutes.",
            "Francais_accent": f"Bonjour, merci d'etre present pour cet entretien de {self.duration_minutes} minutes.",
            "Anglais":  f"Hello, thank you for joining this {self.duration_minutes}-minute interview.",
            "Arabe":    f"مرحباً بك في هذه المقابلة لمدة {self.duration_minutes} دقيقة.",
        }

        msg = greeting.get(self.target_lang,
              "Bonjour, merci d'etre present pour cet entretien de "
              f"{self.duration_minutes} minutes.")
        self.chat_history.append({"role": "Recruteur", "text": msg})
        self.save_to_log("Avatar", msg)
        return msg

    # ─────────────────────────────────────────
    # RAG — SEPARE PAR SOURCE (correction hallucination)
    # ─────────────────────────────────────────

    def _get_rag_context(self, user_input: str) -> str:
        """
        Retourne le contexte RAG SEPARE par type de document avec etiquettes claires.
        Chaque section est etiquetee CV / OFFRE / ENTREPRISE pour que le LLM
        ne confonde jamais les informations du candidat avec celles du poste.
        """
        results = self.db.similarity_search(user_input, k=9)
        if not results:
            return ""

        pairs  = [[user_input, doc.page_content] for doc in results]
        scores = RERANKER.predict(pairs)
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)

        # Max 2 chunks par type de source
        buckets = {"cv": [], "job_offer": [], "company_info": []}
        for score, doc in ranked:
            doc_type = doc.metadata.get("type", "unknown")
            if doc_type in buckets and len(buckets[doc_type]) < 2:
                buckets[doc_type].append(doc.page_content)

        sections = []

        if buckets["cv"]:
            sections.append(
                "=== CV DU CANDIDAT ===\n"
                "(Ce qui suit decrit le parcours, les competences et les PROJETS REELS du candidat.)\n"
                + "\n---\n".join(buckets["cv"])
            )

        if buckets["job_offer"]:
            sections.append(
                "=== OFFRE DE POSTE / STAGE ===\n"
                "(Ce qui suit decrit les EXIGENCES du poste et les PROJETS DE L ENTREPRISE attendus.\n"
                "NE PAS confondre avec les projets du candidat.)\n"
                + "\n---\n".join(buckets["job_offer"])
            )

        if buckets["company_info"]:
            sections.append(
                "=== BASE DE CONNAISSANCE ENTREPRISE ===\n"
                "(Ce qui suit decrit la culture, les valeurs et le contexte de l entreprise.)\n"
                + "\n---\n".join(buckets["company_info"])
            )

        return "\n\n".join(sections) if sections else ""

    async def _get_rag_context_async(self, user_input: str) -> str:
        """Version async : execute le RAG dans un thread."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_rag_context, user_input)

    def _get_recent_history(self) -> str:
        recent = self.chat_history[-(MAX_HISTORY_TURNS * 2):]
        return "\n".join(f"{h['role']}: {h['text']}" for h in recent)

    # ─────────────────────────────────────────
    # Helpers metier
    # ─────────────────────────────────────────

    def _candidate_has_no_questions(self, user_input: str) -> bool:
        triggers = [
            "pas de question", "aucune question", "non merci", "pas d'autres questions",
            "c'est tout", "rien d'autre", "no question", "no thanks", "nothing",
            "لا أسئلة", "لا شكرا", "لا يوجد",
        ]
        lowered = user_input.lower().strip()
        return any(t in lowered for t in triggers)

    def _max_q_for_phase(self, phase: str) -> int:
        return self.MAX_Q_PER_PHASE.get(phase, 2) if isinstance(self.MAX_Q_PER_PHASE, dict) else self.MAX_Q_PER_PHASE

    def _phase_budget_summary(self) -> str:
        rows = []
        for step in self.steps:
            budget    = self._phase_time_budget.get(step, 0)
            max_q     = self._max_q_for_phase(step)
            done_q    = self.questions_per_phase.get(step, 0)
            elapsed   = (time.time() - self._phase_start_time[step]) / 60 if step in self._phase_start_time else 0
            remaining = max(0, round(budget - elapsed, 1))
            marker    = " <-- ACTUELLE" if step == self.steps[self.current_step_index] else ""
            rows.append(f"  {step}: {budget}min (reste {remaining}min) | questions {done_q}/{max_q}{marker}")
        return "\n".join(rows)

    def _is_too_short(self, user_input: str) -> bool:
        words = user_input.strip().split()
        return len(words) < self.MIN_WORDS_THRESHOLD

    def _strip_greeting(self, text: str) -> str:
        """
        Supprime en code les formules de salutation en début de réponse.
        Le LLM ignore parfois les instructions du prompt — cette fonction
        est un filet de sécurité déterministe côté Python.

        Exemples supprimés :
          "Bonjour, c'est avec plaisir..."  → "C'est avec plaisir..."
          "Bonjour ! Pouvez-vous..."        → "Pouvez-vous..."
          "Hello, thank you for..."         → "Thank you for..."
          "Bonjour Zayneb, ..."             → "..."
        """
        import re

        # Patterns de salutation à supprimer en début de texte (insensible à la casse)
        # On supprime la salutation + la virgule/espace/point d'exclamation qui suit
        greeting_patterns = [
            # Français
            r"^bonjour\s+\w+\s*[,!.]?\s*",   # "Bonjour Zayneb, "
            r"^bonjour\s*[,!.]?\s*",           # "Bonjour, " / "Bonjour ! "
            r"^bonsoir\s*[,!.]?\s*",
            r"^bienvenue\s*[,!.]?\s*",
            r"^salut\s*[,!.]?\s*",
            # Anglais
            r"^hello\s+\w+\s*[,!.]?\s*",      # "Hello Zayneb, "
            r"^hello\s*[,!.]?\s*",
            r"^hi\s+\w+\s*[,!.]?\s*",
            r"^hi\s*[,!.]?\s*",
            r"^good\s+(morning|afternoon|evening)\s*[,!.]?\s*",
            # Arabe
            r"^مرحبا[ً\s]*[،,!.]?\s*",
            r"^أهلا[ً\s]*[،,!.]?\s*",
        ]

        stripped = text.strip()
        for pattern in greeting_patterns:
            new_text = re.sub(pattern, "", stripped, flags=re.IGNORECASE).strip()
            if new_text and new_text != stripped:
                # Remettre la majuscule sur le premier caractère
                stripped = new_text[0].upper() + new_text[1:] if new_text else new_text
                break  # Un seul passage suffit

        return stripped

    def _is_farewell_speech(self, speech_text: str) -> bool:
        """
        Detecte si la reponse de l avatar contient une formule de conge
        qui signale la fin reelle de l entretien.
        On detecte cela en code plutot que de laisser le LLM decider —
        plus fiable car le LLM oublie parfois de mettre interview_ended=true.
        """
        farewell_triggers = [
            # Francais — conge explicite
            "bonne journee", "bonne journée", "excellente journee", "excellente journée",
            "bonne continuation", "au revoir", "a bientot", "à bientôt",
            # Francais — suivi RH
            "nous reviendrons vers vous", "l equipe rh reviendra", "notre equipe reviendra",
            "equipe rh vous contactera", "vous recontacter prochainement",
            "suite du processus", "prochainement",
            # Anglais
            "have a great day", "goodbye", "good bye", "best of luck",
            "we will get back to you", "our team will contact",
            "thank you for your time", "wish you all the best",
            "have a good day", "take care",
            # Arabe
            "وداعا", "مع السلامة", "يوما سعيدا", "سنتواصل معك",
            "نتمنى لك يوما", "شكرا لحضورك",
        ]
        lowered = speech_text.lower()
        return any(trigger in lowered for trigger in farewell_triggers)

    def get_final_score(self) -> dict:
        phase_scores = {k: v for k, v in self.scores.items() if v is not None}
        if not phase_scores:
            return {"score_final": 0, "scores_par_phase": {}, "nb_echanges": self.total_turns}
        moyenne = round(sum(phase_scores.values()) / len(phase_scores), 1)
        return {
            "score_final":      moyenne,
            "scores_par_phase": phase_scores,
            "nb_echanges":      self.total_turns,
        }

    # ─────────────────────────────────────────
    # Prompts
    # ─────────────────────────────────────────

    def _build_speech_system(self, current_step: str, history: str, context: str,
                              short_answer: bool = False) -> str:
        time_left = self.get_time_remaining()

        transitions = {
            "MAPPING_PROJETS_OFFRE":  "Merci pour cette presentation. J aimerais maintenant aborder vos projets.",
            "VALIDATION_HARD_SKILLS": "Passons maintenant a vos competences techniques.",
            "SOFT_SKILLS_RH":         "J aimerais vous poser une question sur votre facon de travailler.",
            "QUESTIONS_CANDIDAT":     "Nous avons fait le tour des questions techniques.",
            "CONCLUSION":             "",
        }

        anti_repeat_rule = (
            "ANTI-REPETITION STRICTE : N utilise JAMAIS les formules "
            "Vous avez dit que..., Comme vous l avez mentionne..., "
            "Vous avez indique.... Passe directement a la question suivante "
            "ou approfondis un point precis sans reformuler."
        )

        last_q_rule = ""
        if self.last_question:
            last_q_rule = (
                "INTERDICTION DE REPETER : La derniere question posee etait : "
                f'"{self.last_question}". '
                "Ne la repose PAS. Trouve un angle different."
            )

        short_answer_rule = ""
        if short_answer:
            short_answer_rule = (
                "RELANCE OBLIGATOIRE : La reponse du candidat est trop courte. "
                "Demande-lui de developper davantage."
            )

        phase_instructions = {
            "INTRO_ETUDES": (
                "- Accueille chaleureusement le candidat en une phrase courte.\n"
                "- Pose UNE SEULE question qui invite le candidat a se presenter COMPLETEMENT :\n"
                "  sa formation academique, les projets qu il a realises, et sa motivation pour ce poste.\n"
                "- La question doit etre ouverte et generale — NE PAS mentionner un projet ou une\n"
                "  experience specifique du CV. Le candidat doit se presenter librement.\n"
                "- Exemple de style (ne pas copier mot pour mot) :\n"
                "  Pouvez-vous vous presenter — votre parcours, les projets sur lesquels vous avez\n"
                "  travaille et ce qui vous a amene a postuler pour ce poste ?\n"
                "- NE PAS poser de questions sur un projet particulier dans cette phase.\n"
                "- NE PAS utiliser le CV pour orienter la question ici.\n"
                "- Duree estimee : 2-3 minutes. Laisse le candidat developper librement."
            ),
            "MAPPING_PROJETS_OFFRE": (
                transitions.get("MAPPING_PROJETS_OFFRE", "") + "\n"
                "- Questions UNIQUEMENT sur les projets du CV lies a l offre.\n"
                "- Choisis 1 ou 2 projets les plus pertinents.\n"
                "- Appuie-toi sur la section CV DU CANDIDAT pour personnaliser la question.\n"
                "- UNE seule question a la fois."
            ),
            "VALIDATION_HARD_SKILLS": (
                transitions.get("VALIDATION_HARD_SKILLS", "") + "\n"
                "- Valide 1 ou 2 competences techniques presentes dans le CV ET demandees dans l offre.\n"
                "- Questions precises et techniques basees sur ce que le CV mentionne.\n"
                "- UNE seule question a la fois."
            ),
            "SOFT_SKILLS_RH": (
                transitions.get("SOFT_SKILLS_RH", "") + "\n"
                "- Question comportementale liee au contexte du poste.\n"
                "- Methode STAR implicitement, sans la nommer.\n"
                "- UNE seule question courte et directe."
            ),
            "QUESTIONS_CANDIDAT": (
                transitions.get("QUESTIONS_CANDIDAT", "") + "\n"
                "- Informe le candidat que l entretien technique est termine.\n"
                "- Demande-lui s il a des questions sur le poste, l equipe ou l entreprise.\n"
                "- Reponds concisement en utilisant la section BASE DE CONNAISSANCE si disponible.\n"
                "- Si le candidat n a pas de questions, annonce la conclusion naturellement."
            ),
            "CONCLUSION": (
                "INSTRUCTIONS STRICTES POUR LA CONCLUSION :\n"
                "- Remercie le candidat chaleureusement en 1 a 2 phrases.\n"
                "- Indique que l equipe RH reviendra vers lui prochainement avec la suite du processus.\n"
                "- Termine OBLIGATOIREMENT par une formule de conge explicite,\n"
                "  par exemple : 'Bonne journee et bonne continuation !'.\n"
                "  La formule de conge est OBLIGATOIRE — sans elle, l entretien ne peut pas se terminer.\n"
                "- NE POSE ABSOLUMENT AUCUNE QUESTION. Meme pas 'Cela vous semble-t-il pertinent ?'\n"
                "  meme pas 'Avez-vous d autres remarques ?'. Aucune question, aucune.\n"
                "- Maximum 4 phrases au total.\n"
                "- C est la DERNIERE prise de parole de l entretien. Il se termine ici."
            ),
        }

        phase_guide = phase_instructions.get(current_step, "")

        rules = [
            "1. NATURALITE : Parle comme un vrai recruteur humain. Transitions naturelles.",
            "2. UNE SEULE QUESTION PAR REPONSE.",
            "3. PERTINENCE : Base-toi UNIQUEMENT sur les elements du CV et de l offre dans le contexte.",
            f"4. {anti_repeat_rule}",
            "5. PAS DE VALIDATION EXCESSIVE : Evite Tres bien !, Excellent !",
            f"6. RESPECT DE LA DUREE : Il reste {time_left} minutes.",
            "7. Si le temps est ecoule et la phase n est pas QUESTIONS_CANDIDAT ni CONCLUSION : conclus brievement puis demande au candidat s il a des questions.",
            "8. CONCLUSION UNIQUEMENT : Si la phase est CONCLUSION, tu ne poses AUCUNE question."
            "   Tu remercies, tu annonces le suivi RH, et tu termines par 'Bonne journee !'."
            "   Toute question en CONCLUSION est une erreur grave.",
        ]
        if last_q_rule:
            rules.append(f"8. {last_q_rule}")
        if short_answer_rule:
            rules.append(f"9. {short_answer_rule}")

        rules_text = "\n".join(rules)

        return f"""Tu es un recruteur RH experimente qui conduit un entretien professionnel naturel et fluide.

=== CONTEXTE ===
Langue de l entretien : {self.target_lang}
Phase actuelle : {current_step}
Temps restant : {time_left} minutes
Tours effectues : {self.total_turns} / {self.MAX_TURNS}

=== INSTRUCTIONS GENERALES (OBLIGATOIRES) ===
{rules_text}

=== INSTRUCTIONS POUR LA PHASE ACTUELLE : {current_step} ===
{phase_guide}

=== HISTORIQUE DE LA CONVERSATION ===
{history}

=== BUDGET TEMPS PAR PHASE ===
Structure : 2/3 temps technique (INTRO + PROJETS + COMPETENCES) | 1/3 temps RH (SOFT + Q&A + CONCLUSION)
{self._phase_budget_summary()}
-> Si le budget ou le nb de questions d une phase est atteint : conclus et passe a la suivante.

=== DOCUMENTS DE REFERENCE ===
REGLE CRITIQUE — LIS ATTENTIVEMENT AVANT DE REPONDRE :

Du CV (section "CV DU CANDIDAT") :
  -> Ce sont les projets, experiences et competences REELS du candidat.
  -> Utilise cette section pour poser des questions sur ce que le candidat a FAIT.

De l OFFRE (section "OFFRE DE POSTE") :
  -> Ce sont les missions, competences et projets ATTENDUS par l entreprise.
  -> NE JAMAIS presenter un projet de l offre comme si c etait un projet du candidat.
  -> Exemple INTERDIT : "Vous avez travaille sur l avatar 3D..." si c est dans l offre.
  -> Exemple CORRECT  : "Le poste implique de travailler sur un avatar 3D, avez-vous une experience similaire ?"

De la KB ENTREPRISE (section "BASE DE CONNAISSANCE") :
  -> Utilise uniquement pour repondre aux questions du candidat sur l entreprise.

{context}

=== FORMAT DE SORTIE ===
Genere UNIQUEMENT la reponse verbale du recruteur, en texte brut, sans JSON, sans titre, sans balises.
"""

    def _build_meta_system(self, current_step: str, user_input: str, speech_text: str) -> str:
        time_left  = self.get_time_remaining()
        steps_list = ", ".join(self.steps)

        return f"""Tu analyses un echange d entretien RH pour decider des metadonnees de controle.

=== CONTEXTE ===
Phase actuelle : {current_step}
Temps restant : {time_left} minutes
Sequence des phases : {steps_list}

=== ECHANGE ANALYSE ===
Reponse du candidat : {user_input}
Reponse generee par le recruteur : {speech_text}

=== REGLES DE DECISION ===

1. "next_step" (true/false) :
   - Mets true si la phase actuelle est suffisamment couverte ET que le recruteur passe naturellement a autre chose.
   - Pour QUESTIONS_CANDIDAT : mets true si le candidat n a pas de questions OU si ses questions ont ete repondues.
   - Pour CONCLUSION : mets true TOUJOURS.
   - Sinon mets false.

2. "interview_ended" (true/false) :
   - Mets true UNIQUEMENT si la phase actuelle est CONCLUSION ET que le recruteur
     a explicitement prononce une formule de conge (bonne journee, au revoir, bonne continuation,
     nous reviendrons vers vous, suite du processus, etc.).
   - Si le recruteur est en CONCLUSION mais n a pas encore dit au revoir, mets false.
   - Sinon (toute autre phase) : false obligatoirement.

3. "score" (1-10 ou null) :
   - Donne un score pour MAPPING_PROJETS_OFFRE, VALIDATION_HARD_SKILLS, SOFT_SKILLS_RH.
   - Evalue : pertinence (0-4) + clarte (0-3) + profondeur (0-3) = /10.
   - null pour INTRO_ETUDES, QUESTIONS_CANDIDAT, CONCLUSION.

4. "candidate_sentiment" : "positif" | "neutre" | "negatif" | "hesitant"

5. "communication_score" (1-10 ou null) :
   - Evalue la qualite de communication (clarte, structure, vocabulaire).
   - Donne un score pour toutes les phases sauf CONCLUSION.

6. "motivation_score" (1-10 ou null) :
   - Evalue l enthousiasme et la motivation percue.
   - Score pour INTRO_ETUDES et SOFT_SKILLS_RH. null pour les autres.

=== FORMAT DE SORTIE (JSON STRICT, AUCUN TEXTE AUTOUR) ===
{{
  "candidate_sentiment":  "...",
  "next_step":            true,
  "score":                7,
  "communication_score":  8,
  "motivation_score":     null,
  "interview_ended":      false
}}
"""

    def _parse_meta(self, raw_response: str) -> dict:
        meta = {
            "candidate_sentiment": DEFAULT_RESPONSE["candidate_sentiment"],
            "next_step":           DEFAULT_RESPONSE["next_step"],
            "score":               DEFAULT_RESPONSE["score"],
            "interview_ended":     DEFAULT_RESPONSE["interview_ended"],
            "communication_score": None,
            "motivation_score":    None,
        }
        if not raw_response or not raw_response.strip():
            return meta
        try:
            clean  = raw_response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
            for k in meta:
                if k in parsed:
                    meta[k] = parsed[k]
        except Exception as e:
            print("META PARSE ERROR:", e)
        return meta

    def _update_state(self, user_input: str, speech_text: str, meta: dict):
        current_step = self.steps[self.current_step_index]

        self.save_to_log("Candidate", user_input)
        self.save_to_log("Avatar",    speech_text)

        self.total_turns += 1
        self.questions_per_phase[current_step] = self.questions_per_phase.get(current_step, 0) + 1

        if meta.get("score") is not None:
            self.scores[current_step] = meta["score"]
        if meta.get("communication_score") is not None:
            self.dimension_scores["communication"].append(meta["communication_score"])
        if meta.get("motivation_score") is not None:
            self.dimension_scores["motivation"].append(meta["motivation_score"])
        if meta.get("score") is not None and current_step == "VALIDATION_HARD_SKILLS":
            self.dimension_scores["technique"].append(meta["score"])

        if current_step == "CONCLUSION":
            self.conclusion_turn_count += 1
            # Ne terminer qu apres au moins 1 echange ET que l avatar a dit au revoir.
            # Cela evite de fermer l entretien des la premiere reponse en CONCLUSION
            # si l avatar n a pas encore prononce la formule de conge.
            if self.conclusion_turn_count >= 1 and self._is_farewell_speech(speech_text):
                self.conclusion_done = True
        # Securite supplementaire : si l avatar dit au revoir dans n importe
        # quelle phase, on considere l entretien termine
        if self._is_farewell_speech(speech_text):
            self.conclusion_done = True
            # S assurer qu on est bien en phase CONCLUSION dans les logs
            conclusion_idx = self.steps.index("CONCLUSION")
            if self.current_step_index < conclusion_idx:
                self.current_step_index = conclusion_idx

        questions_index = self.steps.index("QUESTIONS_CANDIDAT")
        time_is_up      = self.get_time_remaining() <= 0

        # Enregistrer le debut de la phase si pas encore fait
        if current_step not in self._phase_start_time:
            self._phase_start_time[current_step] = time.time()

        # Temps passe dans la phase courante (minutes)
        phase_elapsed  = (time.time() - self._phase_start_time[current_step]) / 60
        phase_budget   = self._phase_time_budget.get(current_step, 999)
        phase_overtime = phase_elapsed >= phase_budget

        # Nb questions faites vs max autorise pour cette phase
        max_q  = self.MAX_Q_PER_PHASE.get(current_step, 2)
        q_done = self.questions_per_phase.get(current_step, 0)

        def _advance():
            self.current_step_index += 1
            nxt = self.steps[self.current_step_index]
            self._phase_start_time.setdefault(nxt, time.time())

        # Priorite 1 : temps global ecoule ou max tours → aller a QUESTIONS_CANDIDAT
        if (time_is_up or self.total_turns >= self.MAX_TURNS)                 and self.current_step_index < questions_index:
            self.current_step_index = questions_index
            self._phase_start_time.setdefault(self.steps[self.current_step_index], time.time())

        # Priorite 2 : budget temps OU nb questions depasse → phase suivante
        elif (phase_overtime or q_done >= max_q)                 and self.current_step_index < len(self.steps) - 1                 and current_step not in ("QUESTIONS_CANDIDAT", "CONCLUSION"):
            _advance()

        # Priorite 3 : LLM decide que la phase est couverte
        elif meta.get("next_step") and self.current_step_index < len(self.steps) - 1:
            _advance()

        self.chat_history.append({"role": "Candidat",  "text": user_input})
        self.chat_history.append({"role": "Recruteur", "text": speech_text})

    def generate_report(self) -> dict:
        elapsed = round((time.time() - self.start_time) / 60, 1) if self.start_time else 0
        final   = self.get_final_score()

        comm_scores = self.dimension_scores["communication"]
        tech_scores = self.dimension_scores["technique"]
        moti_scores = self.dimension_scores["motivation"]

        report = {
            "date":             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "langue":           self.target_lang,
            "duree_minutes":    elapsed,
            "nb_echanges":      self.total_turns,
            "scores_par_phase": final["scores_par_phase"],
            "score_final":      final["score_final"],
            "score_communication": round(sum(comm_scores)/len(comm_scores), 1) if comm_scores else 0,
            "score_technique":     round(sum(tech_scores)/len(tech_scores), 1) if tech_scores else 0,
            "score_motivation":    round(sum(moti_scores)/len(moti_scores), 1) if moti_scores else 0,
            "documents_utilises": {
                "cv":           self.ingested_docs.get("cv",           []),
                "offre":        self.ingested_docs.get("job_offer",    []),
                "company_info": self.ingested_docs.get("company_info", []),
            },
        }

        try:
            os.makedirs("data/reports", exist_ok=True)
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/reports/interview_report_{ts}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"Rapport sauvegarde : {path}")
        except Exception as e:
            print(f"Erreur sauvegarde rapport : {e}")

        return report

    # ─────────────────────────────────────────
    # MODE HTTP
    # ─────────────────────────────────────────

    def generate_response(self, user_input: str) -> dict:
        if self.conclusion_done:
            return {
                "candidate_sentiment": "neutre",
                "text":                "",
                "next_step":           False,
                "score":               None,
                "interview_ended":     True,
            }

        current_step = self.steps[self.current_step_index]

        if current_step == "QUESTIONS_CANDIDAT" \
                and self._candidate_has_no_questions(user_input):
            conclusion_idx = self.steps.index("CONCLUSION")
            self.current_step_index = conclusion_idx
            current_step = "CONCLUSION"

        short_answer = self._is_too_short(user_input) \
                       and current_step not in ("QUESTIONS_CANDIDAT", "CONCLUSION", "INTRO_ETUDES")

        context = self._get_rag_context(user_input)
        history = self._get_recent_history()

        speech_text = DEFAULT_RESPONSE["text"]
        try:
            r = self.llm_client.generate(
                model  ="qwen2.5",
                system =self._build_speech_system(current_step, history, context, short_answer),
                prompt =user_input,
            )
            speech_text = self._strip_greeting(r.response.strip())
        except Exception as e:
            print("LLM SPEECH ERROR:", e)

        import re as _re
        questions = _re.findall(r"[^.!?]*\?", speech_text)
        if questions:
            self.last_question = questions[-1].strip()

        meta = self._parse_meta("")
        try:
            r2 = self.llm_client.generate(
                model  ="qwen2.5",
                system =self._build_meta_system(current_step, user_input, speech_text),
                prompt =user_input,
                format ="json",
            )
            meta = self._parse_meta(r2.response)
        except Exception as e:
            print("LLM META ERROR:", e)

        self._update_state(user_input, speech_text, meta)

        # interview_ended : le LLM peut suggerer, mais la decision finale
        # est toujours celle de _update_state (conclusion_done) ou _is_farewell_speech.
        # On ne laisse JAMAIS le LLM seul decider — trop peu fiable.
        interview_ended = self.conclusion_done or self._is_farewell_speech(speech_text)

        return {
            "candidate_sentiment": meta["candidate_sentiment"],
            "text":                speech_text,
            "next_step":           meta["next_step"],
            "score":               meta["score"],
            "interview_ended":     interview_ended,
        }

    # ─────────────────────────────────────────
    # MODE WEBSOCKET
    # ─────────────────────────────────────────

    async def generate_response_stream(self, user_input: str) -> AsyncGenerator[dict, None]:
        import asyncio
        import re

        if self.conclusion_done:
            yield {"type": "stream_done", "full_text": ""}
            yield {
                "type": "meta", "full_text": "", "candidate_sentiment": "neutre",
                "next_step": False, "score": None, "phase": "CONCLUSION",
                "time_left": 0, "interview_ended": True,
            }
            yield {"type": "done"}
            return

        current_step = self.steps[self.current_step_index]

        if current_step == "QUESTIONS_CANDIDAT" \
                and self._candidate_has_no_questions(user_input):
            conclusion_idx = self.steps.index("CONCLUSION")
            self.current_step_index = conclusion_idx
            current_step = "CONCLUSION"

        short_answer = self._is_too_short(user_input) \
                       and current_step not in ("QUESTIONS_CANDIDAT", "CONCLUSION", "INTRO_ETUDES")

        history = self._get_recent_history()

        # ── RAG lancé en tâche parallèle — ne bloque plus le démarrage LLM ──
        rag_task = asyncio.create_task(self._get_rag_context_async(user_input))
        yield {"type": "thinking"}
        context = await rag_task   # attend la fin du RAG avant d'envoyer le prompt LLM

        # ── Regex de coupure de phrase : . ! ? : suivi d'espace ou fin ──
        # On n'émet une phrase que si elle fait au moins MIN_SENTENCE_CHARS caractères
        # pour éviter d'envoyer des fragments d'une seule lettre au TTS.
        SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
        MIN_SENTENCE_CHARS = 30

        speech_text  = ""
        sentence_buf = ""
        sentence_idx = 0
        # meta_task est lancé dès que le texte complet est disponible (stream fini)
        meta_task    = None

        try:
            async for chunk in await self.llm_async_client.generate(
                model  ="qwen2.5",
                system =self._build_speech_system(current_step, history, context, short_answer),
                prompt =user_input,
                stream =True,
            ):
                token = chunk.response
                if not token:
                    continue

                speech_text  += token
                sentence_buf += token

                yield {"type": "token", "token": token}

                # ── Découpe en phrases dès la ponctuation ──────────────────
                # Émet "sentence" pour que main.py lance le TTS sans attendre
                # la fin du stream complet (pipeline TTS par phrase).
                parts = SENTENCE_SPLIT.split(sentence_buf)
                while len(parts) > 1:
                    candidate_sentence = parts.pop(0).strip()
                    if len(candidate_sentence) >= MIN_SENTENCE_CHARS:
                        yield {
                            "type":     "sentence",
                            "text":     candidate_sentence,
                            "index":    sentence_idx,
                        }
                        sentence_idx += 1
                    sentence_buf = parts[0] if parts else ""

        except Exception as e:
            print("LLM STREAM ERROR:", e)
            speech_text  = DEFAULT_RESPONSE["text"]
            sentence_buf = speech_text
            yield {"type": "token", "token": speech_text}

        speech_text = self._strip_greeting(speech_text.strip()) or DEFAULT_RESPONSE["text"]

        # ── Dernier fragment (pas terminé par ponctuation) ─────────────────
        last_fragment = sentence_buf.strip()
        if last_fragment and len(last_fragment) >= 5:
            yield {
                "type":  "sentence",
                "text":  last_fragment,
                "index": sentence_idx,
            }

        _questions = re.findall(r"[^.!?]*[?]", speech_text)
        if _questions:
            self.last_question = _questions[-1].strip()

        # ── Meta LLM lancé en parallèle dès la fin du stream ──────────────
        loop = asyncio.get_event_loop()
        meta_task = loop.run_in_executor(
            None,
            lambda: self.llm_client.generate(
                model  ="qwen2.5",
                system =self._build_meta_system(current_step, user_input, speech_text),
                prompt =user_input,
                format ="json",
            )
        )

        yield {"type": "stream_done", "full_text": speech_text}

        meta = self._parse_meta("")
        try:
            r2   = await meta_task
            meta = self._parse_meta(r2.response)
        except Exception as e:
            print("LLM META ERROR:", e)

        self._update_state(user_input, speech_text, meta)

        # Forcer interview_ended via la detection farewell et conclusion_done —
        # ne pas laisser le LLM decider seul, trop peu fiable.
        interview_ended = self.conclusion_done or self._is_farewell_speech(speech_text)

        yield {
            "type":                "meta",
            "full_text":           speech_text,
            "candidate_sentiment": meta["candidate_sentiment"],
            "next_step":           meta["next_step"],
            "score":               meta["score"],
            "phase":               self.steps[self.current_step_index],
            "time_left":           self.get_time_remaining(),
            "interview_ended":     interview_ended,
        }

        yield {"type": "done"}