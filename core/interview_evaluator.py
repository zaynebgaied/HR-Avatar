"""
interview_evaluator.py  —  v6.0  (Rigorous Answer Evaluation Engine)
=====================================================================

NOUVEAUTÉS v6.0 — Évaluation rigoureuse réponse par réponse
─────────────────────────────────────────────────────────────

  ► RÉPONSE STATUS (par tour) :
      • ANSWERED_FULLY    — Candidat a répondu complètement et correctement
      • ANSWERED_PARTIALLY — Répondu mais incomplet (manque de détails, hors sujet partiel,
                             réponse trop courte ou non concluante)
      • ANSWERED_INCORRECTLY — Répondu mais techniquement ou factuellement erroné
      • EVADED             — Sujet évité, question détournée sans y répondre
      • NOT_ANSWERED       — Silence, "je ne sais pas" explicite, ou refus total
      • OFF_TOPIC          — Réponse sans rapport avec la question posée

  ► CORRECTNESS ANALYZER (nouveau module) :
      • Détecte les affirmations techniques incorrectes ou douteuses
      • Distingue vagueness (style) vs incorrectness (fond)
      • Détecte l'évasion : réponses qui semblent répondre sans répondre

  ► COMPLETENESS SCORER (nouveau module) :
      • Vérifie si TOUTES les sous-parties d'une question sont adressées
      • Détecte les questions composées (multi-part) et mesure le taux de couverture

  ► EVASION DETECTOR (nouveau module) :
      • Détecte les pivots ("c'est une bonne question…"), généralisations,
        redirections vers l'équipe, non-réponses polies

  ► ANSWER QUALITY MATRIX (nouvelle dimension) :
      • Pour chaque Q→R : grille status × correctness × completeness × depth
      • Pénalités automatiques si : NOT_ANSWERED, EVADED, ANSWERED_INCORRECTLY

  ► 15 DIMENSIONS (vs 12 en v5) :
      • 3 nouvelles : answer_status_score, correctness_score, completeness_score

  ► RAPPORT ENRICHI :
      • Section "answer_by_answer" : analyse Q→R individuelle avec verdict
      • Tableau récapitulatif par phase : combien répondus/partiels/éludés/pas répondus
      • Pénalité de scoring explicite pour les non-réponses et évasions

  ► EMOTION INTEGRATION (v6.1) :
      • EmotionAnalyzer : exploite la timeline VisionEngine (DeepFace)
      • Calcule : émotion dominante, taux de stress, évolution par phase,
        pics de stress horodatés, corrélation émotion ↔ phase d'entretien
      • Le LLM reçoit le contexte émotionnel complet dans son prompt de synthèse
      • Rapport enrichi : section "emotion_analysis" avec interprétation recruteur

Architecture :
  1.  LogParser           — parse le format exact de llm_chain (inchangé)
  2.  SignalAnalyzer       — heuristiques regex (enrichi)
  3.  AnswerStatusDetector — détecte si répondu / pas répondu / éludé / partiel
  4.  CompletenessAnalyzer — mesure si toutes les sous-questions sont couvertes
  5.  TurnScorer           — 15 dimensions via LLM (enrichi)
  6.  PhaseAggregator      — agrège avec statistiques réponse/non-réponse
  7.  EmotionAnalyzer      — NOUVEAU v6.1 : analyse timeline VisionEngine
  8.  SynthesisEngine      — rapport final enrichi avec émotion + réponse par réponse
  9.  InterviewEvaluator   — orchestrateur principal
"""

from __future__ import annotations

import os
import re
import json
import glob
import datetime
import sys
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from ollama import Client

# =============================================================================
# CONFIG — Aligné sur llm_chain.py V3
# =============================================================================

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
INTERVIEW_FOLDER = os.path.join(BASE_DIR, "data")
REPORT_FOLDER    = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORT_FOLDER, exist_ok=True)

KNOWN_PHASES = [
    "OPENING",
    "JOB_ALIGNED_EXPLORATION",
    "PROJECT_DEEP_DIVE",
    "TECHNICAL_DEPTH",
    "SOFT_SKILLS_BEHAVIORAL",
    "CANDIDATE_QUESTIONS",
    "FINAL_CHECK",
    "CLOSING",
]

PHASE_LABELS = {
    "OPENING":                 "Opening & Candidate Introduction",
    "JOB_ALIGNED_EXPLORATION": "Job-Aligned Technical Exploration",
    "PROJECT_DEEP_DIVE":       "Project Deep Dive",
    "TECHNICAL_DEPTH":         "Technical Evaluation Depth",
    "SOFT_SKILLS_BEHAVIORAL":  "Soft Skills & Behavioral Assessment",
    "CANDIDATE_QUESTIONS":     "Candidate Questions",
    "FINAL_CHECK":             "Final Check",
    "CLOSING":                 "Closing",
}

PHASE_WEIGHTS = {
    "OPENING":                 5,
    "JOB_ALIGNED_EXPLORATION": 20,
    "PROJECT_DEEP_DIVE":       25,
    "TECHNICAL_DEPTH":         20,
    "SOFT_SKILLS_BEHAVIORAL":  20,
    "CANDIDATE_QUESTIONS":     10,
    "FINAL_CHECK":             0,
    "CLOSING":                 0,
}

SCORED_PHASES = [p for p, w in PHASE_WEIGHTS.items() if w > 0]
COVERAGE_THRESHOLD = 0.70

# ── Answer Status (nouveau — v6) ──────────────────────────────────────────────
ANSWER_STATUSES = [
    "ANSWERED_FULLY",
    "ANSWERED_PARTIALLY",
    "ANSWERED_INCORRECTLY",
    "EVADED",
    "NOT_ANSWERED",
    "OFF_TOPIC",
]

# Score de pénalité par statut (appliqué sur le score pondéré final)
ANSWER_STATUS_PENALTY = {
    "ANSWERED_FULLY":        0.0,    # pas de pénalité
    "ANSWERED_PARTIALLY":   -1.0,    # −1 point sur 10
    "ANSWERED_INCORRECTLY": -2.5,    # erreur factuelle/technique
    "EVADED":               -2.0,    # évasion
    "NOT_ANSWERED":         -3.5,    # silence / refus explicite
    "OFF_TOPIC":            -2.0,    # complètement hors sujet
}

# ── 15 Dimensions d'évaluation (12 de v5 + 3 nouvelles) ──────────────────────
DIMENSION_WEIGHTS = {
    # Dimensions core (réduites pour faire place aux 3 nouvelles)
    "relevance":           0.09,
    "technical_accuracy":  0.09,
    "experience_proof":    0.08,
    "depth":               0.06,
    "clarity":             0.05,
    "quantification":      0.05,
    "job_alignment":       0.05,
    "star_structure":      0.03,
    "vagueness_penalty":   0.03,
    "ownership":           0.05,
    "decision_quality":    0.04,
    "behavioral_quality":  0.02,
    # Nouvelles dimensions v6 (poids forts — coeur de l'évaluation rigoureuse)
    "answer_status_score": 0.13,   # 0=NOT_ANSWERED, 5=PARTIAL, 10=FULLY
    "correctness_score":   0.12,   # 0=incorrect, 5=incertain, 10=correct
    "completeness_score":  0.11,   # 0=incomplet, 5=partiel, 10=complet
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 0.001, "Poids dimensions != 1.0"

# ── Patterns heuristiques ─────────────────────────────────────────────────────

VAGUENESS_PATTERNS = [
    r"\b(i was (?:involved|part of|contributing|helping)|"
    r"i (?:helped|assisted|supported|contributed to)|"
    r"we (?:worked on|built|developed|did)|"
    r"(?:j'ai aidé|j'étais impliqué|on a fait|on a travaillé)|"
    r"(?:كنت جزءاً من|ساعدت في|شاركت في|عملنا على))\b",
    r"\b(kind of|sort of|more or less|somewhat|a bit|a little|"
    r"roughly|approximately|around|basically|essentially|"
    r"plutôt|un peu|à peu près|globalement|en gros|"
    r"نوعاً ما|تقريباً|بشكل عام|إلى حد ما)\b",
    r"\b(euh+|hmm+|uhh+|err+)\b",
    r"\b(dans ce genre|something like that|dans le genre)\b",
    r"\b(je pense que|i think that|peut-être|maybe|possibly)\b",
]
VAGUENESS_RE = re.compile("|".join(VAGUENESS_PATTERNS), re.IGNORECASE)

METRIC_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?[\s]*(?:%|ms|s\b|tb|gb|mb|fps|req|k\b|m\b)\b"
    r"|\b(?:accuracy|f1|latency|throughput|precision|recall|auc|rmse)\b"
    r"|\b\d+[\s]?(?:times|fold|reduction|improvement|faster|slower)\b"
    r"|\b\d+[\s]?(?:%|percent|x\b)",
    re.IGNORECASE
)

STAR_RE = re.compile(
    r"(situation|contexte|context|tâche|task|action|résultat|result|outcome"
    r"|when|quand|once|there was|il y avait|faced|لما|كانت|واجهت)",
    re.IGNORECASE
)

DECISION_RE = re.compile(
    r"(chose|selected|decided to use|opted for|went with|"
    r"j'ai choisi|on a opté|اخترنا|قررنا نستخدم)",
    re.IGNORECASE
)

OWNERSHIP_RE = re.compile(
    r"\b(i (?:built|designed|architected|owned|led|implemented|deployed|created|wrote)"
    r"|j'ai (?:construit|conçu|architecturé|dirigé|implémenté|déployé|créé)"
    r"|بنيت|صممت|قدت|نفذت|أنشأت)\b",
    re.IGNORECASE
)

# ── NOUVEAU : Patterns d'évasion et non-réponse ───────────────────────────────

NOT_ANSWERED_PATTERNS = re.compile(
    r"\b(i don'?t know|i'?m not sure|i haven'?t|i have no|no experience with|"
    r"je ne sais pas|je n'ai pas d'?expérience|jamais fait|"
    r"مو عارف|ما عندي|ما جربت|ما لقيت|لا أعرف|لم أعمل)\b",
    re.IGNORECASE
)

EVASION_PATTERNS = re.compile(
    r"\b(that'?s a good question|great question|interesting question|"
    r"it depends|it'?s complicated|hard to say|"
    r"c'?est une bonne question|ça dépend|c'?est complexe|difficile à dire|"
    r"هذا سؤال وجيه|يعتمد على|صعب أقول|معقد بعض الشيء)\b",
    re.IGNORECASE
)

REDIRECT_TO_TEAM_RE = re.compile(
    r"\b(the team (?:handled|decided|built|chose)|we all|everyone|"
    r"l'?équipe a (?:décidé|géré|construit)|on a tous|"
    r"الفريق قرر|الكل|كلنا اشتغلنا)\b",
    re.IGNORECASE
)

TOO_SHORT_THRESHOLD = 12  # mots — en dessous = réponse probablement incomplète

# ── Patterns de multi-questions (pour CompletenessAnalyzer) ──────────────────

MULTI_QUESTION_RE = re.compile(
    r"(?<=[.?])\s+(?:and\s+(?:also\s+)?|also\s+|additionally\s+|"
    r"et\s+(?:aussi\s+)?|aussi\s+|par ailleurs\s+|"
    r"وكذلك\s+|وأيضاً\s+|إضافةً\s+)",
    re.IGNORECASE
)

QUESTION_PARTS_RE = re.compile(
    r"\?(?:\s+(?:and|also|et|وكذلك|وأيضاً))?",
    re.IGNORECASE
)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ParsedTurn:
    timestamp: str
    phase: str
    speaker: str
    text: str
    answer_quality: str = "N/A"
    behavioral_story: bool = False
    decision_reasoning: bool = False
    weak_signals: List[str] = field(default_factory=list)
    is_candidate: bool = False
    is_recruiter: bool = False


@dataclass
class AnswerStatusResult:
    """Résultat du détecteur de statut de réponse."""
    status: str = "ANSWERED_FULLY"        # voir ANSWER_STATUSES
    confidence: float = 1.0               # 0.0–1.0
    not_answered_signals: List[str] = field(default_factory=list)
    evasion_signals: List[str] = field(default_factory=list)
    redirect_signals: List[str] = field(default_factory=list)
    too_short: bool = False
    word_count: int = 0
    explanation: str = ""


@dataclass
class CompletenessResult:
    """Résultat de l'analyse de complétude."""
    n_question_parts: int = 1              # combien de sous-questions détectées
    n_parts_addressed: int = 1             # combien semblent adressées
    completeness_ratio: float = 1.0        # n_parts_addressed / n_question_parts
    is_multi_part: bool = False
    missing_parts: List[str] = field(default_factory=list)
    completeness_score_raw: int = 10       # 0–10


@dataclass
class TurnScore:
    """Score 15-dimensionnel pour UN échange Question→Réponse."""
    question: str = ""
    answer: str = ""
    phase: str = ""

    # Métadonnées llm_chain
    llm_chain_quality: str = "N/A"
    llm_chain_behavioral: bool = False
    llm_chain_decision: bool = False
    llm_chain_weak_signals: List[str] = field(default_factory=list)

    # Statut de réponse (v6)
    answer_status: str = "ANSWERED_FULLY"
    answer_status_confidence: float = 1.0
    answer_status_explanation: str = ""
    completeness_n_parts: int = 1
    completeness_n_addressed: int = 1
    completeness_ratio: float = 1.0
    answer_verdict: str = ""               # synthèse humaine courte

    # 15 Dimensions (0–10)
    relevance: int = 5
    technical_accuracy: int = 5
    experience_proof: int = 5
    depth: int = 5
    clarity: int = 5
    quantification: int = 5
    job_alignment: int = 5
    star_structure: int = 0
    vagueness_penalty: int = 5
    ownership: int = 5
    decision_quality: int = 5
    behavioral_quality: int = 5
    answer_status_score: int = 5           # NOUVEAU
    correctness_score: int = 5             # NOUVEAU
    completeness_score: int = 5            # NOUVEAU

    # Diagnostics booléens
    is_off_topic: bool = False
    is_vague: bool = False
    has_real_example: bool = False
    has_metrics: bool = False
    star_detected: bool = False
    strong_ownership: bool = False
    decision_reasoning_detected: bool = False
    was_answered: bool = True              # NOUVEAU
    was_evaded: bool = False               # NOUVEAU
    was_incorrect: bool = False            # NOUVEAU
    was_partial: bool = False              # NOUVEAU

    # Score final
    weighted_score: float = 0.0
    status_penalty: float = 0.0           # pénalité appliquée pour non-réponse
    llm_justification: str = ""

    def compute_weighted(self) -> float:
        dims = {
            "relevance":           self.relevance,
            "technical_accuracy":  self.technical_accuracy,
            "experience_proof":    self.experience_proof,
            "depth":               self.depth,
            "clarity":             self.clarity,
            "quantification":      self.quantification,
            "job_alignment":       self.job_alignment,
            "star_structure":      self.star_structure,
            "vagueness_penalty":   self.vagueness_penalty,
            "ownership":           self.ownership,
            "decision_quality":    self.decision_quality,
            "behavioral_quality":  self.behavioral_quality,
            "answer_status_score": self.answer_status_score,
            "correctness_score":   self.correctness_score,
            "completeness_score":  self.completeness_score,
        }

        # Bonus llm_chain quality
        bonus = 0.0
        if self.llm_chain_quality == "STRONG":
            bonus += 0.5
        elif self.llm_chain_quality == "GOOD":
            bonus += 0.2
        elif self.llm_chain_quality in ("VAGUE", "INCOMPLETE"):
            bonus -= 0.3

        # Pénalité statut réponse
        self.status_penalty = ANSWER_STATUS_PENALTY.get(self.answer_status, 0.0)

        raw = sum(v * DIMENSION_WEIGHTS[k] for k, v in dims.items()) * 10.0
        self.weighted_score = round(
            min(10.0, max(0.0, raw + bonus + self.status_penalty)), 2
        )
        return self.weighted_score


@dataclass
class PhaseResult:
    phase: str = ""
    label: str = ""
    weight: int = 0
    turn_scores: List[TurnScore] = field(default_factory=list)

    avg_score: float = 0.0
    obtained_pts: int = 0
    status: str = "NOT_EVALUATED"

    # Diagnostics v5
    n_turns: int = 0
    n_vague: int = 0
    n_off_topic: int = 0
    n_with_metrics: int = 0
    n_with_example: int = 0
    n_star: int = 0
    n_strong_ownership: int = 0
    n_decision: int = 0
    comment: str = ""
    llm_chain_quality_dist: Dict[str, int] = field(default_factory=dict)
    behavioral_stories_count: int = 0
    decision_reasoning_count: int = 0
    weak_signal_types: List[str] = field(default_factory=list)

    # Nouvelles stats réponse v6
    n_answered_fully: int = 0
    n_answered_partially: int = 0
    n_answered_incorrectly: int = 0
    n_evaded: int = 0
    n_not_answered: int = 0
    n_off_topic_status: int = 0
    answer_coverage_rate: float = 0.0     # % questions répondues (fully+partially)
    answer_status_dist: Dict[str, int] = field(default_factory=dict)

    def aggregate(self):
        self.n_turns = len(self.turn_scores)
        if not self.turn_scores:
            self.avg_score = 0.0
            self.obtained_pts = 0
            self.status = "NOT_EVALUATED"
            return

        self.avg_score = round(
            sum(t.weighted_score for t in self.turn_scores) / self.n_turns, 2
        )
        self.obtained_pts = round((self.avg_score / 10.0) * self.weight)

        # Diagnostics v5
        self.n_vague            = sum(1 for t in self.turn_scores if t.is_vague)
        self.n_off_topic        = sum(1 for t in self.turn_scores if t.is_off_topic)
        self.n_with_metrics     = sum(1 for t in self.turn_scores if t.has_metrics)
        self.n_with_example     = sum(1 for t in self.turn_scores if t.has_real_example)
        self.n_star             = sum(1 for t in self.turn_scores if t.star_detected)
        self.n_strong_ownership = sum(1 for t in self.turn_scores if t.strong_ownership)
        self.n_decision         = sum(1 for t in self.turn_scores if t.decision_reasoning_detected)
        self.behavioral_stories_count = sum(1 for t in self.turn_scores if t.llm_chain_behavioral)
        self.decision_reasoning_count = sum(1 for t in self.turn_scores if t.llm_chain_decision)

        qd: Dict[str, int] = {}
        for t in self.turn_scores:
            q = t.llm_chain_quality
            qd[q] = qd.get(q, 0) + 1
        self.llm_chain_quality_dist = qd

        ws_all: List[str] = []
        for t in self.turn_scores:
            ws_all.extend(t.llm_chain_weak_signals)
        self.weak_signal_types = list(set(ws_all))

        # Stats réponse v6
        status_dist: Dict[str, int] = {s: 0 for s in ANSWER_STATUSES}
        for t in self.turn_scores:
            s = t.answer_status
            if s in status_dist:
                status_dist[s] += 1
        self.answer_status_dist       = status_dist
        self.n_answered_fully         = status_dist.get("ANSWERED_FULLY", 0)
        self.n_answered_partially     = status_dist.get("ANSWERED_PARTIALLY", 0)
        self.n_answered_incorrectly   = status_dist.get("ANSWERED_INCORRECTLY", 0)
        self.n_evaded                 = status_dist.get("EVADED", 0)
        self.n_not_answered           = status_dist.get("NOT_ANSWERED", 0)
        self.n_off_topic_status       = status_dist.get("OFF_TOPIC", 0)

        n_ok = self.n_answered_fully + self.n_answered_partially
        self.answer_coverage_rate = round(n_ok / self.n_turns, 2) if self.n_turns else 0.0

        # Status global de la phase
        vague_ratio = self.n_vague / self.n_turns if self.n_turns else 0
        no_ans_ratio = (self.n_not_answered + self.n_evaded) / self.n_turns if self.n_turns else 0

        if no_ans_ratio >= 0.5:
            self.status = "PARTIAL"
        elif vague_ratio >= 0.6:
            self.status = "PARTIAL"
        else:
            self.status = "EVALUATED"

        # Commentaire
        parts = []
        if self.n_answered_fully:
            parts.append(f"{self.n_answered_fully} fully answered ✅")
        if self.n_answered_partially:
            parts.append(f"{self.n_answered_partially} partially answered ⚠️")
        if self.n_answered_incorrectly:
            parts.append(f"{self.n_answered_incorrectly} answered incorrectly ❌")
        if self.n_evaded:
            parts.append(f"{self.n_evaded} evaded 🔄")
        if self.n_not_answered:
            parts.append(f"{self.n_not_answered} not answered 🚫")
        if self.n_off_topic:
            parts.append(f"{self.n_off_topic} off-topic 🚷")
        if self.n_with_example:
            parts.append(f"{self.n_with_example} with real example ✅")
        if self.n_with_metrics:
            parts.append(f"{self.n_with_metrics} with metrics 📊")
        if self.behavioral_stories_count:
            parts.append(f"{self.behavioral_stories_count} STAR story/stories ✅")
        if self.decision_reasoning_count:
            parts.append(f"{self.decision_reasoning_count} decision rationale(s) ✅")
        if self.n_strong_ownership:
            parts.append(f"{self.n_strong_ownership} strong ownership ✅")
        if self.weak_signal_types:
            parts.append(f"Weak signals: {', '.join(self.weak_signal_types)}")
        if not parts:
            parts.append("No candidate turns analyzed")
        self.comment = " | ".join(parts)


# =============================================================================
# 1. LOG PARSER
# =============================================================================

class LogParser:
    LOG_LINE_RE = re.compile(
        r"^\[(?P<timestamp>[^\]]+)\]"
        r"\s+\[(?P<phase>[A-Z_]+)\]"
        r"\s+(?P<speaker>[^\[]+?)"
        r"\s+\[quality=(?P<quality>[A-Z/]+)\]"
        r"\s+\[behavioral=(?P<behavioral>True|False)\]"
        r"\s+\[decision=(?P<decision>True|False)\]"
        r"\s+\[weak_signals=(?P<weak_signals>.*?)\]"  # .*? handles nested [] in list values e.g. ['vague']
        r":\s*(?P<text>.*)$",
        re.IGNORECASE
    )
    HEADER_RE   = re.compile(r"^(?:HR INTERVIEW|ENTRETIEN RH)\s*[—\-]\s*(?P<langue>\S+)", re.IGNORECASE)
    DATE_RE     = re.compile(r"^Date\s*:\s*(.+)$", re.IGNORECASE)
    DURATION_RE = re.compile(r"^Duration\s*:\s*(.+)$", re.IGNORECASE)

    def parse_file(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        return self.parse_text(raw)

    # ── Chat transcript format detection ─────────────────────────────────────
    # Matches lines like "12:55 · joy" or "13:02 · en cours…" (UI timestamps)
    CHAT_TIME_RE  = re.compile(r"^\d{1,2}:\d{2}(?:\s*·\s*.+)?$")
    # Matches the user icon line
    CHAT_USER_RE  = re.compile(r"^👤\s*$")
    # Matches the bot icon line
    CHAT_BOT_RE   = re.compile(r"^🤖\s*$")
    # A line that looks like a plain HH:MM timestamp (possibly with trailing label)
    CHAT_TS_RE    = re.compile(r"^(\d{1,2}:\d{2})")

    def _is_chat_format(self, raw: str) -> bool:
        """Detect if the raw text is a chat-UI transcript (not a structured log)."""
        has_user_icon = "👤" in raw
        has_bot_icon  = "🤖" in raw
        has_log_markers = "[quality=" in raw or "[behavioral=" in raw
        return (has_user_icon or has_bot_icon) and not has_log_markers

    def _assign_phase_from_content(self, text: str, turn_index: int, total_turns: int) -> str:
        """
        Heuristic phase assignment for chat transcripts that lack explicit phase tags.
        Uses turn position and keyword matching.
        """
        text_lower = text.lower()

        # Keyword-based phase detection (order matters — most specific first)
        technical_keywords = [
            "amdec", "rcm", "fmea", "cmms", "sap pm", "plc", "siemens", "tia portal",
            "vibration", "predictive", "preventive", "cmms", "mttr", "mtbf",
            "hydraulic", "conveyor", "failure mode", "root cause", "5 whys",
            "criticality", "functional analysis", "maintenance strategy",
        ]
        behavioral_keywords = [
            "conflict", "team", "manage", "leadership", "communicate",
            "challenge", "difficult", "pressure", "deadline", "collaboration",
            "stakeholder", "cross-functional", "hse", "safety",
        ]
        project_keywords = [
            "project", "implemented", "reduced", "improved", "initiative",
            "breakdown", "downtime", "reliability", "deployment", "result",
            "achieved", "percentage", "32%", "20%", "5s",
        ]

        # First ~15% of turns → OPENING
        if turn_index < max(2, int(total_turns * 0.15)):
            return "OPENING"

        # Last turn (candidate questions / closing)
        if turn_index >= total_turns - 1:
            return "CANDIDATE_QUESTIONS"

        # Keyword matching
        tech_score = sum(1 for kw in technical_keywords if kw in text_lower)
        behav_score = sum(1 for kw in behavioral_keywords if kw in text_lower)
        proj_score  = sum(1 for kw in project_keywords if kw in text_lower)

        if tech_score >= 3:
            return "TECHNICAL_DEPTH"
        if behav_score >= 2:
            return "SOFT_SKILLS_BEHAVIORAL"
        if proj_score >= 2:
            return "PROJECT_DEEP_DIVE"
        if tech_score >= 1:
            return "JOB_ALIGNED_EXPLORATION"

        # Position-based fallback
        ratio = turn_index / max(1, total_turns)
        if ratio < 0.20:
            return "OPENING"
        elif ratio < 0.45:
            return "JOB_ALIGNED_EXPLORATION"
        elif ratio < 0.70:
            return "PROJECT_DEEP_DIVE"
        elif ratio < 0.88:
            return "SOFT_SKILLS_BEHAVIORAL"
        else:
            return "CANDIDATE_QUESTIONS"

    def _parse_chat_transcript(self, raw: str) -> dict:
        """
        Parse a raw chat-UI transcript where speakers are identified by
        👤 (candidate) and 🤖 (recruiter/bot) icons.

        Expected rough format:
            <intro text>
            12:55 · joy
            👤
            <candidate text ...>
            12:56
            🤖
            <recruiter text ...>
            ...
        """
        meta = {"langue": "en", "date": "N/A", "duration": "N/A"}

        # Try to detect language and date from first lines
        lines = raw.splitlines()
        for line in lines[:10]:
            m = self.DATE_RE.match(line.strip())
            if m:
                meta["date"] = m.group(1).strip()
            m = self.DURATION_RE.match(line.strip())
            if m:
                meta["duration"] = m.group(1).strip()

        # --- Segment the transcript into speaker blocks ---
        # Walk line by line; when we see 👤 or 🤖 switch speaker
        blocks = []   # list of {"speaker": "candidate"|"recruiter", "text": str, "timestamp": str}
        current_speaker = None
        current_text_lines: List[str] = []
        current_ts = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Skip noise lines
            if stripped in ("🎤", "➤", "Timeout STT.", "en cours…") or stripped.startswith("➤"):
                continue

            # Timestamp line
            ts_m = self.CHAT_TS_RE.match(stripped)
            if ts_m and len(stripped) <= 20:
                current_ts = ts_m.group(1)
                continue

            # Speaker icon lines
            if "👤" in stripped and len(stripped) <= 4:
                if current_speaker is not None and current_text_lines:
                    blocks.append({
                        "speaker":    current_speaker,
                        "text":       " ".join(current_text_lines).strip(),
                        "timestamp":  current_ts,
                    })
                current_speaker = "candidate"
                current_text_lines = []
                continue

            if "🤖" in stripped and len(stripped) <= 4:
                if current_speaker is not None and current_text_lines:
                    blocks.append({
                        "speaker":    current_speaker,
                        "text":       " ".join(current_text_lines).strip(),
                        "timestamp":  current_ts,
                    })
                current_speaker = "recruiter"
                current_text_lines = []
                continue

            # Content line — append to current block
            if current_speaker is not None:
                current_text_lines.append(stripped)
            # else: preamble text before first speaker icon — ignore

        # flush last block
        if current_speaker is not None and current_text_lines:
            blocks.append({
                "speaker":   current_speaker,
                "text":      " ".join(current_text_lines).strip(),
                "timestamp": current_ts,
            })

        # Remove empty blocks
        blocks = [b for b in blocks if b["text"].strip()]

        # --- Build ParsedTurn objects with heuristic phase assignment ---
        turns: List[ParsedTurn] = []
        phases: Dict[str, List[ParsedTurn]] = {p: [] for p in KNOWN_PHASES}
        total_blocks = len(blocks)

        for idx, block in enumerate(blocks):
            is_candidate = (block["speaker"] == "candidate")
            is_recruiter = (block["speaker"] == "recruiter")
            text = block["text"]

            # Phase assigned using combined text of current + previous block for context
            context_text = text
            if idx > 0:
                context_text = blocks[idx - 1]["text"] + " " + text

            phase = self._assign_phase_from_content(context_text, idx, total_blocks)

            # Heuristic quality assessment for candidate turns
            word_count = len(text.split())
            vague_hits = VAGUENESS_RE.findall(text.lower())
            metric_hits = METRIC_RE.findall(text)

            if not is_candidate:
                quality = "N/A"
            elif word_count < 20 or len(vague_hits) >= 3:
                quality = "VAGUE"
            elif word_count > 80 and metric_hits:
                quality = "STRONG"
            elif word_count > 40:
                quality = "GOOD"
            else:
                quality = "INCOMPLETE"

            turn = ParsedTurn(
                timestamp=block["timestamp"],
                phase=phase,
                speaker="Candidate" if is_candidate else "Recruiter",
                text=text,
                answer_quality=quality,
                behavioral_story=bool(STAR_RE.findall(text.lower())),
                decision_reasoning=bool(DECISION_RE.findall(text.lower())),
                weak_signals=[],
                is_candidate=is_candidate,
                is_recruiter=is_recruiter,
            )
            turns.append(turn)
            if phase in phases:
                phases[phase].append(turn)

        # Detect language from candidate text
        all_cand = " ".join(b["text"] for b in blocks if b["speaker"] == "candidate").lower()
        if any(w in all_cand for w in ["je ", "j'ai", "nous avons", "mon ", "ma "]):
            meta["langue"] = "French"
        elif any(w in all_cand for w in ["my ", "i have", "i've", "i am", "we have"]):
            meta["langue"] = "English"

        print(f"   [ChatParser] {len(blocks)} blocks parsed → {len(turns)} turns across phases: "
              + ", ".join(f"{p}({len(phases[p])})" for p in KNOWN_PHASES if phases[p]))

        return {
            "meta": meta,
            "turns": turns,
            "phases": phases,
            "all_candidate_texts": [t.text for t in turns if t.is_candidate],
        }

    def parse_text(self, raw: str) -> dict:
        # ── Detect format and route to appropriate parser ─────────────────────
        if self._is_chat_format(raw):
            print("   [LogParser] Chat-UI transcript detected → using ChatTranscriptParser")
            return self._parse_chat_transcript(raw)

        # ── Original structured log parser ────────────────────────────────────
        meta = {"langue": "N/A", "date": "N/A", "duration": "N/A"}
        turns: List[ParsedTurn] = []
        phases: Dict[str, List[ParsedTurn]] = {p: [] for p in KNOWN_PHASES}

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            m = self.HEADER_RE.match(line)
            if m:
                meta["langue"] = m.group("langue"); continue
            m = self.DATE_RE.match(line)
            if m:
                meta["date"] = m.group(1).strip(); continue
            m = self.DURATION_RE.match(line)
            if m:
                meta["duration"] = m.group(1).strip(); continue

            m = self.LOG_LINE_RE.match(line)
            if not m:
                if turns:
                    turns[-1].text += " " + line
                continue

            phase   = m.group("phase").strip()
            speaker = m.group("speaker").strip()

            ws_raw = m.group("weak_signals").strip()
            weak_signals: List[str] = []
            if ws_raw and ws_raw != "[]":
                ws_raw = ws_raw.strip("[]'\" ")
                weak_signals = [w.strip(" '\"") for w in ws_raw.split(",") if w.strip(" '\"")]

            turn = ParsedTurn(
                timestamp=m.group("timestamp").strip(),
                phase=phase if phase in KNOWN_PHASES else "OPENING",
                speaker=speaker,
                text=m.group("text").strip(),
                answer_quality=m.group("quality").strip(),
                behavioral_story=m.group("behavioral").lower() == "true",
                decision_reasoning=m.group("decision").lower() == "true",
                weak_signals=weak_signals,
                is_candidate=("Candidate" in speaker or "Candidat" in speaker),
                is_recruiter=("Recruiter" in speaker or "Recruteur" in speaker
                              or "Avatar" in speaker or "Interviewer" in speaker),
            )
            turns.append(turn)
            if turn.phase in phases:
                phases[turn.phase].append(turn)

        return {
            "meta": meta,
            "turns": turns,
            "phases": phases,
            "all_candidate_texts": [t.text for t in turns if t.is_candidate],
        }


# =============================================================================
# 2. SIGNAL ANALYZER
# =============================================================================

class SignalAnalyzer:
    def analyze(self, question: str, answer: str, candidate_turn: ParsedTurn) -> dict:
        vague_hits    = VAGUENESS_RE.findall(answer.lower())
        metric_hits   = METRIC_RE.findall(answer)
        star_hits     = STAR_RE.findall(answer.lower())
        decision_hits = DECISION_RE.findall(answer.lower())
        ownership_hits= OWNERSHIP_RE.findall(answer.lower())

        is_vague          = len(vague_hits) >= 2 or candidate_turn.answer_quality in ("VAGUE", "INCOMPLETE")
        has_metrics       = bool(metric_hits) or candidate_turn.answer_quality == "STRONG"
        has_real_example  = candidate_turn.behavioral_story or bool(ownership_hits)
        star_heuristic    = len(star_hits) >= 2 and candidate_turn.behavioral_story
        decision_heuristic= bool(decision_hits) or candidate_turn.decision_reasoning
        strong_ownership  = bool(ownership_hits) and not any("we " in h.lower() for h in ownership_hits)

        return {
            "vague_hits":           vague_hits,
            "metric_hits":          metric_hits,
            "is_vague":             is_vague,
            "has_metrics":          has_metrics,
            "has_real_example":     has_real_example,
            "star_heuristic":       star_heuristic,
            "decision_heuristic":   decision_heuristic,
            "strong_ownership":     strong_ownership,
            "llm_chain_quality":    candidate_turn.answer_quality,
            "llm_chain_behavioral": candidate_turn.behavioral_story,
            "llm_chain_decision":   candidate_turn.decision_reasoning,
            "llm_chain_weak_signals": candidate_turn.weak_signals,
        }


# =============================================================================
# 3. ANSWER STATUS DETECTOR (NOUVEAU v6)
# =============================================================================

class AnswerStatusDetector:
    """
    Détermine si le candidat a répondu à la question, et de quelle façon.

    Statuts possibles :
      ANSWERED_FULLY       — réponse complète et apparemment correcte
      ANSWERED_PARTIALLY   — réponse mais incomplète / trop courte / lacunes
      ANSWERED_INCORRECTLY — réponse avec erreurs factuelles ou techniques détectées
      EVADED               — sujet évité (pivot, généralisation, redirection)
      NOT_ANSWERED         — "je ne sais pas", silence, refus explicite
      OFF_TOPIC            — réponse sans rapport avec la question

    Note : ANSWERED_INCORRECTLY et les erreurs techniques fines sont confirmées
    par le LLM dans TurnScorer. Ce module fait uniquement une pré-détection heuristique.
    """

    def detect(self, question: str, answer: str, candidate_turn: ParsedTurn) -> AnswerStatusResult:
        result = AnswerStatusResult()
        words  = answer.split()
        result.word_count = len(words)
        result.too_short  = result.word_count < TOO_SHORT_THRESHOLD

        # ── 1. NOT_ANSWERED : "je ne sais pas" explicite ─────────────────────
        not_ans_hits = NOT_ANSWERED_PATTERNS.findall(answer)
        if not_ans_hits:
            result.not_answered_signals = [h for h in not_ans_hits if h]
            # Si court ET not_answered_pattern → clairement pas répondu
            if result.word_count < 30 or len(not_ans_hits) >= 2:
                result.status      = "NOT_ANSWERED"
                result.confidence  = 0.9
                result.explanation = f"Explicit non-answer signals: {not_ans_hits[:3]}"
                return result

        # ── 2. EVADED : pivot ou redirection ─────────────────────────────────
        evasion_hits  = EVASION_PATTERNS.findall(answer)
        redirect_hits = REDIRECT_TO_TEAM_RE.findall(answer)
        result.evasion_signals  = [h for h in evasion_hits if h]
        result.redirect_signals = [h for h in redirect_hits if h]

        # Évasion forte : pivot + pas de contenu concret
        metric_hits   = METRIC_RE.findall(answer)
        ownership_hits= OWNERSHIP_RE.findall(answer)
        has_substance = bool(metric_hits) or bool(ownership_hits) or result.word_count >= 80

        if evasion_hits and not has_substance:
            result.status     = "EVADED"
            result.confidence = 0.8
            result.explanation = (
                f"Evasion pivot detected ({evasion_hits[0]!r}) with no concrete substance"
            )
            return result

        # Redirection systématique vers l'équipe sans ownership personnel
        if len(redirect_hits) >= 2 and not ownership_hits:
            result.status     = "EVADED"
            result.confidence = 0.75
            result.explanation = "Systematic team redirection, no personal ownership found"
            return result

        # ── 3. OFF_TOPIC : pas de lien avec la question ───────────────────────
        # Heuristique légère : si la réponse ne contient aucun mot-clé de la question
        question_keywords = set(re.findall(r"\b\w{4,}\b", question.lower())) - {
            "what", "when", "where", "which", "that", "this", "your", "have",
            "vous", "votre", "quel", "quelle", "pour", "dans", "comment",
            "وش", "كيف", "ليش", "عندك", "هذا",
        }
        answer_lower = answer.lower()
        matched_kw = sum(1 for kw in question_keywords if kw in answer_lower)

        if question_keywords and matched_kw == 0 and result.word_count > 20:
            result.status     = "OFF_TOPIC"
            result.confidence = 0.7
            result.explanation = "Answer shares no keywords with the question"
            return result

        # ── 4. llm_chain quality override ────────────────────────────────────
        if candidate_turn.answer_quality in ("VAGUE", "INCOMPLETE"):
            if result.too_short or len(VAGUENESS_RE.findall(answer)) >= 3:
                result.status     = "ANSWERED_PARTIALLY"
                result.confidence = 0.8
                result.explanation = (
                    f"llm_chain quality={candidate_turn.answer_quality}, "
                    f"word_count={result.word_count}, vague patterns detected"
                )
                return result

        # ── 5. ANSWERED_PARTIALLY : réponse trop courte ou llm_chain signal ──
        if result.too_short and not has_substance:
            result.status     = "ANSWERED_PARTIALLY"
            result.confidence = 0.75
            result.explanation = f"Answer too short ({result.word_count} words) with no concrete substance"
            return result

        # ── 6. ANSWERED_FULLY : tout le reste ────────────────────────────────
        result.status     = "ANSWERED_FULLY"
        result.confidence = 0.85
        result.explanation = "Answer appears complete and on-topic"
        return result


# =============================================================================
# 4. COMPLETENESS ANALYZER (NOUVEAU v6)
# =============================================================================

class CompletenessAnalyzer:
    """
    Analyse si le candidat a adressé TOUTES les sous-parties d'une question.

    Exemple : "Pourquoi cette technologie ET quel trade-off avez-vous accepté ?"
    → 2 parties. Si seule la première est adressée → ratio = 0.5
    """

    def analyze(self, question: str, answer: str) -> CompletenessResult:
        result = CompletenessResult()

        # Compter les sous-questions
        q_marks = len(re.findall(r"\?", question))
        result.n_question_parts = max(1, q_marks)
        result.is_multi_part    = result.n_question_parts > 1

        if not result.is_multi_part:
            result.n_parts_addressed     = 1
            result.completeness_ratio    = 1.0
            result.completeness_score_raw = 10
            return result

        # Pour les multi-questions : heuristique de couverture
        # On split la question sur ses "?" et on cherche des indices dans la réponse
        sub_questions = [sq.strip() for sq in re.split(r"\?", question) if sq.strip()]
        n_addressed = 0
        missing = []

        for sq in sub_questions:
            kw = set(re.findall(r"\b\w{4,}\b", sq.lower())) - {
                "what", "when", "where", "which", "that", "this", "your",
                "vous", "votre", "quel", "pour", "comment",
                "وش", "كيف", "ليش", "عندك",
            }
            if not kw:
                n_addressed += 1
                continue
            answer_lower = answer.lower()
            matched = sum(1 for k in kw if k in answer_lower)
            if matched >= max(1, len(kw) // 3):
                n_addressed += 1
            else:
                missing.append(sq[:80])

        result.n_parts_addressed  = n_addressed
        result.missing_parts      = missing
        result.completeness_ratio = round(n_addressed / len(sub_questions), 2) if sub_questions else 1.0
        result.completeness_score_raw = round(result.completeness_ratio * 10)
        return result


# =============================================================================
# 5. TURN SCORER — 15 dimensions via LLM (enrichi v6)
# =============================================================================

class TurnScorer:
    def __init__(self, llm_client: Client, model: str):
        self.client          = llm_client
        self.model           = model
        self.analyzer        = SignalAnalyzer()
        self.status_detector = AnswerStatusDetector()
        self.completeness    = CompletenessAnalyzer()

    def score_turn(
        self,
        question: str,
        answer: str,
        phase: str,
        candidate_turn: ParsedTurn,
        job_context: str = "",
    ) -> TurnScore:

        ts = TurnScore(
            question=question,
            answer=answer,
            phase=phase,
            llm_chain_quality=candidate_turn.answer_quality,
            llm_chain_behavioral=candidate_turn.behavioral_story,
            llm_chain_decision=candidate_turn.decision_reasoning,
            llm_chain_weak_signals=candidate_turn.weak_signals,
        )

        signals = self.analyzer.analyze(question, answer, candidate_turn)

        # Pre-fill heuristics
        ts.is_vague         = signals["is_vague"]
        ts.has_metrics      = signals["has_metrics"]
        ts.has_real_example = signals["has_real_example"]
        ts.star_detected    = signals["star_heuristic"]
        ts.decision_reasoning_detected = signals["decision_heuristic"]
        ts.strong_ownership = signals["strong_ownership"]

        # ── Détection statut réponse (pré-LLM) ───────────────────────────────
        status_result = self.status_detector.detect(question, answer, candidate_turn)
        ts.answer_status            = status_result.status
        ts.answer_status_confidence = status_result.confidence
        ts.answer_status_explanation = status_result.explanation

        # ── Analyse complétude ────────────────────────────────────────────────
        comp_result = self.completeness.analyze(question, answer)
        ts.completeness_n_parts    = comp_result.n_question_parts
        ts.completeness_n_addressed= comp_result.n_parts_addressed
        ts.completeness_ratio      = comp_result.completeness_ratio

        # Pré-initialisation des scores v6
        status_to_score = {
            "ANSWERED_FULLY":        10,
            "ANSWERED_PARTIALLY":     5,
            "ANSWERED_INCORRECTLY":   3,
            "EVADED":                 2,
            "NOT_ANSWERED":           0,
            "OFF_TOPIC":              2,
        }
        ts.answer_status_score  = status_to_score.get(ts.answer_status, 5)
        ts.completeness_score   = comp_result.completeness_score_raw
        ts.correctness_score    = 5  # neutre — LLM va affiner

        # Booleans dérivés du statut
        ts.was_answered  = ts.answer_status in ("ANSWERED_FULLY", "ANSWERED_PARTIALLY")
        ts.was_evaded    = ts.answer_status == "EVADED"
        ts.was_incorrect = ts.answer_status == "ANSWERED_INCORRECTLY"
        ts.was_partial   = ts.answer_status == "ANSWERED_PARTIALLY"
        ts.is_off_topic  = ts.answer_status == "OFF_TOPIC"

        # ── LLM scoring prompt (15 dimensions) ───────────────────────────────
        quality_hint = ""
        if candidate_turn.answer_quality != "N/A":
            quality_hint = (
                f"\n[llm_chain pre-classified: quality={candidate_turn.answer_quality}]"
                f"\n[behavioral_story={candidate_turn.behavioral_story}]"
                f"\n[decision_reasoning={candidate_turn.decision_reasoning}]"
                f"\n[weak_signals={candidate_turn.weak_signals or 'none'}]"
            )

        pre_status_hint = (
            f"\n[PRE-DETECTION answer_status={ts.answer_status} "
            f"(confidence={ts.answer_status_confidence:.2f}): {ts.answer_status_explanation}]"
            f"\n[completeness: {ts.completeness_n_addressed}/{ts.completeness_n_parts} "
            f"sub-questions addressed]"
        )

        prompt = f"""You are an expert senior HR evaluator. Score ONE candidate answer on 15 dimensions.

JOB CONTEXT:
{job_context if job_context else "(not provided)"}

INTERVIEW PHASE: {PHASE_LABELS.get(phase, phase)}

RECRUITER QUESTION:
{question}

CANDIDATE ANSWER:
{answer}
{quality_hint}
{pre_status_hint}

PRE-ANALYSIS SIGNALS:
- Vagueness patterns: {signals["vague_hits"][:5] or "none"}
- Metrics found: {signals["metric_hits"][:5] or "none"}
- Strong personal ownership: {signals["strong_ownership"]}
- STAR hints: {signals["star_heuristic"]}
- Decision reasoning: {signals["decision_heuristic"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RIGOROUS EVALUATION RULES (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FIRST determine: Did the candidate actually ANSWER the question?
   - If the answer is silence, "I don't know", or a clear refusal → NOT_ANSWERED
   - If the candidate pivoted or generalized without addressing the specific question → EVADED
   - If the candidate answered a DIFFERENT question → OFF_TOPIC
   - If partially answered (missing parts, too vague to be useful) → ANSWERED_PARTIALLY
   - If answered with factual/technical errors → ANSWERED_INCORRECTLY
   - If answered completely and correctly → ANSWERED_FULLY

2. CORRECTNESS: Evaluate whether technical/factual claims are correct.
   - Look for contradictions, incorrect terminology, impossible metrics, false best practices
   - Score 0 = clearly incorrect, 5 = uncertain/unverifiable, 10 = correct and precise

3. COMPLETENESS: If the question had multiple parts, did the candidate address all of them?
   - Score proportionally to the fraction of sub-questions addressed

4. DO NOT give generous scores to avoid conflict. If something is wrong, say so.
5. A high answer_status_score requires the answer to be both CORRECT and COMPLETE.
6. If answer_status is NOT_ANSWERED/EVADED, all content dimensions (depth, experience_proof,
   technical_accuracy, decision_quality) should be LOW (0-3), not inflated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSIONS (score 0–10 each):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT DIMENSIONS:
1.  relevance           — Does the answer directly address the question? (0=completely off-topic)
2.  technical_accuracy  — Are technical claims correct? (0=major errors/false claims, 10=expert accuracy)
3.  experience_proof    — Real personal experience cited? (0=pure theory, 10=specific project with details)
4.  depth               — Technical sophistication level (0=basic definition, 10=expert with trade-offs)
5.  clarity             — Structure and communication quality (0=confused, 10=crisp)
6.  quantification      — Concrete numbers, metrics, measurable outcomes (0=none, 10=rich quantification)
7.  job_alignment       — Alignment with role tools and competencies (0=misaligned, 10=perfect fit)

BEHAVIORAL DIMENSIONS:
8.  star_structure      — For behavioral Qs: Situation→Task→Action→Result (0=absent, 10=complete)
9.  ownership           — Personal accountability vs collective deflection (0=full deflection, 10=clear ownership)
10. decision_quality    — Trade-offs reasoning, alternatives considered (0=none, 10=rigorous)
11. behavioral_quality  — Quality of behavioral/soft-skill STAR demonstration

STYLE DIMENSION:
12. vagueness_penalty   — Absence of hedging language (0=extremely vague, 10=precise throughout)

NEW v6 DIMENSIONS (most important — be strict):
13. answer_status_score — Did candidate answer the question at all?
    0 = NOT_ANSWERED (silence, I don't know, refusal)
    2 = EVADED (pivot, generalization, redirect without answering)
    2 = OFF_TOPIC (answered a different question)
    3 = ANSWERED_INCORRECTLY (answered but with significant errors)
    5 = ANSWERED_PARTIALLY (answered but incomplete, key parts missing)
    10 = ANSWERED_FULLY (complete, correct, on-point answer)

14. correctness_score — Technical/factual correctness of the content
    0 = clearly incorrect (wrong facts, impossible claims, false statements)
    3 = mostly incorrect with some correct elements
    5 = uncertain/unverifiable or mixed accuracy
    7 = mostly correct with minor imprecisions
    10 = fully correct and precise

15. completeness_score — Coverage of all parts of the question
    0 = nothing addressed (0% of sub-questions)
    5 = half addressed (50%)
    10 = everything addressed (100%)

BOOLEAN DIAGNOSTICS:
- is_off_topic      : true if relevance ≤ 3 or answer_status = OFF_TOPIC
- is_vague          : true if vagueness_penalty ≤ 4
- has_real_example  : true if experience_proof ≥ 6
- has_metrics       : true if quantification ≥ 5
- star_detected     : true if star_structure ≥ 6
- strong_ownership  : true if ownership ≥ 7
- decision_reasoning_detected : true if decision_quality ≥ 6
- was_answered      : true if answer_status_score ≥ 5 (FULLY or PARTIALLY)
- was_evaded        : true if answer_status = EVADED
- was_incorrect     : true if answer_status = ANSWERED_INCORRECTLY or correctness_score ≤ 3
- was_partial       : true if answer_status = ANSWERED_PARTIALLY

ANSWER_STATUS choices: ANSWERED_FULLY | ANSWERED_PARTIALLY | ANSWERED_INCORRECTLY | EVADED | NOT_ANSWERED | OFF_TOPIC

Return ONLY this exact JSON (no text before/after):
{{
    "relevance":                   <int 0-10>,
    "technical_accuracy":          <int 0-10>,
    "experience_proof":            <int 0-10>,
    "depth":                       <int 0-10>,
    "clarity":                     <int 0-10>,
    "quantification":              <int 0-10>,
    "job_alignment":               <int 0-10>,
    "star_structure":              <int 0-10>,
    "vagueness_penalty":           <int 0-10>,
    "ownership":                   <int 0-10>,
    "decision_quality":            <int 0-10>,
    "behavioral_quality":          <int 0-10>,
    "answer_status_score":         <int 0-10>,
    "correctness_score":           <int 0-10>,
    "completeness_score":          <int 0-10>,
    "answer_status":               "<ANSWERED_FULLY|ANSWERED_PARTIALLY|ANSWERED_INCORRECTLY|EVADED|NOT_ANSWERED|OFF_TOPIC>",
    "answer_verdict":              "<1-sentence plain-English verdict on this specific answer>",
    "is_off_topic":                <bool>,
    "is_vague":                    <bool>,
    "has_real_example":            <bool>,
    "has_metrics":                 <bool>,
    "star_detected":               <bool>,
    "strong_ownership":            <bool>,
    "decision_reasoning_detected": <bool>,
    "was_answered":                <bool>,
    "was_evaded":                  <bool>,
    "was_incorrect":               <bool>,
    "was_partial":                 <bool>,
    "justification":               "<2-3 sentences: what was good, what was missing or wrong, why this status>"
}}"""

        try:
            resp = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.05, "num_predict": 700},
            )
            raw = getattr(resp, "response", None) or resp.get("response", "")
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)

            # Fill 15 dimensions
            for dim in DIMENSION_WEIGHTS:
                val = data.get(dim)
                if isinstance(val, (int, float)):
                    setattr(ts, dim, max(0, min(10, int(val))))

            # Fill booleans
            for flag in ("is_off_topic", "is_vague", "has_real_example",
                         "has_metrics", "star_detected", "strong_ownership",
                         "decision_reasoning_detected",
                         "was_answered", "was_evaded", "was_incorrect", "was_partial"):
                v = data.get(flag)
                if isinstance(v, bool):
                    setattr(ts, flag, v)

            # Override statut si LLM dit quelque chose de plus précis
            llm_status = data.get("answer_status", "")
            if llm_status in ANSWER_STATUSES:
                ts.answer_status = llm_status

            ts.answer_verdict    = str(data.get("answer_verdict", ""))
            ts.llm_justification = str(data.get("justification", ""))
            ts.is_off_topic      = ts.is_off_topic or ts.answer_status == "OFF_TOPIC"

        except Exception as exc:
            print(f"    ⚠️  TurnScorer LLM error ({type(exc).__name__}): {exc} — using heuristics")
            self._heuristic_fallback(ts, signals, candidate_turn, comp_result, status_result)

        # llm_chain quality overrides
        if candidate_turn.answer_quality == "STRONG" and ts.depth < 7:
            ts.depth = max(ts.depth, 7)
        elif candidate_turn.answer_quality in ("VAGUE", "INCOMPLETE") and ts.vagueness_penalty > 5:
            ts.vagueness_penalty = min(ts.vagueness_penalty, 4)

        # Cohérence : si NOT_ANSWERED/EVADED → plafonner les dimensions de contenu
        if ts.answer_status in ("NOT_ANSWERED", "EVADED"):
            ts.depth             = min(ts.depth, 3)
            ts.experience_proof  = min(ts.experience_proof, 2)
            ts.technical_accuracy= min(ts.technical_accuracy, 2)
            ts.decision_quality  = min(ts.decision_quality, 2)
            ts.correctness_score = min(ts.correctness_score, 2)
        elif ts.answer_status == "ANSWERED_INCORRECTLY":
            ts.technical_accuracy= min(ts.technical_accuracy, 3)
            ts.correctness_score = min(ts.correctness_score, 3)
            ts.depth             = min(ts.depth, 4)

        ts.compute_weighted()
        return ts

    def _heuristic_fallback(
        self,
        ts: TurnScore,
        signals: dict,
        candidate_turn: ParsedTurn,
        comp_result: CompletenessResult,
        status_result: AnswerStatusResult,
    ):
        """Valeurs par défaut heuristiques si le LLM échoue."""
        ts.relevance           = 3 if ts.is_off_topic else 6
        ts.technical_accuracy  = 4
        ts.experience_proof    = 7 if signals["has_real_example"] else 3
        ts.depth               = 5
        ts.clarity             = 5
        ts.quantification      = 7 if signals["has_metrics"] else 2
        ts.job_alignment       = 5
        ts.star_structure      = 7 if signals["star_heuristic"] else 1
        ts.vagueness_penalty   = 3 if signals["is_vague"] else 7
        ts.ownership           = 7 if signals["strong_ownership"] else 4
        ts.decision_quality    = 7 if signals["decision_heuristic"] else 3
        ts.behavioral_quality  = 6 if candidate_turn.behavioral_story else 3

        # Nouvelles dimensions
        status_to_score_h = {
            "ANSWERED_FULLY": 10, "ANSWERED_PARTIALLY": 5,
            "ANSWERED_INCORRECTLY": 3, "EVADED": 2,
            "NOT_ANSWERED": 0, "OFF_TOPIC": 2,
        }
        ts.answer_status_score = status_to_score_h.get(ts.answer_status, 5)
        ts.correctness_score   = 5
        ts.completeness_score  = comp_result.completeness_score_raw

        ts.llm_justification = "LLM unavailable — heuristic defaults applied."


# =============================================================================
# 6. PHASE AGGREGATOR
# =============================================================================

class PhaseAggregator:
    def __init__(self):
        self._turns: Dict[str, List[TurnScore]] = {p: [] for p in KNOWN_PHASES}

    def add(self, phase: str, ts: TurnScore):
        if phase in self._turns:
            self._turns[phase].append(ts)

    def get_phase_result(self, phase: str) -> PhaseResult:
        pr = PhaseResult(
            phase=phase,
            label=PHASE_LABELS.get(phase, phase),
            weight=PHASE_WEIGHTS.get(phase, 0),
            turn_scores=self._turns.get(phase, []),
        )
        pr.aggregate()
        return pr

    def get_all_results(self) -> Dict[str, PhaseResult]:
        return {p: self.get_phase_result(p) for p in SCORED_PHASES}

    def total_score(self) -> int:
        return min(100, sum(
            self.get_phase_result(p).obtained_pts for p in SCORED_PHASES
        ))

    def all_turn_scores(self) -> List[TurnScore]:
        result = []
        for p in KNOWN_PHASES:
            result.extend(self._turns.get(p, []))
        return result

    def quality_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {"STRONG": 0, "GOOD": 0, "INCOMPLETE": 0, "VAGUE": 0, "N/A": 0}
        for ts in self.all_turn_scores():
            q = ts.llm_chain_quality
            dist[q] = dist.get(q, 0) + 1 if q in dist else dist.setdefault("N/A", 0) + 1
        return dist

    def answer_status_distribution(self) -> Dict[str, int]:
        dist = {s: 0 for s in ANSWER_STATUSES}
        for ts in self.all_turn_scores():
            if ts.answer_status in dist:
                dist[ts.answer_status] += 1
        return dist


# =============================================================================
# 7. Q→R PAIR EXTRACTOR
# =============================================================================

class QAPairExtractor:
    def extract(self, turns: List[ParsedTurn]) -> List[dict]:
        pairs = []
        i = 0
        while i < len(turns):
            t = turns[i]
            if t.is_recruiter:
                q_parts = [t.text]
                j = i + 1
                while j < len(turns) and turns[j].is_recruiter:
                    q_parts.append(turns[j].text)
                    j += 1

                a_parts = []
                candidate_turn_ref: Optional[ParsedTurn] = None
                while j < len(turns) and turns[j].is_candidate:
                    a_parts.append(turns[j].text)
                    if candidate_turn_ref is None:
                        candidate_turn_ref = turns[j]
                    else:
                        if turns[j].behavioral_story:
                            candidate_turn_ref.behavioral_story = True
                        if turns[j].decision_reasoning:
                            candidate_turn_ref.decision_reasoning = True
                        candidate_turn_ref.weak_signals.extend(turns[j].weak_signals)
                        quality_order = ["STRONG", "GOOD", "INCOMPLETE", "VAGUE", "N/A"]
                        if (candidate_turn_ref.answer_quality == "N/A" or
                                (turns[j].answer_quality != "N/A" and
                                 quality_order.index(turns[j].answer_quality) <
                                 quality_order.index(candidate_turn_ref.answer_quality))):
                            candidate_turn_ref.answer_quality = turns[j].answer_quality
                    j += 1

                if a_parts and candidate_turn_ref is not None:
                    pairs.append({
                        "question":       " ".join(q_parts),
                        "answer":         " ".join(a_parts),
                        "candidate_turn": candidate_turn_ref,
                    })
                i = j
            else:
                i += 1
        return pairs


# =============================================================================
# 8. COVERAGE ANALYZER
# =============================================================================

class CoverageAnalyzer:
    def analyze(self, phases: Dict[str, List[ParsedTurn]]) -> dict:
        covered, uncovered = [], []
        for phase in SCORED_PHASES:
            turns = phases.get(phase, [])
            candidate_spoke = any(t.is_candidate for t in turns)
            (covered if candidate_spoke else uncovered).append(phase)

        rate = len(covered) / len(SCORED_PHASES) if SCORED_PHASES else 1.0
        return {
            "rate":             round(rate, 3),
            "rate_pct":         round(rate * 100),
            "covered_phases":   covered,
            "uncovered_phases": uncovered,
            "is_partial":       rate < COVERAGE_THRESHOLD,
        }


# =============================================================================
# 9. EMOTION ANALYZER — Exploite la timeline VisionEngine (v6.1)
# =============================================================================

# Mapping VisionEngine labels → catégories recruteur
_EMOTION_STRESS_SET   = {"tendu", "anxieux", "découragé"}
_EMOTION_POSITIVE_SET = {"détendu", "surpris"}
_EMOTION_NEUTRAL_SET  = {"neutre"}

# Labels traduits pour le rapport
_EMOTION_EN = {
    "tendu":     "tense/stressed",
    "anxieux":   "anxious",
    "découragé": "discouraged",
    "détendu":   "relaxed/confident",
    "surpris":   "surprised",
    "neutre":    "neutral",
}


class EmotionAnalyzer:
    """
    Analyse la timeline émotionnelle produite par VisionEngine.

    Reçoit le dict `analyse_emotion_vision` construit par main._build_vision_analysis()
    et produit un bloc structuré exploitable par le LLM de synthèse.

    Champs produits :
      - available           : bool — données disponibles ou non
      - dominant_emotion    : émotion la plus fréquente (label FR + EN)
      - stress_rate_pct     : % de frames dans un état de stress
      - positive_rate_pct   : % de frames dans un état positif/détendu
      - neutral_rate_pct    : % de frames neutres
      - stress_peaks        : liste de pics de stress horodatés avec phase
      - phase_emotion_map   : dict phase → émotion dominante dans cette phase
      - evolution_summary   : résumé textuel de l'évolution émotionnelle
      - n_frames            : nombre total de frames analysées
      - avg_confidence      : confiance moyenne de DeepFace
      - recruiter_flags     : liste de signaux recruteur interprétés
      - llm_context_block   : bloc texte prêt à être injecté dans le prompt LLM
    """

    def analyze(self, vision_data: Optional[dict]) -> dict:
        """
        Args:
            vision_data : dict produit par main._build_vision_analysis()
                          ou None / {"disponible": False} si pas de données.
        """
        empty = {
            "available":         False,
            "dominant_emotion":  "N/A",
            "stress_rate_pct":   0,
            "positive_rate_pct": 0,
            "neutral_rate_pct":  0,
            "stress_peaks":      [],
            "phase_emotion_map": {},
            "evolution_summary": "No vision data available.",
            "n_frames":          0,
            "avg_confidence":    0.0,
            "recruiter_flags":   [],
            "llm_context_block": "(No emotion data available — VisionEngine was not active or no frames were captured.)",
        }

        if not vision_data or not vision_data.get("disponible"):
            return empty

        timeline = vision_data.get("timeline", [])
        if not timeline:
            return {**empty, "available": True,
                    "llm_context_block": "(VisionEngine active but no frames were analyzed — candidate may have had camera off.)"}

        n = len(timeline)
        from collections import Counter

        # ── Distribution des émotions ─────────────────────────────────────────
        emotion_counts = Counter(e["emotion"] for e in timeline)
        dominant_fr    = emotion_counts.most_common(1)[0][0]
        dominant_en    = _EMOTION_EN.get(dominant_fr, dominant_fr)

        n_stress   = sum(1 for e in timeline if e["emotion"] in _EMOTION_STRESS_SET)
        n_positive = sum(1 for e in timeline if e["emotion"] in _EMOTION_POSITIVE_SET)
        n_neutral  = sum(1 for e in timeline if e["emotion"] in _EMOTION_NEUTRAL_SET)

        stress_rate   = round(n_stress   / n * 100)
        positive_rate = round(n_positive / n * 100)
        neutral_rate  = round(n_neutral  / n * 100)

        # ── Pics de stress (confiance ≥ 65%) ─────────────────────────────────
        stress_peaks = [
            {
                "timestamp": e["timestamp"],
                "phase":     e.get("phase", "UNKNOWN"),
                "emotion":   e["emotion"],
                "emotion_en":_EMOTION_EN.get(e["emotion"], e["emotion"]),
                "confidence":e.get("confidence", 0.0),
            }
            for e in timeline
            if e["emotion"] in _EMOTION_STRESS_SET and e.get("confidence", 0) >= 65.0
        ]

        # ── Émotion dominante par phase ───────────────────────────────────────
        phase_frames: Dict[str, List[str]] = {}
        for e in timeline:
            ph = e.get("phase", "UNKNOWN")
            phase_frames.setdefault(ph, []).append(e["emotion"])

        phase_emotion_map = {}
        for ph, emotions in phase_frames.items():
            ph_counts = Counter(emotions)
            ph_dom_fr = ph_counts.most_common(1)[0][0]
            phase_emotion_map[ph] = {
                "dominant_fr": ph_dom_fr,
                "dominant_en": _EMOTION_EN.get(ph_dom_fr, ph_dom_fr),
                "stress_pct":  round(sum(1 for em in emotions if em in _EMOTION_STRESS_SET) / len(emotions) * 100),
                "n_frames":    len(emotions),
            }

        # ── Évolution (transitions uniques) ──────────────────────────────────
        evolution = []
        prev = None
        for e in timeline:
            if e["emotion"] != prev:
                evolution.append(_EMOTION_EN.get(e["emotion"], e["emotion"]))
                prev = e["emotion"]
        evolution_summary = " → ".join(evolution) if evolution else "stable"

        # ── Confiance moyenne ─────────────────────────────────────────────────
        avg_conf = round(sum(e.get("confidence", 0) for e in timeline) / n, 1)

        # ── Signaux recruteur interprétés ─────────────────────────────────────
        recruiter_flags: List[str] = []

        if stress_rate >= 60:
            recruiter_flags.append(
                f"HIGH STRESS: {stress_rate}% of frames show stress/anxiety — candidate appeared significantly stressed throughout the interview"
            )
        elif stress_rate >= 35:
            recruiter_flags.append(
                f"MODERATE STRESS: {stress_rate}% stress rate — normal pre-interview anxiety, worth monitoring"
            )

        if positive_rate >= 50:
            recruiter_flags.append(
                f"POSITIVE/CONFIDENT: {positive_rate}% of frames show relaxed/confident state — candidate appeared at ease"
            )

        # Stress par phase spécifique
        for ph, ph_data in phase_emotion_map.items():
            if ph_data["stress_pct"] >= 70 and ph_data["n_frames"] >= 2:
                ph_label = PHASE_LABELS.get(ph, ph)
                recruiter_flags.append(
                    f"PHASE STRESS PEAK [{ph_label}]: {ph_data['stress_pct']}% stress in this phase — "
                    f"candidate showed visible discomfort when questioned on this topic"
                )

        # Stress montant vs descendant
        if len(timeline) >= 6:
            first_half  = timeline[:n // 2]
            second_half = timeline[n // 2:]
            stress_first  = sum(1 for e in first_half  if e["emotion"] in _EMOTION_STRESS_SET) / len(first_half)
            stress_second = sum(1 for e in second_half if e["emotion"] in _EMOTION_STRESS_SET) / len(second_half)
            if stress_second > stress_first + 0.25:
                recruiter_flags.append(
                    "INCREASING STRESS: emotion became more negative in the second half of the interview — "
                    "may indicate fatigue, difficult questions, or mounting pressure"
                )
            elif stress_first > stress_second + 0.25:
                recruiter_flags.append(
                    "STRESS RECOVERY: candidate started stressed but relaxed as interview progressed — "
                    "positive sign of adaptability and composure"
                )

        if not recruiter_flags:
            recruiter_flags.append("No significant emotional signals detected — candidate appeared stable throughout.")

        # ── Bloc texte pour le prompt LLM ─────────────────────────────────────
        phase_emo_lines = "\n".join(
            f"    [{PHASE_LABELS.get(ph, ph)[:35]}]: {d['dominant_en']} "
            f"({d['stress_pct']}% stress, {d['n_frames']} frames)"
            for ph, d in phase_emotion_map.items()
        )
        peaks_lines = "\n".join(
            f"    [{p['phase']} @ {p['timestamp']}]: {p['emotion_en']} ({p['confidence']:.0f}% confidence)"
            for p in stress_peaks[:8]
        ) or "    (none)"

        llm_block = f"""CANDIDATE EMOTION ANALYSIS (DeepFace real-time — {n} frames, avg confidence {avg_conf}%):
  Dominant emotion     : {dominant_en} ({dominant_fr})
  Stress rate          : {stress_rate}% of interview
  Positive/relaxed rate: {positive_rate}%
  Neutral rate         : {neutral_rate}%
  Emotional evolution  : {evolution_summary}

  Emotion by phase:
{phase_emo_lines}

  Stress peaks (confidence ≥ 65%):
{peaks_lines}

  Recruiter signals:
{chr(10).join("    ⚠️  " + f for f in recruiter_flags)}"""

        return {
            "available":         True,
            "dominant_emotion":  dominant_en,
            "dominant_fr":       dominant_fr,
            "stress_rate_pct":   stress_rate,
            "positive_rate_pct": positive_rate,
            "neutral_rate_pct":  neutral_rate,
            "stress_peaks":      stress_peaks,
            "phase_emotion_map": phase_emotion_map,
            "evolution_summary": evolution_summary,
            "n_frames":          n,
            "avg_confidence":    avg_conf,
            "recruiter_flags":   recruiter_flags,
            "llm_context_block": llm_block,
            # Données brutes pour le rapport JSON
            "emotion_distribution": dict(emotion_counts),
        }


# =============================================================================
# 10. SYNTHESIS ENGINE — Rapport final enrichi v6.1
# =============================================================================

class SynthesisEngine:
    def __init__(self, llm_client: Client, model: str):
        self.client          = llm_client
        self.model           = model
        self.emotion_analyzer = EmotionAnalyzer()

    def synthesize(
        self,
        parsed: dict,
        aggregator: PhaseAggregator,
        coverage: dict,
        job_context: str = "",
        vision_data: Optional[dict] = None,   # ← nouveau v6.1
    ) -> dict:

        # ── Analyse émotionnelle ──────────────────────────────────────────────
        emotion = self.emotion_analyzer.analyze(vision_data)

        results     = aggregator.get_all_results()
        score_total = aggregator.total_score()
        all_turns   = aggregator.all_turn_scores()
        qual_dist   = aggregator.quality_distribution()
        status_dist = aggregator.answer_status_distribution()

        # ── Answer-by-answer summary ──────────────────────────────────────────
        aba_lines = []
        for i, ts in enumerate(all_turns, 1):
            status_icon = {
                "ANSWERED_FULLY":        "✅",
                "ANSWERED_PARTIALLY":    "⚠️ ",
                "ANSWERED_INCORRECTLY":  "❌",
                "EVADED":                "🔄",
                "NOT_ANSWERED":          "🚫",
                "OFF_TOPIC":             "🚷",
            }.get(ts.answer_status, "❓")

            aba_lines.append(
                f"  Q{i:02d} [{PHASE_LABELS.get(ts.phase, ts.phase)[:30]}] "
                f"{status_icon} {ts.answer_status:<22} "
                f"score={ts.weighted_score:4.1f}/10 | "
                f"correct={ts.correctness_score}/10 | "
                f"complete={ts.completeness_score}/10 | "
                f"penalty={ts.status_penalty:+.1f}"
                + (f"\n       Q: {ts.question[:80]}…" if len(ts.question) > 80
                   else f"\n       Q: {ts.question}")
                + (f"\n       → {ts.answer_verdict}" if ts.answer_verdict else "")
                + (f"\n       ⚡ {ts.llm_justification[:120]}" if ts.llm_justification else "")
            )

        # ── Phase summary ─────────────────────────────────────────────────────
        phase_summary_lines = []
        phase_data_out = {}
        for ph, pr in results.items():
            line = (
                f"  [{pr.label}]: {pr.obtained_pts}/{pr.weight}pts "
                f"(avg={pr.avg_score}/10, n={pr.n_turns}, status={pr.status})"
            )
            if pr.comment:
                line += f"\n    → {pr.comment}"
            phase_summary_lines.append(line)
            phase_data_out[ph] = {
                "label":               pr.label,
                "weight":              pr.weight,
                "obtained_pts":        pr.obtained_pts,
                "avg_score":           pr.avg_score,
                "n_turns":             pr.n_turns,
                "status":              pr.status,
                "comment":             pr.comment,
                "n_vague":             pr.n_vague,
                "n_off_topic":         pr.n_off_topic,
                "n_with_metrics":      pr.n_with_metrics,
                "n_with_example":      pr.n_with_example,
                "n_star":              pr.n_star,
                "n_strong_ownership":  pr.n_strong_ownership,
                "n_decision":          pr.n_decision,
                "behavioral_stories":  pr.behavioral_stories_count,
                "decision_reasoning":  pr.decision_reasoning_count,
                "weak_signal_types":   pr.weak_signal_types,
                "quality_dist":        pr.llm_chain_quality_dist,
                # v6 stats
                "n_answered_fully":       pr.n_answered_fully,
                "n_answered_partially":   pr.n_answered_partially,
                "n_answered_incorrectly": pr.n_answered_incorrectly,
                "n_evaded":               pr.n_evaded,
                "n_not_answered":         pr.n_not_answered,
                "answer_coverage_rate":   pr.answer_coverage_rate,
                "answer_status_dist":     pr.answer_status_dist,
            }

        ts_summary = self._build_turns_summary(all_turns)

        uncovered_labels = [PHASE_LABELS.get(p, p) for p in coverage["uncovered_phases"]]
        cov_warning = ""
        if uncovered_labels:
            cov_warning = (
                f"\n⚠️ PHASES NOT EVALUATED: {', '.join(uncovered_labels)}"
                "\n→ These phases score 0. Distinguish 'not tested' from 'gap'.\n"
            )
        if coverage["is_partial"]:
            cov_warning += (
                f"\n⚠️ INSUFFICIENT COVERAGE: {coverage['rate_pct']}% "
                f"(threshold {round(COVERAGE_THRESHOLD*100)}%).\n"
            )

        score_bar = "\n".join(
            f"  {PHASE_LABELS[p]}: /{PHASE_WEIGHTS[p]}pts" for p in SCORED_PHASES
        )

        prompt = f"""You are an expert senior recruiter producing a final interview synthesis.
Numerical scores are LOCKED — do NOT modify them.
Your role: qualitative analysis, answer-by-answer review, emotion interpretation, verdict, and recommendations.

INTERVIEW META:
Language: {parsed['meta']['langue']} | Date: {parsed['meta']['date']} | Duration: {parsed['meta']['duration']}

JOB CONTEXT:
{job_context[:1500] if job_context else "(not provided)"}

SCORING SCALE:
{score_bar}
TOTAL SCORE (LOCKED): {score_total}/100
COVERAGE: {coverage['rate_pct']}%

ANSWER STATUS DISTRIBUTION (v6 — key signal):
{json.dumps(status_dist, indent=2, ensure_ascii=False)}

ANSWER-BY-ANSWER ANALYSIS:
{chr(10).join(aba_lines) if aba_lines else "(no turns)"}

PHASE SCORES (LOCKED):
{chr(10).join(phase_summary_lines)}

ANSWER QUALITY DISTRIBUTION (llm_chain):
{qual_dist}

{emotion["llm_context_block"]}
{cov_warning}

SYNTHESIS RULES:
1. Use pre-computed scores EXACTLY — never change numeric values.
2. competencies_detected: list ONLY competencies where experience_proof ≥ 6 and
   answer_status was ANSWERED_FULLY or ANSWERED_PARTIALLY with good content.
3. gaps_identified: distinguish:
   - "VERIFIED GAP (tested, insufficient answer)" — candidate answered badly or partially
   - "EVASION GAP (tested, evaded)" — candidate actively avoided the topic
   - "KNOWLEDGE GAP (tested, incorrect)" — candidate gave wrong answer
   - "NOT TESTED (phase not covered)" — was never asked
4. VERDICT RULES (strict):
   - "HIRE"         : score ≥ 65 AND coverage ≥ 70% AND no NOT_ANSWERED on key technical questions
   - "RECONSIDER"   : score 45–64 AND potential shown but clear gaps
   - "TO_COMPLETE"  : strong profile BUT coverage < 70% OR multiple EVADED answers on key questions
   - "NOT_RETAINED" : score < 45 OR multiple NOT_ANSWERED/INCORRECT on core competencies
5. Answer specific observations about patterns:
   - If candidate consistently evaded a topic → flag as behavioral signal
   - If candidate had correct partial answers → note what was missing specifically
   - If candidate gave incorrect answers → note the specific error
6. EMOTION INTERPRETATION RULES (use the emotion data above):
   - Correlate stress peaks with specific phases/questions — what topic triggered discomfort?
   - High stress during TECHNICAL_DEPTH but good answers → pressure resilience signal
   - High stress + poor answers → genuine knowledge gap confirmed by non-verbal cues
   - Stress then recovery → note adaptability
   - Consistently relaxed → confident candidate OR possibly not taking interview seriously
   - If no vision data: note "emotion data unavailable" in emotion_analysis
7. recruiter_self_assessment: evaluate interview quality, question depth, neglected topics.

Return ONLY this JSON (no text before/after):
{{
    "meta": {{
        "interview_date":  "{parsed['meta']['date']}",
        "language":        "{parsed['meta']['langue']}",
        "duration":        "{parsed['meta']['duration']}",
        "report_date":     "{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    }},
    "scores_by_phase": {{
        "OPENING":                 {{"obtained": {phase_data_out.get('OPENING',{{}}).get('obtained_pts',0)},           "max": 5,  "avg_turn": {phase_data_out.get('OPENING',{{}}).get('avg_score',0)},            "n_turns": {phase_data_out.get('OPENING',{{}}).get('n_turns',0)},            "status": "<str>", "comment": "<str>"}},
        "JOB_ALIGNED_EXPLORATION": {{"obtained": {phase_data_out.get('JOB_ALIGNED_EXPLORATION',{{}}).get('obtained_pts',0)}, "max": 20, "avg_turn": {phase_data_out.get('JOB_ALIGNED_EXPLORATION',{{}}).get('avg_score',0)}, "n_turns": {phase_data_out.get('JOB_ALIGNED_EXPLORATION',{{}}).get('n_turns',0)}, "status": "<str>", "comment": "<str>"}},
        "PROJECT_DEEP_DIVE":       {{"obtained": {phase_data_out.get('PROJECT_DEEP_DIVE',{{}}).get('obtained_pts',0)},       "max": 25, "avg_turn": {phase_data_out.get('PROJECT_DEEP_DIVE',{{}}).get('avg_score',0)},       "n_turns": {phase_data_out.get('PROJECT_DEEP_DIVE',{{}}).get('n_turns',0)},       "status": "<str>", "comment": "<str>"}},
        "TECHNICAL_DEPTH":         {{"obtained": {phase_data_out.get('TECHNICAL_DEPTH',{{}}).get('obtained_pts',0)},         "max": 20, "avg_turn": {phase_data_out.get('TECHNICAL_DEPTH',{{}}).get('avg_score',0)},         "n_turns": {phase_data_out.get('TECHNICAL_DEPTH',{{}}).get('n_turns',0)},         "status": "<str>", "comment": "<str>"}},
        "SOFT_SKILLS_BEHAVIORAL":  {{"obtained": {phase_data_out.get('SOFT_SKILLS_BEHAVIORAL',{{}}).get('obtained_pts',0)},  "max": 20, "avg_turn": {phase_data_out.get('SOFT_SKILLS_BEHAVIORAL',{{}}).get('avg_score',0)},  "n_turns": {phase_data_out.get('SOFT_SKILLS_BEHAVIORAL',{{}}).get('n_turns',0)},  "status": "<str>", "comment": "<str>"}},
        "CANDIDATE_QUESTIONS":     {{"obtained": {phase_data_out.get('CANDIDATE_QUESTIONS',{{}}).get('obtained_pts',0)},     "max": 10, "avg_turn": {phase_data_out.get('CANDIDATE_QUESTIONS',{{}}).get('avg_score',0)},     "n_turns": {phase_data_out.get('CANDIDATE_QUESTIONS',{{}}).get('n_turns',0)},     "status": "<str>", "comment": "<str>"}}
    }},
    "score_total":           {score_total},
    "score_technical":       <int 0-100>,
    "score_behavioral":      <int 0-100>,
    "score_communication":   <int 0-100>,
    "coverage_rate_pct":     {coverage['rate_pct']},
    "is_partial_evaluation": {"true" if coverage['is_partial'] else "false"},
    "answer_quality_distribution": {json.dumps(qual_dist, ensure_ascii=False)},
    "answer_status_distribution":  {json.dumps(status_dist, ensure_ascii=False)},
    "answer_response_rate_pct":    {round((status_dist.get("ANSWERED_FULLY",0) + status_dist.get("ANSWERED_PARTIALLY",0)) / max(1, sum(status_dist.values())) * 100)},
    "competencies_detected":       ["<competency + evidence>", "..."],
    "gaps_identified":             ["<gap — VERIFIED GAP|EVASION GAP|KNOWLEDGE GAP|NOT TESTED>", "..."],
    "answer_patterns": {{
        "evasion_topics":          ["<topic the candidate systematically evaded>", "..."],
        "incorrect_answers":       ["<Q + what was wrong>", "..."],
        "partial_answer_themes":   ["<theme + what was missing>", "..."],
        "strongest_answers":       ["<Q where candidate excelled>", "..."]
    }},
    "emotion_analysis": {{
        "available":              {"true" if emotion["available"] else "false"},
        "dominant_emotion":       "{emotion['dominant_emotion']}",
        "stress_rate_pct":        {emotion['stress_rate_pct']},
        "positive_rate_pct":      {emotion['positive_rate_pct']},
        "overall_emotional_state":"<str: 1-sentence summary — e.g. 'Candidate appeared anxious throughout'>",
        "stress_interpretation":  "<str: what does the stress level suggest about the candidate — competence gap, pressure sensitivity, or normal pre-interview nervousness?>",
        "phase_observations":     ["<str: phase + what emotion revealed — e.g. 'Candidate became visibly tense when probed on architecture decisions'>", "..."],
        "emotion_answer_correlation": "<str: did emotional state correlate with answer quality? e.g. 'Stress peaks aligned with evasive answers on technical questions'>",
        "recruiter_recommendation": "<str: specific advice based on emotion data — should the recruiter follow up? is there a topic that triggered unusual anxiety?>"
    }},
    "motivation_analysis":      "<str: integrate emotion data — did the candidate seem genuinely engaged or detached?>",
    "soft_skills_analysis":     "<str>",
    "decision_making_analysis": "<str>",
    "ownership_patterns":       "<str>",
    "strengths":                ["<str>", "..."],
    "improvement_areas":        ["<str>", "..."],
    "verdict":                  "HIRE | RECONSIDER | TO_COMPLETE | NOT_RETAINED",
    "recommendation_detail":    "<str: specific, actionable, referencing both answer patterns and emotion signals>",
    "red_flags":                ["<str>", "..."],
    "green_flags":              ["<str>", "..."],
    "turns_summary": {{
        "total_turns":               {ts_summary['total_turns']},
        "answered_fully_count":      {ts_summary['answered_fully_count']},
        "answered_partially_count":  {ts_summary['answered_partially_count']},
        "answered_incorrectly_count":{ts_summary['answered_incorrectly_count']},
        "evaded_count":              {ts_summary['evaded_count']},
        "not_answered_count":        {ts_summary['not_answered_count']},
        "off_topic_count":           {ts_summary['off_topic_count']},
        "response_rate_pct":         {ts_summary['response_rate_pct']},
        "avg_correctness":           {ts_summary['avg_correctness']},
        "avg_completeness":          {ts_summary['avg_completeness']},
        "avg_relevance":             {ts_summary['avg_relevance']},
        "avg_depth":                 {ts_summary['avg_depth']},
        "avg_experience":            {ts_summary['avg_experience']},
        "avg_ownership":             {ts_summary['avg_ownership']},
        "avg_decision":              {ts_summary['avg_decision']},
        "vague_count":               {ts_summary['vague_count']},
        "with_metrics":              {ts_summary['with_metrics']},
        "star_count":                {ts_summary['star_count']},
        "strong_ownership_count":    {ts_summary['strong_ownership_count']},
        "decision_reasoning_count":  {ts_summary['decision_reasoning_count']},
        "total_penalty_applied":     {ts_summary['total_penalty_applied']}
    }},
    "recruiter_self_assessment": {{
        "overall_interview_quality": "<EXCELLENT|GOOD|AVERAGE|POOR>",
        "phases_well_covered":       ["<str>", "..."],
        "phases_neglected":          {json.dumps(uncovered_labels, ensure_ascii=False)},
        "probing_quality":           "<str>",
        "topics_over_explored":      ["<str>", "..."],
        "topics_under_explored":     ["<str>", "..."],
        "recommendation_to_recruiter": "<str: specific advice to improve the interview>"
    }}
}}\n"""

        try:
            resp = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.1, "num_predict": 2500},
            )
            raw = getattr(resp, "response", None) or resp.get("response", "")
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            report = json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"  ❌ Synthesis JSON error: {e} — using fallback report")
            report = {}
        except Exception as e:
            print(f"  ❌ Synthesis LLM error: {e} — using fallback report")
            report = {}

        # ── Lock computed values ──────────────────────────────────────────────
        report["score_total"]              = score_total
        report["coverage_rate_pct"]        = coverage["rate_pct"]
        report["is_partial_evaluation"]    = coverage["is_partial"]
        report["answer_quality_distribution"] = qual_dist
        report["answer_status_distribution"]  = status_dist
        report["turns_summary"]            = ts_summary

        # ── Lock & inject emotion data (computed values, LLM cannot change) ──
        raw_emotion = {
            "available":          emotion["available"],
            "dominant_emotion":   emotion["dominant_emotion"],
            "dominant_fr":        emotion.get("dominant_fr", "neutre"),
            "stress_rate_pct":    emotion["stress_rate_pct"],
            "positive_rate_pct":  emotion["positive_rate_pct"],
            "neutral_rate_pct":   emotion["neutral_rate_pct"],
            "n_frames":           emotion["n_frames"],
            "avg_confidence":     emotion["avg_confidence"],
            "evolution_summary":  emotion["evolution_summary"],
            "stress_peaks":       emotion["stress_peaks"],
            "phase_emotion_map":  emotion["phase_emotion_map"],
            "emotion_distribution": emotion.get("emotion_distribution", {}),
            "recruiter_flags":    emotion["recruiter_flags"],
        }
        # Merge LLM qualitative fields into raw_emotion block
        llm_emotion = report.get("emotion_analysis", {})
        raw_emotion["overall_emotional_state"]     = llm_emotion.get("overall_emotional_state", "")
        raw_emotion["stress_interpretation"]       = llm_emotion.get("stress_interpretation", "")
        raw_emotion["phase_observations"]          = llm_emotion.get("phase_observations", [])
        raw_emotion["emotion_answer_correlation"]  = llm_emotion.get("emotion_answer_correlation", "")
        raw_emotion["recruiter_recommendation"]    = llm_emotion.get("recruiter_recommendation", "")
        report["emotion_analysis"] = raw_emotion

        # Lock phase scores
        if "scores_by_phase" not in report:
            report["scores_by_phase"] = {}
        for ph, pr_data in phase_data_out.items():
            if ph in report["scores_by_phase"]:
                report["scores_by_phase"][ph]["obtained"] = pr_data["obtained_pts"]
                report["scores_by_phase"][ph]["max"]      = pr_data["weight"]
                report["scores_by_phase"][ph]["avg_turn"] = pr_data["avg_score"]
                report["scores_by_phase"][ph]["n_turns"]  = pr_data["n_turns"]
            else:
                report["scores_by_phase"][ph] = {
                    "obtained": pr_data["obtained_pts"],
                    "max":      pr_data["weight"],
                    "avg_turn": pr_data["avg_score"],
                    "n_turns":  pr_data["n_turns"],
                    "status":   pr_data["status"],
                    "comment":  pr_data["comment"],
                }

        # Verdict correction
        if coverage["is_partial"] and report.get("verdict") == "HIRE":
            report["verdict"] = "TO_COMPLETE"
            report["recommendation_detail"] = "[PARTIAL EVALUATION — auto-corrected] " + report.get("recommendation_detail", "")

        # ── Inject full answer-by-answer detail ──────────────────────────────
        report["answer_by_answer"] = [
            {
                "turn_index":             i + 1,
                "phase":                  ts.phase,
                "phase_label":            PHASE_LABELS.get(ts.phase, ts.phase),
                "question":               ts.question[:400],
                "answer_preview":         ts.answer[:400],
                "answer_status":          ts.answer_status,
                "answer_status_icon":     {"ANSWERED_FULLY":"✅","ANSWERED_PARTIALLY":"⚠️",
                                           "ANSWERED_INCORRECTLY":"❌","EVADED":"🔄",
                                           "NOT_ANSWERED":"🚫","OFF_TOPIC":"🚷"}.get(ts.answer_status,"❓"),
                "answer_verdict":         ts.answer_verdict,
                "weighted_score":         ts.weighted_score,
                "status_penalty":         ts.status_penalty,
                "correctness_score":      ts.correctness_score,
                "completeness_score":     ts.completeness_score,
                "completeness_ratio":     ts.completeness_ratio,
                "n_question_parts":       ts.completeness_n_parts,
                "n_parts_addressed":      ts.completeness_n_addressed,
                "llm_chain_quality":      ts.llm_chain_quality,
                "was_answered":           ts.was_answered,
                "was_evaded":             ts.was_evaded,
                "was_incorrect":          ts.was_incorrect,
                "was_partial":            ts.was_partial,
                "relevance":              ts.relevance,
                "technical_accuracy":     ts.technical_accuracy,
                "experience_proof":       ts.experience_proof,
                "depth":                  ts.depth,
                "clarity":                ts.clarity,
                "quantification":         ts.quantification,
                "job_alignment":          ts.job_alignment,
                "star_structure":         ts.star_structure,
                "vagueness_penalty":      ts.vagueness_penalty,
                "ownership":              ts.ownership,
                "decision_quality":       ts.decision_quality,
                "behavioral_quality":     ts.behavioral_quality,
                "is_vague":               ts.is_vague,
                "has_real_example":       ts.has_real_example,
                "has_metrics":            ts.has_metrics,
                "star_detected":          ts.star_detected,
                "strong_ownership":       ts.strong_ownership,
                "decision_reasoning":     ts.decision_reasoning_detected,
                "justification":          ts.llm_justification,
            }
            for i, ts in enumerate(all_turns)
        ]

        report["phase_detail"] = phase_data_out

        return report

    def _build_turns_summary(self, all_turns: List[TurnScore]) -> dict:
        if not all_turns:
            return {
                "total_turns": 0,
                "answered_fully_count": 0, "answered_partially_count": 0,
                "answered_incorrectly_count": 0, "evaded_count": 0,
                "not_answered_count": 0, "off_topic_count": 0,
                "response_rate_pct": 0, "avg_correctness": 0.0,
                "avg_completeness": 0.0, "avg_relevance": 0.0,
                "avg_depth": 0.0, "avg_experience": 0.0,
                "avg_ownership": 0.0, "avg_decision": 0.0,
                "vague_count": 0, "with_metrics": 0, "star_count": 0,
                "strong_ownership_count": 0, "decision_reasoning_count": 0,
                "total_penalty_applied": 0.0,
            }
        n = len(all_turns)
        answered_fully   = sum(1 for t in all_turns if t.answer_status == "ANSWERED_FULLY")
        answered_partial = sum(1 for t in all_turns if t.answer_status == "ANSWERED_PARTIALLY")
        answered_inc     = sum(1 for t in all_turns if t.answer_status == "ANSWERED_INCORRECTLY")
        evaded           = sum(1 for t in all_turns if t.answer_status == "EVADED")
        not_answered     = sum(1 for t in all_turns if t.answer_status == "NOT_ANSWERED")
        off_topic        = sum(1 for t in all_turns if t.answer_status == "OFF_TOPIC")
        response_rate    = round((answered_fully + answered_partial) / n * 100)

        return {
            "total_turns":               n,
            "answered_fully_count":      answered_fully,
            "answered_partially_count":  answered_partial,
            "answered_incorrectly_count":answered_inc,
            "evaded_count":              evaded,
            "not_answered_count":        not_answered,
            "off_topic_count":           off_topic,
            "response_rate_pct":         response_rate,
            "avg_correctness":           round(sum(t.correctness_score for t in all_turns) / n, 1),
            "avg_completeness":          round(sum(t.completeness_score for t in all_turns) / n, 1),
            "avg_relevance":             round(sum(t.relevance for t in all_turns) / n, 1),
            "avg_depth":                 round(sum(t.depth for t in all_turns) / n, 1),
            "avg_experience":            round(sum(t.experience_proof for t in all_turns) / n, 1),
            "avg_ownership":             round(sum(t.ownership for t in all_turns) / n, 1),
            "avg_decision":              round(sum(t.decision_quality for t in all_turns) / n, 1),
            "vague_count":               sum(1 for t in all_turns if t.is_vague),
            "with_metrics":              sum(1 for t in all_turns if t.has_metrics),
            "star_count":                sum(1 for t in all_turns if t.star_detected),
            "strong_ownership_count":    sum(1 for t in all_turns if t.strong_ownership),
            "decision_reasoning_count":  sum(1 for t in all_turns if t.decision_reasoning_detected),
            "total_penalty_applied":     round(sum(t.status_penalty for t in all_turns), 2),
        }


# =============================================================================
# 10. INTERVIEW EVALUATOR — Orchestrateur principal
# =============================================================================

class InterviewEvaluator:
    """
    Orchestrateur principal v6.1.

    Usage:
        evaluator = InterviewEvaluator(model_name="qwen2.5:7b")
        evaluator.set_job_context(job_offer_text)
        evaluator.set_vision_data(vision_dict)   # ← nouveau v6.1
        report = evaluator.evaluate_file("data/interview_20250115_143211.txt")
        evaluator.display(report)
        evaluator.save_json(report)
        evaluator.save_markdown(report)
    """

    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model        = model_name
        self.client       = Client()
        self.parser       = LogParser()
        self.qa_extractor = QAPairExtractor()
        self.coverage     = CoverageAnalyzer()
        self.scorer       = TurnScorer(self.client, model_name)
        self.synthesis    = SynthesisEngine(self.client, model_name)
        self._job_context = ""
        self._vision_data: Optional[dict] = None   # ← nouveau v6.1

    def set_job_context(self, text: str):
        self._job_context = text[:2500]

    def set_vision_data(self, vision_data: Optional[dict]):
        """
        Injecte les données VisionEngine dans l'évaluateur.
        Appelé depuis main._build_full_report() avant evaluate_file/text.

        Args:
            vision_data : dict produit par main._build_vision_analysis()
                          Structure attendue :
                          {
                            "disponible": bool,
                            "timeline": [...],          ← liste d'entrées {timestamp, phase, emotion, raw, confidence}
                            "emotion_dominante": str,
                            "pics_stress": [...],
                            "evolution": str,
                            "nb_frames": int,
                            "confiance_globale": float,
                          }
        """
        self._vision_data = vision_data

    def evaluate_file(self, file_path: str) -> dict:
        print(f"\n📂 Loading: {file_path}")
        parsed = self.parser.parse_file(file_path)
        return self._run_pipeline(parsed, source_file=file_path)

    def evaluate_text(self, raw_text: str) -> dict:
        parsed = self.parser.parse_text(raw_text)
        return self._run_pipeline(parsed)

    def evaluate_latest(self) -> Tuple[dict, str]:
        pattern = os.path.join(INTERVIEW_FOLDER, "interview_*.txt")
        files   = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No interview_*.txt found in: {INTERVIEW_FOLDER}")
        file_path = max(files, key=os.path.getmtime)
        return self.evaluate_file(file_path), file_path

    def _run_pipeline(self, parsed: dict, source_file: str = "") -> dict:
        meta = parsed["meta"]
        print(f"✅ Parsed — language: {meta['langue']} | duration: {meta['duration']}")
        print(f"   Candidate turns: {len(parsed['all_candidate_texts'])}")

        # Résumé vision dispo
        if self._vision_data and self._vision_data.get("disponible"):
            n_frames = self._vision_data.get("nb_frames", len(self._vision_data.get("timeline", [])))
            print(f"   👁️  Vision data: {n_frames} frames | dominant={self._vision_data.get('emotion_dominante', 'N/A')}")
        else:
            print("   👁️  Vision data: not available")

        cov = self.coverage.analyze(parsed["phases"])
        print(f"   Coverage: {cov['rate_pct']}% ({'PARTIAL' if cov['is_partial'] else 'OK'})")
        if cov["uncovered_phases"]:
            print("   ⚠️  Not covered: " + ", ".join(PHASE_LABELS.get(p, p) for p in cov["uncovered_phases"]))

        print("\n🔬 Phase 1 — Rigorous turn-level scoring (15 dimensions + answer status)...")
        agg = self._score_all_turns(parsed)

        ts_summary  = self.synthesis._build_turns_summary(agg.all_turn_scores())
        status_dist = agg.answer_status_distribution()

        print(f"\n   Intermediate score: {agg.total_score()}/100")
        print(f"   Quality distribution: {agg.quality_distribution()}")
        print(f"   Answer status distribution: {status_dist}")
        print(f"   Response rate: {ts_summary['response_rate_pct']}%")
        print(f"   Total penalty applied: {ts_summary['total_penalty_applied']:+.1f}")

        print("\n🧠 Phase 2 — Qualitative synthesis with emotion + answer-by-answer review...")
        report = self.synthesis.synthesize(
            parsed,
            agg,
            cov,
            job_context=self._job_context,
            vision_data=self._vision_data,   # ← passé ici
        )

        if source_file:
            report.setdefault("meta", {})["source_file"] = source_file

        print("✅ Evaluation complete.")
        return report

    def _score_all_turns(self, parsed: dict) -> PhaseAggregator:
        agg = PhaseAggregator()
        total_pairs = 0

        for phase in SCORED_PHASES:
            turns = parsed["phases"].get(phase, [])
            if not turns:
                continue

            pairs = self.qa_extractor.extract(turns)
            if not pairs:
                continue

            label = PHASE_LABELS.get(phase, phase)
            print(f"  🔍 [{label}]: {len(pairs)} Q→R pair(s)...")

            for idx, pair in enumerate(pairs, 1):
                ts = self.scorer.score_turn(
                    question       = pair["question"],
                    answer         = pair["answer"],
                    phase          = phase,
                    candidate_turn = pair["candidate_turn"],
                    job_context    = self._job_context,
                )
                agg.add(phase, ts)
                total_pairs += 1

                # Status icon
                status_icon = {
                    "ANSWERED_FULLY":        "✅",
                    "ANSWERED_PARTIALLY":    "⚠️ ",
                    "ANSWERED_INCORRECTLY":  "❌",
                    "EVADED":                "🔄",
                    "NOT_ANSWERED":          "🚫",
                    "OFF_TOPIC":             "🚷",
                }.get(ts.answer_status, "❓")

                flags = []
                if ts.has_real_example:          flags.append("✅EX")
                if ts.has_metrics:               flags.append("📊MT")
                if ts.star_detected:             flags.append("🌟ST")
                if ts.strong_ownership:          flags.append("👤OW")
                if ts.decision_reasoning_detected: flags.append("🧠DEC")
                if ts.is_vague:                  flags.append("⚠️VG")

                just_short = (ts.llm_justification[:100] + "…") if len(ts.llm_justification) > 100 else ts.llm_justification

                print(
                    f"    Q{idx} {status_icon} {ts.answer_status:<22}"
                    f" | score={ts.weighted_score:4.1f}/10"
                    f" | correct={ts.correctness_score}/10"
                    f" | penalty={ts.status_penalty:+.1f}"
                    f" | {' '.join(flags) or '—'}"
                    + (f"\n        {just_short}" if just_short else "")
                )

        print(f"  ✅ Scoring done — {total_pairs} turn(s) analyzed.")
        return agg

    # ── Display ───────────────────────────────────────────────────────────────

    def display(self, report: dict):
        W = 82
        print("\n" + "═" * W)
        print("   🏆  INTERVIEW EVALUATION  —  Rigorous Answer Analysis v6.0")
        print("   15 dimensions | Answer status | Correctness | Completeness | Penalty")
        print("═" * W)

        meta = report.get("meta", {})
        print(f"  📅 Date      : {meta.get('interview_date', 'N/A')}")
        print(f"  🌐 Language  : {meta.get('language', 'N/A')}")
        print(f"  ⏱️  Duration  : {meta.get('duration', 'N/A')}")
        if meta.get("source_file"):
            print(f"  📂 Source    : {os.path.basename(meta['source_file'])}")
        print()

        # Coverage
        cov_pct = report.get("coverage_rate_pct", "N/A")
        partial  = report.get("is_partial_evaluation", False)
        print(f"  📊 Coverage  : {cov_pct}%" + (" ⚠️  PARTIAL EVALUATION" if partial else " ✅"))
        print()

        # ── EMOTION ANALYSIS (nouveau v6.1) ──────────────────────────────────
        ea = report.get("emotion_analysis", {})
        if ea.get("available"):
            print("  😐 CANDIDATE EMOTION ANALYSIS (DeepFace real-time):")
            bar_stress   = "█" * (ea['stress_rate_pct']   // 10) + "░" * (10 - ea['stress_rate_pct']   // 10)
            bar_positive = "█" * (ea['positive_rate_pct'] // 10) + "░" * (10 - ea['positive_rate_pct'] // 10)
            bar_neutral  = "█" * (ea['neutral_rate_pct']  // 10) + "░" * (10 - ea['neutral_rate_pct']  // 10)
            print(f"    Dominant emotion  : {ea.get('dominant_emotion', 'N/A')} ({ea.get('dominant_fr', '')})")
            print(f"    Frames analyzed   : {ea.get('n_frames', 0)} (avg confidence {ea.get('avg_confidence', 0)}%)")
            print(f"    Stress/anxiety    : {bar_stress}  {ea['stress_rate_pct']}%")
            print(f"    Relaxed/confident : {bar_positive}  {ea['positive_rate_pct']}%")
            print(f"    Neutral           : {bar_neutral}  {ea['neutral_rate_pct']}%")
            print(f"    Evolution         : {ea.get('evolution_summary', 'N/A')}")
            if ea.get("overall_emotional_state"):
                print(f"    Overall state     : {ea['overall_emotional_state']}")
            if ea.get("stress_interpretation"):
                print(f"    Interpretation    : {ea['stress_interpretation']}")
            for flag in ea.get("recruiter_flags", []):
                print(f"    ⚠️   {flag}")
            for obs in ea.get("phase_observations", []):
                print(f"    📍 {obs}")
            if ea.get("emotion_answer_correlation"):
                print(f"    🔗 Correlation     : {ea['emotion_answer_correlation']}")
            if ea.get("recruiter_recommendation"):
                print(f"    💡 Recommendation  : {ea['recruiter_recommendation']}")
            # Stress peaks
            peaks = ea.get("stress_peaks", [])
            if peaks:
                print(f"    ⚠️   Stress peaks ({len(peaks)}):")
                for p in peaks[:5]:
                    print(f"       [{p.get('phase','')} @ {p.get('timestamp','')}]: "
                          f"{p.get('emotion_en', p.get('emotion', ''))} ({p.get('confidence', 0):.0f}%)")
            print()
        elif ea:
            print("  😐 EMOTION ANALYSIS: Vision data not available (camera off or DeepFace inactive)")
            print()

        # ── ANSWER STATUS OVERVIEW ────────────────────────────────────────────
        sd = report.get("answer_status_distribution", {})
        ts_sum = report.get("turns_summary", {})
        if sd:
            print("  📋 ANSWER STATUS OVERVIEW (v6 — rigorous evaluation):")
            status_cfg = [
                ("ANSWERED_FULLY",        "✅ Fully answered"),
                ("ANSWERED_PARTIALLY",    "⚠️  Partially answered"),
                ("ANSWERED_INCORRECTLY",  "❌ Answered incorrectly"),
                ("EVADED",                "🔄 Evaded / pivoted"),
                ("NOT_ANSWERED",          "🚫 Not answered"),
                ("OFF_TOPIC",             "🚷 Off-topic"),
            ]
            total_turns = sum(sd.values())
            for key, label in status_cfg:
                count = sd.get(key, 0)
                if count == 0:
                    continue
                pct = round(count / total_turns * 100) if total_turns else 0
                bar = "█" * count + "░" * max(0, 8 - count)
                print(f"    {label:<30} {bar}  {count:2d} ({pct:2d}%)")
            rr = ts_sum.get("response_rate_pct", 0)
            pen = ts_sum.get("total_penalty_applied", 0.0)
            print(f"    {'─'*52}")
            print(f"    Response rate (fully+partially): {rr}%")
            print(f"    Total penalty applied to scores: {pen:+.1f} pts")
            print()

        # ── ANSWER-BY-ANSWER TABLE ────────────────────────────────────────────
        aba = report.get("answer_by_answer", [])
        if aba:
            print("  🔍 ANSWER-BY-ANSWER ANALYSIS:")
            print(f"    {'#':<4} {'Phase':<30} {'Status':<24} {'Score':>5} {'Corr':>5} {'Pen':>5}")
            print(f"    {'─'*80}")
            for item in aba:
                icon   = item.get("answer_status_icon", "❓")
                status = item.get("answer_status", "?")
                score  = item.get("weighted_score", 0)
                corr   = item.get("correctness_score", 0)
                pen    = item.get("status_penalty", 0.0)
                phase_lbl = item.get("phase_label", "")[:28]
                q_preview = item.get("question", "")[:55]
                print(f"    Q{item['turn_index']:<3} {phase_lbl:<30} {icon} {status:<22} {score:4.1f}  {corr:4d}  {pen:+4.1f}")
                print(f"         ↳ {q_preview}")
                verdict = item.get("answer_verdict", "")
                if verdict:
                    print(f"         → {verdict}")
            print()

        # ── Quality distribution ──────────────────────────────────────────────
        qd = report.get("answer_quality_distribution", {})
        if qd:
            print("  📈 llm_chain Quality Distribution:")
            for q, count in sorted(qd.items()):
                bar = "█" * count + "░" * max(0, 8 - count)
                print(f"    {q:<12} {bar}  {count}")
            print()

        # ── 15-Dimension summary ──────────────────────────────────────────────
        if ts_sum.get("total_turns", 0) > 0:
            print("  🔬 15-DIMENSION SCORING SUMMARY:")
            rows = [
                ("Turns analyzed",        ts_sum["total_turns"],                ""),
                ("Response rate",         str(ts_sum["response_rate_pct"]) + "%",""),
                ("Avg correctness",        ts_sum["avg_correctness"],            "/10"),
                ("Avg completeness",       ts_sum["avg_completeness"],           "/10"),
                ("Avg relevance",          ts_sum["avg_relevance"],              "/10"),
                ("Avg depth",              ts_sum["avg_depth"],                  "/10"),
                ("Avg experience proof",   ts_sum["avg_experience"],             "/10"),
                ("Avg ownership",          ts_sum["avg_ownership"],              "/10"),
                ("Avg decision quality",   ts_sum["avg_decision"],               "/10"),
                ("Vague answers",          ts_sum["vague_count"],                ""),
                ("With metrics",           ts_sum["with_metrics"],               ""),
                ("STAR structure",         ts_sum["star_count"],                 ""),
                ("Strong ownership",       ts_sum["strong_ownership_count"],     ""),
                ("Decision reasoning",     ts_sum["decision_reasoning_count"],   ""),
                ("Total penalty applied",  ts_sum["total_penalty_applied"],      ""),
            ]
            for name, val, unit in rows:
                print(f"    {name:<28}: {val}{unit}")
            print()

        # ── Phase scores ──────────────────────────────────────────────────────
        phases = report.get("scores_by_phase", {})
        if phases:
            print("  📊 SCORES BY PHASE:")
            total_obtained = total_max = 0
            for ph in SCORED_PHASES:
                if ph not in phases:
                    continue
                pdata    = phases[ph]
                obtained = pdata.get("obtained", 0)
                maxi     = pdata.get("max", PHASE_WEIGHTS.get(ph, 0))
                avg      = pdata.get("avg_turn", 0)
                n        = pdata.get("n_turns", 0)
                status   = pdata.get("status", "")
                comment  = pdata.get("comment", "")
                total_obtained += obtained
                total_max      += maxi
                pct = round((obtained / maxi) * 100) if maxi else 0
                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                icon = "🚫" if status == "NOT_EVALUATED" else ("⚠️" if status == "PARTIAL" else "✅")
                print(f"    {icon} {PHASE_LABELS.get(ph, ph):<38} {bar}  {obtained:2d}/{maxi:2d}pts"
                      + (f"  (avg={avg:.1f}, n={n})" if n else ""))
                if comment:
                    print(f"    {'':43}  → {comment}")
            print(f"    {'─'*65}")
            print(f"    {'TOTAL':<40}{'':13}  {total_obtained:2d}/{total_max:2d}pts")
            print()

        # ── Main scores ───────────────────────────────────────────────────────
        print(f"  ⭐ Score Total        : {report.get('score_total', 0)}/100")
        print(f"  🔧 Score Technical    : {report.get('score_technical', 0)}/100")
        print(f"  🤝 Score Behavioral   : {report.get('score_behavioral', 0)}/100")
        print(f"  💬 Score Communication: {report.get('score_communication', 0)}/100")
        print()

        # ── Answer patterns (nouveau v6) ──────────────────────────────────────
        ap = report.get("answer_patterns", {})
        if ap:
            print("  🎯 ANSWER PATTERNS (v6):")
            for key, label, icon in [
                ("evasion_topics",        "Topics evaded",          "🔄"),
                ("incorrect_answers",     "Incorrect answers",      "❌"),
                ("partial_answer_themes", "Partial answer themes",  "⚠️ "),
                ("strongest_answers",     "Strongest answers",      "✅"),
            ]:
                items = ap.get(key, [])
                if items:
                    print(f"    {icon} {label}:")
                    for item in items:
                        print(f"       • {item}")
            print()

        # ── Qualitative sections ──────────────────────────────────────────────
        for key, label, icon in [
            ("competencies_detected",    "COMPETENCIES DETECTED",   "✅"),
            ("gaps_identified",          "GAPS IDENTIFIED",          "⚠️ "),
            ("green_flags",              "GREEN FLAGS",              "🟢"),
            ("red_flags",                "RED FLAGS",                "🔴"),
            ("strengths",                "STRENGTHS",                "💪"),
            ("improvement_areas",        "AREAS FOR IMPROVEMENT",    "📈"),
        ]:
            items = report.get(key, [])
            if items:
                print(f"  {icon} {label}:")
                for item in items:
                    print(f"    • {item}")
                print()

        for key, label, icon in [
            ("motivation_analysis",      "MOTIVATION ANALYSIS",      "🎯"),
            ("soft_skills_analysis",     "SOFT SKILLS ANALYSIS",     "🤝"),
            ("decision_making_analysis", "DECISION-MAKING ANALYSIS", "🧠"),
            ("ownership_patterns",       "OWNERSHIP PATTERNS",       "👤"),
        ]:
            val = report.get(key, "")
            if val and val not in ("N/A", ""):
                print(f"  {icon} {label}:")
                print(f"    {val}")
                print()

        # ── Verdict ───────────────────────────────────────────────────────────
        verdict = report.get("verdict", "N/A")
        v_icon  = {"HIRE": "✅", "RECONSIDER": "⚠️ ", "TO_COMPLETE": "🔁", "NOT_RETAINED": "❌"}.get(verdict, "❓")
        print(f"  {v_icon} VERDICT: {verdict}")
        reco = report.get("recommendation_detail", "")
        if reco:
            print(f"     → {reco}")
        print()

        # ── Recruiter self-assessment ─────────────────────────────────────────
        rsa = report.get("recruiter_self_assessment", {})
        if rsa:
            print("  🔍 RECRUITER SELF-ASSESSMENT:")
            print(f"    Overall quality  : {rsa.get('overall_interview_quality', 'N/A')}")
            for item in rsa.get("phases_neglected", []):
                print(f"    🚫 Neglected     : {item}")
            for item in rsa.get("topics_over_explored", []):
                print(f"    🔁 Over-explored : {item}")
            for item in rsa.get("topics_under_explored", []):
                print(f"    ❓ Under-explored: {item}")
            if rsa.get("probing_quality"):
                print(f"    💡 Probing quality: {rsa['probing_quality']}")
            if rsa.get("recommendation_to_recruiter"):
                print(f"    📋 Recommendation: {rsa['recommendation_to_recruiter']}")

        print("═" * W)

    # ── Save JSON ─────────────────────────────────────────────────────────────

    def save_json(self, report: dict, output_path: str = "") -> str:
        if not output_path:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            lang = report.get("meta", {}).get("language", "unknown").replace(" ", "_")
            output_path = os.path.join(REPORT_FOLDER, f"eval_v6_{lang}_{ts}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON report saved: {output_path}")
        return output_path

    # ── Save Markdown ─────────────────────────────────────────────────────────

    def save_markdown(self, report: dict, output_path: str = "") -> str:
        if not output_path:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            lang = report.get("meta", {}).get("language", "unknown").replace(" ", "_")
            output_path = os.path.join(REPORT_FOLDER, f"eval_v6_{lang}_{ts}.md")

        meta    = report.get("meta", {})
        verdict = report.get("verdict", "N/A")
        score   = report.get("score_total", 0)
        ts_sum  = report.get("turns_summary", {})
        sd      = report.get("answer_status_distribution", {})

        lines = [
            "# Interview Evaluation Report — v6.0 (Rigorous Answer Analysis)",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Date | {meta.get('interview_date', 'N/A')} |",
            f"| Language | {meta.get('language', 'N/A')} |",
            f"| Duration | {meta.get('duration', 'N/A')} |",
            f"| Report Generated | {meta.get('report_date', 'N/A')} |",
            f"| Coverage | {report.get('coverage_rate_pct', 0)}% {'⚠️ PARTIAL' if report.get('is_partial_evaluation') else '✅'} |",
            "",
            "## Score Summary",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| **Total Score** | **{score}/100** |",
            f"| Technical | {report.get('score_technical', 0)}/100 |",
            f"| Behavioral | {report.get('score_behavioral', 0)}/100 |",
            f"| Communication | {report.get('score_communication', 0)}/100 |",
            "",
            "## Answer Status Overview (v6 — Rigorous Evaluation)",
            "",
            "| Status | Count | % |",
            "|--------|-------|---|",
        ]

        total_t = max(1, sum(sd.values()))
        status_labels = {
            "ANSWERED_FULLY":        "✅ Fully answered",
            "ANSWERED_PARTIALLY":    "⚠️ Partially answered",
            "ANSWERED_INCORRECTLY":  "❌ Answered incorrectly",
            "EVADED":                "🔄 Evaded / pivoted",
            "NOT_ANSWERED":          "🚫 Not answered",
            "OFF_TOPIC":             "🚷 Off-topic",
        }
        for key, label in status_labels.items():
            count = sd.get(key, 0)
            pct   = round(count / total_t * 100)
            lines.append(f"| {label} | {count} | {pct}% |")

        lines += [
            f"| **Response Rate (fully+partially)** | — | **{ts_sum.get('response_rate_pct', 0)}%** |",
            f"| Total penalty applied | — | **{ts_sum.get('total_penalty_applied', 0.0):+.1f} pts** |",
            "",
        ]

        # Emotion analysis section
        ea = report.get("emotion_analysis", {})
        if ea.get("available"):
            lines += [
                "",
                "## 😐 Candidate Emotion Analysis (DeepFace Real-Time)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Dominant emotion | {ea.get('dominant_emotion', 'N/A')} ({ea.get('dominant_fr', '')}) |",
                f"| Frames analyzed | {ea.get('n_frames', 0)} (avg confidence {ea.get('avg_confidence', 0)}%) |",
                f"| Stress / anxiety rate | **{ea.get('stress_rate_pct', 0)}%** |",
                f"| Relaxed / confident rate | {ea.get('positive_rate_pct', 0)}% |",
                f"| Neutral rate | {ea.get('neutral_rate_pct', 0)}% |",
                f"| Emotional evolution | {ea.get('evolution_summary', 'N/A')} |",
                "",
            ]
            if ea.get("overall_emotional_state"):
                lines += [f"**Overall state**: {ea['overall_emotional_state']}", ""]
            if ea.get("stress_interpretation"):
                lines += [f"**Interpretation**: {ea['stress_interpretation']}", ""]
            if ea.get("emotion_answer_correlation"):
                lines += [f"**Emotion ↔ Answer correlation**: {ea['emotion_answer_correlation']}", ""]

            flags = ea.get("recruiter_flags", [])
            if flags:
                lines += ["**Recruiter signals:**", ""] + [f"- ⚠️  {f}" for f in flags] + [""]

            obs = ea.get("phase_observations", [])
            if obs:
                lines += ["**Phase observations:**", ""] + [f"- 📍 {o}" for o in obs] + [""]

            peaks = ea.get("stress_peaks", [])
            if peaks:
                lines += [
                    f"**Stress peaks ({len(peaks)}):**", "",
                    "| Phase | Timestamp | Emotion | Confidence |",
                    "|-------|-----------|---------|------------|",
                ]
                for p in peaks[:8]:
                    lines.append(
                        f"| {p.get('phase','')} | {p.get('timestamp','')} "
                        f"| {p.get('emotion_en', p.get('emotion',''))} | {p.get('confidence',0):.0f}% |"
                    )
                lines.append("")

            if ea.get("recruiter_recommendation"):
                lines += [f"**Recruiter recommendation**: {ea['recruiter_recommendation']}", ""]

        # Answer by answer
        aba = report.get("answer_by_answer", [])
        if aba:
            lines += [
                "## Answer-by-Answer Analysis",
                "",
                "| # | Phase | Status | Score | Correct | Penalty | Question (preview) |",
                "|---|-------|--------|-------|---------|---------|---------------------|",
            ]
            for item in aba:
                icon    = item.get("answer_status_icon", "❓")
                status  = item.get("answer_status", "?")
                q_prev  = item.get("question", "")[:50].replace("|", "\\|")
                lines.append(
                    f"| Q{item['turn_index']} "
                    f"| {item.get('phase_label','')[:25]} "
                    f"| {icon} {status} "
                    f"| {item.get('weighted_score', 0):.1f} "
                    f"| {item.get('correctness_score', 0)}/10 "
                    f"| {item.get('status_penalty', 0.0):+.1f} "
                    f"| {q_prev}… |"
                )
                if item.get("answer_verdict"):
                    lines.append(f"|   |   |   |   |   |   | → *{item['answer_verdict'][:80]}* |")
            lines.append("")

        # Phase scores
        lines += [
            "## Scores by Phase",
            "",
            "| Phase | Obtained | Max | Avg/Turn | Turns | Status |",
            "|-------|----------|-----|----------|-------|--------|",
        ]
        for ph in SCORED_PHASES:
            pdata = report.get("scores_by_phase", {}).get(ph, {})
            lines.append(
                f"| {PHASE_LABELS.get(ph, ph)} "
                f"| {pdata.get('obtained', 0)} "
                f"| {pdata.get('max', PHASE_WEIGHTS.get(ph, 0))} "
                f"| {pdata.get('avg_turn', 0):.1f} "
                f"| {pdata.get('n_turns', 0)} "
                f"| {pdata.get('status', 'N/A')} |"
            )

        lines += ["", "## 15-Dimension Scoring Summary", ""]
        if ts_sum:
            lines += [
                "| Dimension | Value |",
                "|-----------|-------|",
                f"| Total turns analyzed | {ts_sum.get('total_turns', 0)} |",
                f"| Response rate | {ts_sum.get('response_rate_pct', 0)}% |",
                f"| Avg correctness | {ts_sum.get('avg_correctness', 0)}/10 |",
                f"| Avg completeness | {ts_sum.get('avg_completeness', 0)}/10 |",
                f"| Avg relevance | {ts_sum.get('avg_relevance', 0)}/10 |",
                f"| Avg depth | {ts_sum.get('avg_depth', 0)}/10 |",
                f"| Avg experience proof | {ts_sum.get('avg_experience', 0)}/10 |",
                f"| Avg ownership | {ts_sum.get('avg_ownership', 0)}/10 |",
                f"| Avg decision quality | {ts_sum.get('avg_decision', 0)}/10 |",
                f"| Vague answers | {ts_sum.get('vague_count', 0)} |",
                f"| With quantified metrics | {ts_sum.get('with_metrics', 0)} |",
                f"| STAR structure | {ts_sum.get('star_count', 0)} |",
                f"| Strong personal ownership | {ts_sum.get('strong_ownership_count', 0)} |",
                f"| Decision reasoning | {ts_sum.get('decision_reasoning_count', 0)} |",
                f"| Total penalty applied | {ts_sum.get('total_penalty_applied', 0.0):+.1f} pts |",
            ]

        # Answer patterns
        ap = report.get("answer_patterns", {})
        if ap:
            lines += ["", "## 🎯 Answer Patterns (v6)", ""]
            for key, title in [
                ("evasion_topics",        "Topics Evaded 🔄"),
                ("incorrect_answers",     "Incorrect Answers ❌"),
                ("partial_answer_themes", "Partial Answer Themes ⚠️"),
                ("strongest_answers",     "Strongest Answers ✅"),
            ]:
                items = ap.get(key, [])
                if items:
                    lines += [f"### {title}", ""] + [f"- {i}" for i in items] + [""]

        def _list_section(key, title, icon):
            items = report.get(key, [])
            if not items:
                return []
            return [f"", f"## {icon} {title}", ""] + [f"- {i}" for i in items]

        for key, title, icon in [
            ("competencies_detected",    "Competencies Detected",     "✅"),
            ("gaps_identified",          "Gaps Identified",            "⚠️"),
            ("green_flags",              "Green Flags",                "🟢"),
            ("red_flags",                "Red Flags",                  "🔴"),
            ("strengths",                "Strengths",                  "💪"),
            ("improvement_areas",        "Areas for Improvement",      "📈"),
        ]:
            lines.extend(_list_section(key, title, icon))

        for key, title in [
            ("motivation_analysis",      "Motivation Analysis"),
            ("soft_skills_analysis",     "Soft Skills Analysis"),
            ("decision_making_analysis", "Decision-Making Analysis"),
            ("ownership_patterns",       "Ownership Patterns"),
        ]:
            val = report.get(key, "")
            if val and val != "N/A":
                lines += ["", f"## {title}", "", val]

        verdict_icon = {"HIRE": "✅", "RECONSIDER": "⚠️", "TO_COMPLETE": "🔁", "NOT_RETAINED": "❌"}.get(verdict, "❓")
        lines += [
            "",
            f"## {verdict_icon} Verdict: {verdict}",
            "",
            report.get("recommendation_detail", ""),
        ]

        rsa = report.get("recruiter_self_assessment", {})
        if rsa:
            lines += [
                "",
                "## 🔍 Recruiter Self-Assessment",
                "",
                f"**Overall interview quality**: {rsa.get('overall_interview_quality', 'N/A')}",
                f"**Probing quality**: {rsa.get('probing_quality', 'N/A')}",
                "",
            ]
            for item in rsa.get("phases_neglected", []):
                lines.append(f"- 🚫 Neglected: {item}")
            for item in rsa.get("topics_over_explored", []):
                lines.append(f"- 🔁 Over-explored: {item}")
            for item in rsa.get("topics_under_explored", []):
                lines.append(f"- ❓ Under-explored: {item}")
            rr = rsa.get("recommendation_to_recruiter", "")
            if rr:
                lines += ["", f"**Recommendation to Recruiter**: {rr}"]

        lines += [
            "",
            "---",
            "*Generated by interview_evaluator.py v6.0 — Rigorous Answer Analysis Engine*",
            "*15 dimensions | Answer status | Correctness | Completeness | Penalty system*",
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"📝 Markdown report saved: {output_path}")
        return output_path


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interview Evaluator v6.0 — Rigorous Answer Analysis Engine"
    )
    parser.add_argument("--file",  "-f", type=str, default=None,
                        help="Path to interview_*.txt (default: latest in data/).")
    parser.add_argument("--model", "-m", type=str, default="qwen2.5:7b",
                        help="Ollama model name.")
    parser.add_argument("--job",   "-j", type=str, default=None,
                        help="Path to job description .txt for alignment scoring.")
    parser.add_argument("--no-markdown", action="store_true",
                        help="Skip Markdown report generation.")
    args = parser.parse_args()

    evaluator = InterviewEvaluator(model_name=args.model)

    if args.job:
        try:
            with open(args.job, "r", encoding="utf-8", errors="ignore") as jf:
                evaluator.set_job_context(jf.read())
            print(f"✅ Job context loaded: {args.job}")
        except Exception as e:
            print(f"⚠️  Could not load job file: {e}")

    try:
        if args.file:
            report      = evaluator.evaluate_file(args.file)
            source_file = args.file
        else:
            report, source_file = evaluator.evaluate_latest()

        evaluator.display(report)
        evaluator.save_json(report)
        if not args.no_markdown:
            evaluator.save_markdown(report)

    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ Fatal error: {e}")
        print(traceback.format_exc())
        sys.exit(1)