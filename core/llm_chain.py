"""
llm_chain.py  ·  HR Interactive Brain
Senior Hiring Manager — Structured 8-Phase Interview

═══════════════════════════════════════════════════════════════════════════════
CHANGELOG — V3  (all V1+V2 bug-fixes retained + new RAG improvements)
═══════════════════════════════════════════════════════════════════════════════

[V1 Bug fixes retained — BUG 1–9 + DIVERSITY FIX]
[V2 Improvements retained — A through G + FIX-AR-1 through FIX-AR-10]

NEW RAG IMPROVEMENTS IN V3
═══════════════════════════════════════════════════════════════════════════════

RAG-IMPROVEMENT-1 ── Hybrid Search: BM25 + Dense Vector Fusion
    • SimpleVectorStore now maintains a BM25 index in parallel.
    • _retrieve_context() calls BOTH BM25 and dense vector search,
      then fuses scores with Reciprocal Rank Fusion (RRF, k=60).
    • Falls back to dense-only if rank_bm25 is not installed.
    • Configurable fusion weight: HYBRID_ALPHA (0=BM25 only, 1=dense only).

RAG-IMPROVEMENT-2 ── Cross-Encoder Re-ranking (after top-K fusion)
    • After hybrid fusion, top-K candidates are re-ranked with a
      lightweight cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2).
    • Cross-encoder scores replace cosine scores for final ordering.
    • Falls back to vector-only ranking if model unavailable.
    • Configurable via: RERANKER_MODEL, RERANKER_TOP_K, USE_CROSS_ENCODER.

RAG-IMPROVEMENT-3 ── Semantic Chunking (replaces fixed-size chunking)
    • _chunk_text() is replaced by _semantic_chunk_text().
    • Splits on sentence boundaries using a regex sentence splitter.
    • Groups sentences into chunks such that each chunk's embedding
      stays semantically coherent (cosine-breakpoint detection).
    • Breakpoint threshold: SEMANTIC_BREAKPOINT_THRESHOLD (default 0.35).
    • Falls back to fixed-size chunking if embeddings unavailable.
    • Respects SEMANTIC_MAX_CHUNK_TOKENS (soft cap, in words).
"""
import hashlib
import pickle
import os
import re
import sys
import json
import math
import time
import random
import glob
import queue
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

import numpy as np
from ollama import Client

# ── GPU/CPU configuration (RTX 2050 optimisé) ─────────────────────────────────
from device_config import DEVICE_CONFIG, log_device_config

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLAMA_MODEL = "qwen2.5:7b"
ARABIC_MODEL = "qwen2.5:7b"
MAX_HISTORY_TURNS = 14
MAX_RETRIEVED_CHUNKS = 8
RETRIEVAL_TOP_K = 12
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
HF_CACHE_DIR = BASE_DIR / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_CACHE_DIR = DATA_DIR / "embed_cache"
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.getenv("HF_TOKEN"))

# ── Maximum follow-up questions per JD tool before forcing a switch ──────────
MAX_TOOL_FOLLOWUPS = 3

# ── IMPROVEMENT A: max vague retries before abandoning topic ─────────────────
MAX_VAGUE_RETRIES = 3

# ── IMPROVEMENT C: n-gram size for fingerprinting ────────────────────────────
NGRAM_SIZE = 4

# ── IMPROVEMENT G: humanness score minimum ───────────────────────────────────
MIN_HUMANNESS_SCORE = 3

# =============================================================================
# RAG-IMPROVEMENT-1 CONFIG — Hybrid BM25 + Dense Vector Search
# =============================================================================
# Alpha weight for score fusion: 1.0 = dense-only, 0.0 = BM25-only, 0.5 = equal.
HYBRID_ALPHA: float = 0.6
# Reciprocal Rank Fusion constant (higher = smoother fusion).
RRF_K: int = 60
# Whether to attempt BM25 import (set False to disable).
USE_BM25: bool = True

# =============================================================================
# RAG-IMPROVEMENT-2 CONFIG — Cross-Encoder Re-ranking
# =============================================================================
# HuggingFace cross-encoder model for re-ranking.
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Number of candidates to pass to the cross-encoder after fusion.
RERANKER_TOP_K: int = 16
# Set to False to disable cross-encoder (fallback to hybrid-only ranking).
USE_CROSS_ENCODER: bool = True

# =============================================================================
# RAG-IMPROVEMENT-3 CONFIG — Semantic Chunking
# =============================================================================
# Cosine-distance breakpoint threshold: a new chunk starts when consecutive
# sentence-pair similarity drops below this value.
SEMANTIC_BREAKPOINT_THRESHOLD: float = 0.35
# Soft cap on chunk size (in words) before forcing a split regardless.
SEMANTIC_MAX_CHUNK_TOKENS: int = 220
# Minimum words per semantic chunk.
SEMANTIC_MIN_CHUNK_TOKENS: int = 40
# =============================================================================
# STRUCTURED CHUNKING — Section detection patterns
# =============================================================================

CV_SECTION_PATTERNS = re.compile(
    r"^(?:"
    r"expériences?(?:\s+professionnelles?)?|experiences?(?:\s+professionnelles?)?|"
    r"تجارب(?:\s+مهنية)?|خبرات?|"
    r"formations?|education|diplômes?|studies|تعليم|دراسة|"
    r"compétences?(?:\s+techniques?)?|skills?(?:\s+techniques?)?|مهارات?|"
    r"projets?(?:\s+professionnels?)?|projects?|مشاريع?|"
    r"certifications?|certifiés?|شهادات?|"
    r"langues?|languages?|لغات?|"
    r"centres?\s+d'intérêt|interests?|hobbies?|اهتمامات?|"
    r"références?|references?|مراجع?"
    r")[\s:*\-]*$",
    re.IGNORECASE | re.MULTILINE
)

CV_EXPERIENCE_ENTRY_PATTERNS = re.compile(
    r"(?:"
    # Dates en début de ligne
    r"(?:^|\n)\s*(?:\d{4}|\w+\s+\d{4})\s*[-–—/|]\s*(?:\d{4}|present|aujourd'hui|current|الآن|حتى الآن)"
    r"|"
    # Nom d'entreprise suivi d'un poste (ligne courte en majuscules ou suivie de •|-)
    r"(?:^|\n)\s*[A-Z\u0600-\u06FF][A-Za-z\u0600-\u06FF\s&,.'-]{3,50}\s*(?:\||•|-)\s*[A-Z\u0600-\u06FF]"
    r")",
    re.MULTILINE
)

JD_SECTION_PATTERNS = re.compile(
    r"^(?:"
    r"requirements?|required(?:\s+skills?)?|prérequis?|qualifications?\s+requises?|متطلبات?|"
    r"responsibilities?|missions?|tâches?|rôle|مهام?|مسؤوليات?|"
    r"nice\s+to\s+have|preferred(?:\s+qualifications?)?|bonus|un\s+plus|ميزة إضافية|"
    r"about\s+(?:the\s+)?(?:role|job|position|company|us)|à\s+propos|عن\s+(?:الدور|الشركة)|"
    r"benefits?|avantages?|what\s+we\s+offer|مزايا?|"
    r"tech(?:nical)?\s+stack|technologies?|stack|تقنيات?|"
    r"experience\s+(?:required|level)|niveau\s+d'expérience|مستوى\s+الخبرة"
    r")[\s:*\-]*$",
    re.IGNORECASE | re.MULTILINE
)

# Header contextuel préfixé aux sous-chunks d'expérience
CV_SUBCHUNK_PREFIX_MAX_WORDS = 15
# Whether to use semantic chunking (False = legacy fixed-size).
USE_SEMANTIC_CHUNKING: bool = True

# =============================================================================
# PROMPTING UPGRADE: English system prompt + strict output language gate
# =============================================================================

BASE_SYSTEM_PROMPT = """
You are a senior hiring manager conducting a structured, high-stakes interview.
You must follow instructions precisely.

Your job is to produce the recruiter's NEXT spoken turn only.
You ask exactly one focused follow-up question unless explicitly instructed otherwise.
You do not praise the candidate.
You do not add filler.
You do not summarize the candidate's answer first.
You do not use markdown, bullets, or labels in the final answer.
Return plain text only.
""".strip()

OUTPUT_LANGUAGE_GATES: Dict[str, str] = {
    "Arabe": (
        "OUTPUT LANGUAGE GATE — Your ENTIRE response must be written in Saudi Arabic dialect. "
        "Technical English terms are allowed only when necessary."
    ),
    "Français": (
        "OUTPUT LANGUAGE GATE — Your ENTIRE response must be written in French. "
        "Technical English terms are allowed only when necessary."
    ),
    "Anglais": (
        "OUTPUT LANGUAGE GATE — Your ENTIRE response must be written in English."
    ),
}

FEW_SHOT_EXAMPLES: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "Arabe": {
        "OPENING": [
            {
                "recruiter_question": "عرّفني بسرعة على أكثر تجربة في مسارك مرتبطة بهذا الدور.",
                "candidate_answer": "اشتغلت على منصة بيانات داخلية وبنيت pipelines لمعالجة البيانات وتحسين جودة التقارير.",
                "your_follow_up": "وش القرار التقني الأهم اللي أخذته بنفسك في هالمنصة، وليش اخترته؟",
            }
        ],
        "TECHNICAL_DEPTH": [
            {
                "recruiter_question": "وش النتيجة الرقمية اللي تقدر تدافع عنها من هذا العمل؟",
                "candidate_answer": "خفّضنا زمن التنفيذ تقريباً 35% بعد إعادة تصميم الـ pipeline.",
                "your_follow_up": "كيف قست هذا الـ 35% بالضبط، وما كان الـ baseline قبل التغيير؟",
            }
        ],
        "PROJECT_DEEP_DIVE": [
            {
                "recruiter_question": "وش أصعب تحدٍ تقني واجهته في هذا المشروع؟",
                "candidate_answer": "كان عندنا bottleneck في ingestion وصار فيه تأخير واضح تحت الضغط.",
                "your_follow_up": "خذني لأول قرار اتخذته لما اكتشفت الـ bottleneck، وش البدائل اللي استبعدتها؟",
            }
        ],
        "SOFT_SKILLS_BEHAVIORAL": [
            {
                "recruiter_question": "أعطني مثالاً محدداً على خلاف تقني مع زميل.",
                "candidate_answer": "اختلفت مع مهندس ثاني حول استخدام queue أو direct calls بين الخدمات.",
                "your_follow_up": "كيف دفعت النقاش لقرار عملي، وإذا رجع الوقت هل كنت ستتعامل معه بنفس الطريقة؟",
            }
        ],
    },
    "Français": {
        "TECHNICAL_DEPTH": [
            {
                "recruiter_question": "Quel résultat chiffré pouvez-vous défendre sur ce travail ?",
                "candidate_answer": "Nous avons réduit la latence d'environ 40% après refonte du pipeline.",
                "your_follow_up": "Comment avez-vous mesuré ces 40%, et quelle était la baseline exacte ?",
            }
        ]
    },
    "Anglais": {
        "TECHNICAL_DEPTH": [
            {
                "recruiter_question": "What concrete metric can you defend from that project?",
                "candidate_answer": "We cut processing time by roughly 40% after redesigning the pipeline.",
                "your_follow_up": "How exactly did you measure that 40%, and what baseline are you comparing against?",
            }
        ]
    },
}

# =============================================================================
# FIX-AR-9: French/English token patterns to detect drift in Arabic output
# =============================================================================
FRENCH_DRIFT_PATTERN = re.compile(
    r"\b(le|la|les|un|une|des|je|vous|nous|vous|ils|est|sont|avez|avons"
    r"|pour|dans|sur|avec|par|que|qui|quand|comment|pourquoi|quel|quelle"
    r"|voici|passons|donnez|dites|décrivez|parlez|racontez|expliquez"
    r"|très|bien|merci|bonjour|d'accord|donc|mais|aussi|encore|toujours"
    r"|pouvez|voulez|devez|faut|peut|doit|font|fait|être|avoir|faire)\b",
    re.IGNORECASE
)

ENGLISH_DRIFT_PATTERN = re.compile(
    r"\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should"
    r"|tell|walk|give|describe|help|what|how|why|when|where|who|which"
    r"|your|you|me|my|our|we|they|it|this|that|these|those"
    r"|and|or|but|so|if|then|because|while|after|before|during"
    r"|great|good|noted|thank|sure|absolutely|certainly|of course)\b",
    re.IGNORECASE
)

# =============================================================================
# IMPROVEMENT F: Skill synonym normalisation table
# =============================================================================
SKILL_SYNONYMS: Dict[str, List[str]] = {}

# =============================================================================
# IMPROVEMENT E: Human opener bank (40+ openers, multilingual)
# =============================================================================
HUMAN_OPENERS = {
    "Anglais": [
        "On that — ", "Let me push on that — ", "And concretely — ",
        "I want to dig into that — ", "So on that point — ",
        "That raises something — ", "Let's go one level deeper — ",
        "Walk me through — ", "Help me understand — ",
        "What I'm curious about — ", "Tell me specifically — ",
        "I want to understand your thinking — ", "One thing I'd like to probe — ",
        "Building on that — ", "Let's zoom in — ",
        "Put me in that moment — ", "Give me the detail on — ",
        "What I'd like to understand — ", "The thing I'm missing — ",
        "Let's be specific — ", "On the technical side — ",
        "From your perspective — ", "In your own words — ",
        "Here's what I want to know — ", "Before we move on — ",
        "To make this concrete — ", "Take me back to — ",
        "Let's talk about — ", "What I want to pressure-test — ",
        "Stepping back slightly — ", "One angle I want to cover — ",
        "And the harder question — ", "Specifically on — ",
        "I'd like to challenge that — ", "Let's test that — ",
    ],
    "Français": [
        "Sur ce point — ", "Ce qui m'intéresse — ", "Allons plus loin — ",
        "Je veux creuser — ", "Concrètement — ", "Soyons précis — ",
        "Dites-moi exactement — ", "Ce que je veux comprendre — ",
        "Sur le plan technique — ", "Revenons à — ",
        "Avant d'avancer — ", "Ce qui me manque — ",
        "Allons un niveau plus bas — ", "Mettez-moi dans ce moment — ",
        "Sur votre raisonnement — ", "Une chose qui m'intéresse — ",
        "Pour être précis — ", "Et la question difficile — ",
        "Testez cette affirmation — ", "Ce que je veux valider — ",
        "Parlez-moi précisément de — ", "Creusons sur — ",
        "Un angle que je veux couvrir — ", "Sur ce sujet — ",
        "Ce qui m'échappe — ", "Retournons sur — ",
        "Un détail que je veux comprendre — ", "Pour rendre ça concret — ",
        "Avant de passer à autre chose — ", "Sur la partie technique — ",
        "De votre point de vue — ", "Et pour challenger — ",
        "Ce qui me pose question — ", "Restons sur ce point — ",
        "Allons au fond de — ",
    ],
    "Arabe": [
        "على هذه النقطة — ", "ما يشدّ انتباهي — ", "خلّنا نعمّق — ",
        "بالتحديد — ", "أريد أن أفهم — ", "فلنكن واضحين — ",
        "أخبرني بالضبط — ", "ما يهمني هنا — ",
        "من ناحية تقنية — ", "نرجع إلى — ",
        "قبل أن ننتقل — ", "ما يغيب عني — ",
        "نزل معي خطوة — ", "ضعني في ذلك الموقف — ",
        "على صعيد التفكير — ", "شيء يثير اهتمامي — ",
        "لنكن دقيقين — ", "والسؤال الأصعب — ",
        "تحدّ هذه النقطة — ", "ما أريد التحقق منه — ",
        "حدثني تحديداً عن — ", "نعمّق في — ",
        "زاوية أريد تغطيتها — ", "على هذا الموضوع — ",
        "ما يفوتني — ", "نرجع إلى — ",
        "تفصيلة أريد فهمها — ", "لجعل هذا ملموساً — ",
        "قبل الانتقال — ", "على الجانب التقني — ",
        "من منظورك — ", "للتحدي — ",
        "ما يثير تساؤلاً — ", "نبقى على هذه النقطة — ",
        "نصل لعمق — ",
    ],
}

# =============================================================================
# IMPROVEMENT B: Weak signal detection patterns
# =============================================================================
WEAK_SIGNAL_PATTERNS = {
    "hedged_ownership": [
        r"\b(i was (?:involved|part of|contributing|helping)|"
        r"i (?:helped|assisted|supported|contributed to)|"
        r"we (?:worked on|built|developed|did)|"
        r"(?:j'ai aidé|j'étais impliqué|on a fait|on a travaillé)|"
        r"(?:كنت جزءاً من|ساعدت في|شاركت في|عملنا على))\b"
    ],
    "vague_impact": [
        r"\b(improved (?:performance|efficiency|quality)|"
        r"reduced (?:cost|time|latency)|"
        r"increased (?:speed|quality|output)|"
        r"(?:amélioré|réduit|augmenté) (?:la performance|le temps|la qualité)|"
        r"(?:حسّنا|قللنا|رفعنا))\b(?!\s*(?:by|de|بنسبة)\s*\d)"
    ],
    "understatement": [
        r"\b(kind of|sort of|more or less|somewhat|a bit|a little|"
        r"roughly|approximately|around|basically|essentially|"
        r"plutôt|un peu|à peu près|globalement|en gros|"
        r"نوعاً ما|تقريباً|بشكل عام|إلى حد ما)\b"
    ],
    "unanchored_comparison": [
        r"\b(much (?:faster|better|cheaper|simpler|more efficient)|"
        r"significantly (?:faster|better|cheaper|improved)|"
        r"(?:beaucoup plus|nettement|bien plus) (?:rapide|efficace|performant)|"
        r"(?:أسرع بكثير|أفضل بكثير|أكثر كفاءة بكثير))\b(?!\s*(?:than|que|من)\s*\d)"
    ],
}

# =============================================================================
# IMPROVEMENT D: Coverage matrix structure
# =============================================================================
COVERAGE_QUADRANTS = ["TECH", "PROJECT", "BEHAVIORAL", "DECISION"]

# =============================================================================
# INTERVIEW PHASES
# =============================================================================
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


# AJOUTER ICI
BLOCK_A_PHASES = [
    "OPENING",
    "JOB_ALIGNED_EXPLORATION",
    "PROJECT_DEEP_DIVE",
    "TECHNICAL_DEPTH",
]

BLOCK_B_PHASES = [
    "SOFT_SKILLS_BEHAVIORAL",
    "CANDIDATE_QUESTIONS",
    "FINAL_CHECK",
    "CLOSING",
]

BLOCK_A_RATIO = 2 / 3
BLOCK_B_RATIO = 1 / 3

BLOCK_B_RESERVED_MINUTES = {
    "SOFT_SKILLS_BEHAVIORAL": 0.45,
    "CANDIDATE_QUESTIONS": 0.35,
    "FINAL_CHECK": 0.10,
    "CLOSING": 0.10,
}

MIN_PHASE_TURNS = {...}
MAX_PHASE_TURNS = {...}

MIN_PHASE_TURNS = {
    "OPENING":                 1,
    "JOB_ALIGNED_EXPLORATION": 3,
    "PROJECT_DEEP_DIVE":       3,
    "TECHNICAL_DEPTH":         3,
    "SOFT_SKILLS_BEHAVIORAL":  2,
    "CANDIDATE_QUESTIONS":     1,
    "FINAL_CHECK":             1,
    "CLOSING":                 0,
}

MAX_PHASE_TURNS = {
    "OPENING":                 2,
    "JOB_ALIGNED_EXPLORATION": 5,
    "PROJECT_DEEP_DIVE":       6,
    "TECHNICAL_DEPTH":         6,
    "SOFT_SKILLS_BEHAVIORAL":  4,
    "CANDIDATE_QUESTIONS":     3,
    "FINAL_CHECK":             2,
    "CLOSING":                 0,
}

LANG_ALIASES = {
    "fr": "Français", "francais": "Français", "français": "Français", "french": "Français",
    "en": "Anglais",  "english": "Anglais",   "anglais": "Anglais",
    "ar": "Arabe",    "arabic": "Arabe",       "arabe": "Arabe",
    "sa": "Arabe",    "saudi": "Arabe",
}

ANSWER_QUALITY_THRESHOLDS = {
    "STRONG":     {"min_words": 60, "min_skills": 2, "min_metrics": 1},
    "GOOD":       {"min_words": 30, "min_skills": 1, "min_metrics": 0},
    "INCOMPLETE": {"min_words": 12, "min_skills": 0, "min_metrics": 0},
}

METRICS_PROBE_AFTER_TURNS = 3

BEHAVIORAL_QUESTIONS = {
    "conflict": {
        "Français": "Racontez-moi un désaccord technique ou humain concret que vous avez eu — qu'est-ce qui s'est passé, ce que vous avez fait, et quel en a été le résultat ?",
        "Anglais":  "Walk me through a real conflict — technical or interpersonal — you had with someone at work. What happened, what did you do, and how did it end?",
        "Arabe":    "خذني بالتفصيل لخلاف حقيقي — تقني أو شخصي — واجهته مع أحد في العمل. إيش صار، وش اللي سويته، وكيف انتهى؟",
    },
    "prioritization": {
        "Français": "Décrivez un moment où vous aviez trop à faire et pas assez de temps — comment avez-vous décidé quoi laisser de côté ?",
        "Anglais":  "Tell me about a time you had more work than you could handle — how did you decide what to drop or defer?",
        "Arabe":    "حدثني عن وقت كان عندك شغل أكثر مما تقدر تتحمل — كيف قررت إيش تترك أو تأجل؟",
    },
    "failure": {
        "Français": "Parlez-moi d'une décision technique que vous avez prise et qui n'a pas marché — qu'est-ce qui s'est passé et qu'est-ce que ça vous a appris sur vous-même ?",
        "Anglais":  "Tell me about a technical decision you made that backfired — what happened and what did it reveal about how you work?",
        "Arabe":    "أخبرني عن قرار تقني اتخذته وكان خاطئاً — إيش صار وإيش كشف لك عن نفسك؟",
    },
    "leadership": {
        "Français": "Décrivez une situation où vous deviez faire avancer quelque chose sans avoir l'autorité formelle pour l'imposer. Comment vous l'avez géré ?",
        "Anglais":  "Tell me about a time you had to get something done without the authority to impose it. How did you make it happen?",
        "Arabe":    "احكيلي عن وقت كنت محتاج تنجز شيء بدون صلاحية رسمية تفرضه. كيف نجحت؟",
    },
    "pressure": {
        "Français": "Racontez-moi la dernière fois où vous étiez vraiment sous pression — deadline intenable, incident critique — et ce que vous avez fait concrètement.",
        "Anglais":  "Walk me through the last time you were genuinely under pressure — impossible deadline, production fire — and what you actually did.",
        "Arabe":    "خذني لآخر مرة كنت فعلاً تحت ضغط — deadline مستحيل، حادثة في الـ production — وإيش اللي سويته بالضبط؟",
    },
}

DECISION_MAKING_QUESTIONS = {
    "tradeoff": {
        "Français": "Sur ce projet, quel trade-off technique avez-vous dû faire — et comment avez-vous décidé où tracer la ligne ?",
        "Anglais":  "On that project, what technical trade-off did you face, and how did you decide where to draw the line?",
        "Arabe":    "في هذا المشروع، وش trade-off تقني واجهته، وكيف قررت أين ترسم الحد؟",
    },
    "why_technology": {
        "Français": "Pourquoi cette technologie et pas une autre — qui a pris la décision, sur quelle base, et qu'est-ce qui a été rejeté ?",
        "Anglais":  "Why that specific technology and not an alternative — who made the call, on what basis, and what got ruled out?",
        "Arabe":    "ليش هذه التقنية بالذات وليس بديلاً آخر — من اتخذ القرار وعلى أي أساس وإيش تم رفضه؟",
    },
    "rollback": {
        "Français": "Si vous refaisiez ce projet aujourd'hui, quelle décision changeriez-vous en premier et pourquoi ?",
        "Anglais":  "If you redid that project from scratch today, what's the first decision you'd change and why?",
        "Arabe":    "لو تعيد هذا المشروع من الصفر اليوم، وش أول قرار تغيره ولماذا؟",
    },
    "incident_decision": {
        "Français": "Décrivez le dernier incident en production que vous avez traité — quelle a été votre première décision, et avec le recul, était-elle juste ?",
        "Anglais":  "Describe the last production incident you personally handled — what was your first call, and in hindsight, was it right?",
        "Arabe":    "صف لي آخر incident في الـ production تعاملت معه بنفسك — وش أول قرار اتخذته وهل كان صحيحاً بالنظر للخلف؟",
    },
}

PROBE_DEPTH_CONFIG = {
    "max_depth_per_topic":        3,
    "min_depth_before_switch":    1,
    "challenge_threshold":        "GOOD",
    "signal_validation_enabled":  True,
    "behavioral_min_turns":       2,
    "decision_probe_after_turns": 2,
}

FORBIDDEN_PRAISE_TOKENS = [
    "great answer", "excellent answer", "perfect answer", "impressive answer",
    "wonderful", "fantastic", "brilliant", "outstanding", "superb",
    "très bonne réponse", "excellente réponse", "parfait", "impressionnant",
    "إجابة ممتازة", "رائع", "ممتاز",
    "noted", "noted.", "good.", "good,", "thank you for sharing",
    "thank you for that", "that's interesting", "interesting.",
    "i see", "i understand", "understood",
    "très bien", "bien noté", "je vois", "je comprends", "d'accord,",
    "c'est intéressant", "merci pour ce partage",
    "حسناً", "فهمت", "شكراً على هذا",
    "absolutely", "certainly", "of course", "sure,", "sure.",
    "definitely", "great,", "great.", "perfect,", "perfect.",
    "bien sûr", "évidemment", "absolument",
    "بالتأكيد", "طبعاً",
    "here's my first question", "here is my first question",
    "here's my next question", "here is my next question",
    "here's a question", "here is a question",
    "let's get started with the interview",
    "let's begin with", "let's start with",
    "on that note,", "on that note —",
    "with that in mind,", "with that,",
    "my first question is", "my next question is",
    "moving on,", "moving forward,",
    "voici ma première question", "voici ma question",
    "passons à ma première question", "passons à la suite,",
    "السؤال الأول", "إليك سؤالي",
]

INTENT_VERB_CATEGORIES = {
    "why_choice":     ["why", "pourquoi", "ليش", "rationale", "reason"],
    "how_measured":   ["measured", "metric", "baseline", "mesur", "قياس"],
    "walk_through":   ["walk me through", "describe", "tell me about", "décrivez", "احكيلي"],
    "challenge":      ["challenge", "critique", "alternative", "instead", "بدلاً"],
    "ownership":      ["personally", "yourself", "your role", "you specifically", "شخصياً"],
    "failure":        ["failed", "wrong", "mistake", "backfire", "فشل"],
    "impact":         ["impact", "outcome", "result", "effect", "نتيجة"],
}

# =============================================================================
# UTILITIES
# =============================================================================

def _safe_read_text(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        from docx import Document
        doc = Document(str(p))
        return "\n".join(par.text for par in doc.paragraphs if par.text.strip())
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported document extension: {ext}")


def _normalize_lang(lang: str) -> str:
    if not lang:
        return "Français"
    key = str(lang).strip().lower().replace("-", "_").replace(" ", "_")
    return LANG_ALIASES.get(key, LANG_ALIASES.get(key.replace("_", ""), "Français"))


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[""\"'`´']", "", text)
    text = re.sub(r"[\u200f\u200e]", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF\-+.#/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


_embeddings_singleton = None
_cross_encoder_singleton = None
_bm25_available: Optional[bool] = None


def _get_embeddings():
    """
    Lazy singleton SentenceTransformer.
    RTX 2050 : chargé sur CUDA (float16) — libère ~400 Mo VRAM.
    CPU fallback automatique si CUDA absent.
    """
    global _embeddings_singleton
    if _embeddings_singleton is None:
        from sentence_transformers import SentenceTransformer
        import torch as _torch
        st_device = DEVICE_CONFIG["st_model_device"]
        dtype_map  = {"float16": _torch.float16, "float32": _torch.float32}
        torch_dtype = dtype_map.get(DEVICE_CONFIG["torch_dtype"], _torch.float32)

        print(f"[RAG] Chargement SentenceTransformer → device={st_device}  dtype={DEVICE_CONFIG['torch_dtype']}")
        try:
            _embeddings_singleton = SentenceTransformer(
                EMBED_MODEL,
                cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
                device=st_device,
                # model_kwargs disponible depuis sentence-transformers 2.7+
                model_kwargs={"torch_dtype": torch_dtype} if st_device == "cuda" else {},
            )
            print(f"[RAG] SentenceTransformer OK  ({st_device})")
        except TypeError:
            # Ancienne version sentence-transformers : sans model_kwargs
            _embeddings_singleton = SentenceTransformer(
                EMBED_MODEL,
                cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
                device=st_device,
            )
            print(f"[RAG] SentenceTransformer OK  ({st_device}, legacy init)")
        except Exception as e:
            print(f"[RAG] ⚠️  SentenceTransformer CUDA échoué ({e}) → CPU fallback")
            _embeddings_singleton = SentenceTransformer(
                EMBED_MODEL,
                cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
                device="cpu",
            )
    return _embeddings_singleton


def _get_cross_encoder():
    """
    RAG-IMPROVEMENT-2: Lazy singleton cross-encoder re-ranker.
    RTX 2050 : cross-encoder/ms-marco-MiniLM-L-6-v2 tient aisément en VRAM.
    Désactivé si DEVICE_CONFIG['disable_cross_encoder'] = True.
    """
    global _cross_encoder_singleton
    if _cross_encoder_singleton is None:
        # Respect de l'override profil (lite / env USE_CROSS_ENCODER=0)
        if DEVICE_CONFIG.get("disable_cross_encoder", False):
            print("[RAG] CrossEncoder désactivé (profil GPU)")
            _cross_encoder_singleton = False
            return None

        try:
            from sentence_transformers import CrossEncoder
            ce_device = DEVICE_CONFIG["cross_encoder_device"]
            print(f"[RAG] Chargement CrossEncoder → device={ce_device}")
            _cross_encoder_singleton = CrossEncoder(
                RERANKER_MODEL,
                max_length=512,
                device=ce_device,
            )
            print(f"[RAG] CrossEncoder OK  ({ce_device})")
        except Exception as e:
            print(f"WARNING _get_cross_encoder: {e} — cross-encoder disabled.")
            _cross_encoder_singleton = False  # sentinel: failed
    return _cross_encoder_singleton if _cross_encoder_singleton is not False else None


def _check_bm25() -> bool:
    """RAG-IMPROVEMENT-1: Check BM25 availability at runtime."""
    global _bm25_available
    if _bm25_available is None:
        try:
            from rank_bm25 import BM25Okapi
            _bm25_available = True
        except ImportError:
            _bm25_available = False
            print("INFO: rank_bm25 not installed — falling back to dense-only search. "
                  "Install with: pip install rank-bm25")
    return _bm25_available


def _tokenize_for_bm25(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"[a-zA-Z\u0600-\u06FF0-9]+", text.lower())


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
@dataclass
class ChunkRecord:
    chunk_id: str
    source_type: str
    source_name: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    # Nouveaux champs de section
    section: str = ""          # ex: "experience", "skills", "requirements"
    entity: str = ""           # ex: "Entreprise X", "Poste Y"
    date_range: str = ""       # ex: "2021-2023"


@dataclass
class TurnRecord:
    phase: str
    speaker: str
    text: str
    timestamp: str
    emotion: str = "neutre"
    answer_quality: str = "N/A"
    signal_validated: bool = False
    behavioral_story_detected: bool = False
    decision_reasoning_detected: bool = False
    weak_signals_detected: List[str] = field(default_factory=list)


# =============================================================================
# VECTOR STORE — V3: Hybrid BM25 + Dense with Cross-Encoder Re-ranking
# =============================================================================

class SimpleVectorStore:
    """
    RAG-IMPROVEMENT-1: Maintains a BM25 index alongside dense embeddings.
    RAG-IMPROVEMENT-2: Re-ranks fused results with a cross-encoder.
    """

    def __init__(self):
        self.chunks: List[ChunkRecord] = []
        # BM25 index (rebuilt lazily on first search after any add_chunks)
        self._bm25_index = None
        self._bm25_dirty: bool = False  # True when chunks changed since last index build

    # ── Index management ────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        self.chunks.extend(chunks)
        self._bm25_dirty = True  # invalidate BM25 index

    def has_source_type(self, source_type: str) -> bool:
        return any(ch.source_type == source_type for ch in self.chunks)

    def _rebuild_bm25(self) -> None:
        """RAG-IMPROVEMENT-1: (Re)build BM25Okapi index from all chunks."""
        if not _check_bm25():
            return
        try:
            from rank_bm25 import BM25Okapi
            corpus = [_tokenize_for_bm25(ch.text) for ch in self.chunks]
            self._bm25_index = BM25Okapi(corpus)
            self._bm25_dirty = False
        except Exception as e:
            print(f"WARNING _rebuild_bm25: {e}")
            self._bm25_index = None

    # ── Dense vector search ──────────────────────────────────────────────────

    def _dense_search(
        self,
        query: str,
        top_k: int,
        source_boost: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[float, int]]:
        """Returns (score, chunk_index) sorted descending."""
        if not self.chunks:
            return []
        model = _get_embeddings()
        q = model.encode([query], normalize_embeddings=True)[0]
        scored = []
        for idx, ch in enumerate(self.chunks):
            if ch.embedding is None:
                continue
            score = _cosine(q, ch.embedding)
            if source_boost and ch.source_type in source_boost:
                score *= source_boost[ch.source_type]
            scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    # ── BM25 keyword search ──────────────────────────────────────────────────

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        source_boost: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[float, int]]:
        """RAG-IMPROVEMENT-1: Returns (score, chunk_index) sorted descending."""
        if not USE_BM25 or not _check_bm25():
            return []
        if self._bm25_dirty or self._bm25_index is None:
            self._rebuild_bm25()
        if self._bm25_index is None:
            return []
        try:
            tokens = _tokenize_for_bm25(query)
            raw_scores = self._bm25_index.get_scores(tokens)
            scored = []
            for idx, score in enumerate(raw_scores):
                if source_boost and idx < len(self.chunks):
                    ch = self.chunks[idx]
                    if ch.source_type in source_boost:
                        score *= source_boost[ch.source_type]
                scored.append((float(score), idx))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]
        except Exception as e:
            print(f"WARNING _bm25_search: {e}")
            return []

    # ── Reciprocal Rank Fusion ───────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        dense_ranked: List[Tuple[float, int]],
        bm25_ranked: List[Tuple[float, int]],
        alpha: float = HYBRID_ALPHA,
        k: int = RRF_K,
        ) -> List[Tuple[float, int]]:
        """
        Fuse two ranked lists with Reciprocal Rank Fusion.
        Returns (fused_score, chunk_index) sorted descending.
        """
        fused: Dict[int, float] = {}

        # Dense contribution
        for rank, (_, idx) in enumerate(dense_ranked):
            fused[idx] = fused.get(idx, 0.0) + alpha * (1.0 / (k + rank + 1))

        # BM25 contribution
        for rank, (_, idx) in enumerate(bm25_ranked):
            fused[idx] = fused.get(idx, 0.0) + (1.0 - alpha) * (1.0 / (k + rank + 1))

        # IMPORTANT: convertir (idx, score) -> (score, idx)
        return sorted(
            [(score, idx) for idx, score in fused.items()],
            key=lambda x: x[0],
            reverse=True,
        )

    # ── Cross-encoder re-ranking ─────────────────────────────────────────────

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Tuple[float, int]],
        top_k: int,
    ) -> List[ChunkRecord]:
        if not USE_CROSS_ENCODER:
            return [self.chunks[int(idx)] for _, idx in candidates[:top_k] if 0 <= int(idx) < len(self.chunks)]

        encoder = _get_cross_encoder()
        if encoder is None:
            return [self.chunks[int(idx)] for _, idx in candidates[:top_k] if 0 <= int(idx) < len(self.chunks)]

        rerank_pool = []
        for score, idx in candidates[:RERANKER_TOP_K]:
            try:
                idx = int(idx)
            except Exception:
                continue
            if 0 <= idx < len(self.chunks):
                rerank_pool.append((float(score), idx))

        if not rerank_pool:
            return []

        pairs = [(query, self.chunks[idx].text[:512]) for _, idx in rerank_pool]

        try:
            ce_scores = encoder.predict(pairs)
            scored = sorted(
                zip(ce_scores, [idx for _, idx in rerank_pool]),
                key=lambda x: x[0],
                reverse=True,
            )
            return [self.chunks[idx] for _, idx in scored[:top_k]]
        except Exception as e:
            print(f"WARNING _cross_encoder_rerank: {e} — using fusion order.")
            return [self.chunks[idx] for _, idx in rerank_pool[:top_k]]

    # ── Public search interface ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        source_boost: Optional[Dict[str, float]] = None,
    ) -> List[ChunkRecord]:
        """
        RAG-IMPROVEMENT-1+2: Hybrid BM25+vector search, then cross-encoder re-rank.
        Gracefully degrades: cross-encoder → fusion only → dense only.
        """
        if not self.chunks:
            return []

        dense_results = self._dense_search(query, top_k=top_k * 2, source_boost=source_boost)

        if USE_BM25 and _check_bm25():
            bm25_results = self._bm25_search(query, top_k=top_k * 2, source_boost=source_boost)
            fused = self._rrf_fuse(dense_results, bm25_results)
        else:
            # Fall back to dense-only (no RRF needed)
            fused = [(score, idx) for score, idx in dense_results]

        return self._cross_encoder_rerank(query, fused, top_k=top_k)


# =============================================================================
# SEMANTIC SENTENCE SPLITTER (RAG-IMPROVEMENT-3 helper)
# =============================================================================

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?؟])\s+(?=[A-Z\u0600-\u06FF\u00C0-\u024F])"
    r"|(?<=\n)\s*(?=[A-Z\u0600-\u06FF\u00C0-\u024F])"
    r"|(?<=[:;])\s{2,}"
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a multilingual heuristic regex."""
    raw = _SENTENCE_SPLIT_RE.split(text)
    sentences = []
    for s in raw:
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences if sentences else [text]


# =============================================================================
# MAIN CLASS
# =============================================================================

class HRInteractiveBrain:

    def _select_model(self, normalized_lang: str) -> str:
        return LLAMA_MODEL 

    def __init__(self, target_lang: str = "fr", duration_minutes: int = 20):
        from recruiter_translator_agent import OrchestrateurTraducteur

        self.client = Client()                                    # ← EN PREMIER
        self.target_lang = _normalize_lang(target_lang)
        self.duration_minutes = int(duration_minutes)
        self.model_name = self._select_model(self.target_lang)

        self._ot_agent = OrchestrateurTraducteur(self.client, self.model_name)  # ← APRÈS client et model_name

        self.steps = KNOWN_PHASES.copy()
        self.current_step_index = 0
        # ... reste inchangé
                # =========================================================
        # TIME-BUDGET CONTROL (2/3 TECH + 1/3 SOFT + QUESTIONS)
        # =========================================================

        # Global time split
        self.block_a_budget_minutes = round(self.duration_minutes * BLOCK_A_RATIO, 2)
        self.block_b_budget_minutes = round(self.duration_minutes * BLOCK_B_RATIO, 2)

        # Candidate questions flow control
        self.candidate_questions_started = False
        self.candidate_questions_completed = False
        self.final_check_completed = False

        # Loop control for Q&A
        self._candidate_question_rounds = 0
        self._max_candidate_question_rounds = 3
        self.started_at = time.time()
        self.ended = False

        self.vector_store = SimpleVectorStore()
        self.ingested_docs: Dict[str, List[str]] = {"cv": [], "job_offer": [], "company_info": []}
        self.doc_texts: Dict[str, str] = {"cv": "", "job_offer": "", "company_info": ""}
        self.pending_chunks: Dict[str, List[ChunkRecord]] = {
            "cv": [],
            "job_offer": [],
            "company_info": [],
        }

        self.turns: List[TurnRecord] = []
        self.conversation_history: List[Dict[str, str]] = []

        self.last_questions_normalized: List[str] = []
        self.asked_question_signatures: Set[str] = set()
        self._fallback_counter: Dict[str, int] = {p: 0 for p in KNOWN_PHASES}

        # IMPROVEMENT C
        self._question_ngram_fingerprints: Set[frozenset] = set()
        self._intent_verb_targets: Set[Tuple[str, str]] = set()

        self.covered_topics: set = set()
        self.covered_projects: set = set()
        self.covered_skills: set = set()
        self.explored_jd_tools: set = set()
        self.deep_dived_projects: set = set()
        self.behavioral_asked: set = set()

        self.followup_depth = 0
        self._topic_followup_count: Dict[str, int] = {}
        self._max_topic_followups: int = 3  # max questions sur un même topic hors JD_EXPLORATION
        self.current_topic_focus = ""
        self.current_focus_project = ""
        self.current_focus_skill = ""

        self.vision_stress_flag = False
        self.vision_emotion_label = "neutre"

        self._answer_quality_history: List[str] = []

        self._internal_scores = {
            "clarte": 0, "structure": 0, "technicite": 0,
            "pertinence": 0, "motivation": 0,
        }
        self.scores = {p: 0 for p in KNOWN_PHASES if p not in ("FINAL_CHECK", "CLOSING")}
        self._hs_score: int = 0
        self._ss_score: int = 0
        self._hs_count: int = 0
        self._ss_count: int = 0

        self._dynamic_skill_keywords: set = set()
        self._jd_tools: List[str] = []

        self._jd_tool_why_asked: Set[str] = set()

        self._tool_followup_count: Dict[str, int] = {}
        self._current_tool_focus: str = ""
        self._tool_rotation_order: List[str] = []
        self._tools_fully_done: Set[str] = set()
        self._tool_project_map: Dict[str, List[str]] = {}

        self._unvalidated_claims: List[Dict[str, str]] = []
        self._challenged_claim_types: set = set()

        self._metrics_probed: bool = False
        self._turns_since_metrics_probe: int = 0

        self._behavioral_coverage: Dict[str, bool] = {k: False for k in BEHAVIORAL_QUESTIONS}
        self._behavioral_stories_count: int = 0
        self._behavioral_asked: set = set()

        self._decision_coverage: Dict[str, bool] = {k: False for k in DECISION_MAKING_QUESTIONS}
        self._decision_asked: set = set()
        self._decision_shown_per_phase: Dict[str, bool] = {p: False for p in KNOWN_PHASES}

        self._job_required_skills: List[str] = []
        self._job_required_behaviors: List[str] = []
        self._job_seniority_level: str = "mid"
        self._job_domain: str = ""
        self._job_domain_directive: str = ""
        self._job_topic_keywords: Dict[str, List[str]] = {}

        self._leadership_signals: List[str] = []
        self._leadership_score: int = 0

        self.candidate_questions_mode = False
        self._candidate_declined_questions: bool = False

        self.session_notes = {
            "candidate_name": "",
            "key_projects": [],
            "matched_skills": [],
            "missing_skills": [],
            "strengths": [],
            "weaknesses": [],
        }

        # IMPROVEMENT A
        self._vague_retry_count: Dict[str, int] = {}
        self._current_vague_topic: str = ""
        self._relance_tier_used: Set[str] = set()

        # IMPROVEMENT B
        self._weak_signal_queue: List[Dict[str, str]] = []
        self._probed_weak_signals: Set[str] = set()

        # IMPROVEMENT D
        self._coverage_matrix: Dict[str, Dict[str, bool]] = {
            "TECH": {},
            "PROJECT": {},
            "BEHAVIORAL": {k: False for k in BEHAVIORAL_QUESTIONS},
            "DECISION": {k: False for k in DECISION_MAKING_QUESTIONS},
        }

        # IMPROVEMENT E
        self._opener_index: int = 0
        self._last_opener_used: str = ""

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = str(LOGS_DIR / f"interview_{ts}.txt")
        self._write_log_header()

    # =========================================================================
    # Public API
    # =========================================================================

    def ingest_document(self, path: str, source_type: str, build_embeddings: bool = False) -> None:
        source_type = source_type.strip()
        if source_type not in self.ingested_docs:
            raise ValueError(f"Unknown source_type: {source_type}")

        text = _safe_read_text(path)
        text = self._clean_text(text)

        self.ingested_docs[source_type].append(Path(path).name)
        self.doc_texts[source_type] = (
            self.doc_texts.get(source_type, "") + "\n\n" + text
        ).strip()

        # RAG-IMPROVEMENT-3: use semantic chunking instead of fixed-size
        chunks = self._semantic_chunk_text(text, source_type=source_type, source_name=Path(path).name)
        if chunks:
            self.pending_chunks[source_type].extend(chunks)
            if build_embeddings:
                self.ensure_embeddings_ready(source_type=source_type)

        self._refresh_static_analysis()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def _embed_cache_path(self, source_type: str, source_name: str, text: str) -> Path:
        digest = self._hash_text(text)
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", source_name)
        return EMBED_CACHE_DIR / f"{source_type}__{safe_name}__{digest}.pkl"

    def _embed_and_store_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return

        model = _get_embeddings()
        texts = [c.text for c in chunks]
        _batch = DEVICE_CONFIG.get("st_batch_size", 32)
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=_batch, show_progress_bar=False)

        for c, v in zip(chunks, vecs):
            c.embedding = np.array(v, dtype=np.float32)

        self.vector_store.add_chunks(chunks)

    def ensure_embeddings_ready(self, source_type: Optional[str] = None) -> None:
        source_types = [source_type] if source_type else list(self.pending_chunks.keys())

        for st in source_types:
            pending = self.pending_chunks.get(st, [])
            if not pending:
                continue

            ready_chunks: List[ChunkRecord] = []
            chunks_to_encode: List[ChunkRecord] = []

            for ch in pending:
                cache_path = self._embed_cache_path(ch.source_type, ch.source_name, ch.text)
                if cache_path.exists():
                    try:
                        with cache_path.open("rb") as f:
                            cached_vec = pickle.load(f)
                        ch.embedding = np.array(cached_vec, dtype=np.float32)
                        ready_chunks.append(ch)
                        continue
                    except Exception:
                        pass
                chunks_to_encode.append(ch)

            if ready_chunks:
                self.vector_store.add_chunks(ready_chunks)

            if chunks_to_encode:
                self._embed_and_store_chunks(chunks_to_encode)
                for ch in chunks_to_encode:
                    cache_path = self._embed_cache_path(ch.source_type, ch.source_name, ch.text)
                    try:
                        with cache_path.open("wb") as f:
                            pickle.dump(ch.embedding, f)
                    except Exception:
                        pass

            self.pending_chunks[st] = []

    def get_initial_greeting(self) -> str:
        intro = {
            "Français": (
                f"Bonjour, merci d'être là. On a environ {self.duration_minutes} minutes ensemble — "
                "on va commencer par votre parcours, vos projets et vos sujets techniques, "
                "puis on gardera la dernière partie pour les aspects comportementaux et vos questions éventuelles. "
                "Pour commencer, présentez-vous : votre formation, vos expériences principales, "
                "et ce qui vous a amené à postuler."
            ),
            "Anglais": (
                f"Hello, and welcome. We have about {self.duration_minutes} minutes together — "
                "we'll start with your background, relevant projects, and technical topics, "
                "then keep the final part for behavioral questions and any questions you may have. "
                "To get started, please introduce yourself: your education, your main experiences, "
                "and what brought you to apply for this position."
            ),
            "Arabe": (
                f"هلا، وشكراً على حضورك. عندنا تقريباً {self.duration_minutes} دقيقة مع بعض — "
                "راح نبدأ بخلفيتك، مشاريعك، والجوانب التقنية، "
                "وبعدين نخلي الجزء الأخير للأسئلة السلوكية وأي أسئلة عندك. "
                "للبداية، عرّف بنفسك: تعليمك، تجاربك الرئيسية، وما الذي دفعك للتقدم لهذا المنصب."
            ),
        }
        return intro[self.target_lang]
    def get_time_remaining(self) -> float:
        elapsed = (time.time() - self.started_at) / 60.0
        return max(0.0, round(self.duration_minutes - elapsed, 2))
    def _elapsed_minutes(self) -> float:
        return (time.time() - self.started_at) / 60.0

    def _is_in_block_b_window(self) -> bool:
        return self._elapsed_minutes() >= self.block_a_budget_minutes

    def _minutes_left_in_block_b(self) -> float:
        return max(
            0.0,
            self.duration_minutes - max(self._elapsed_minutes(), self.block_a_budget_minutes)
        )

    def _phase_share_target_minutes(self, phase: str) -> float:
        if phase in BLOCK_B_RESERVED_MINUTES:
            return round(self.block_b_budget_minutes * BLOCK_B_RESERVED_MINUTES[phase], 2)
        return 0.0

    def _candidate_turns_in_phase(self, phase: str) -> int:
        return len([
            t for t in self.turns
            if t.phase == phase and self._is_candidate_speaker(t.speaker)
        ])

    async def generate_response_async(self, user_text: str) -> Dict[str, Any]:
        events = []
        async for ev in self.generate_response_stream(user_text):
            events.append(ev)

        full_text = ""
        meta = {}
        for ev in events:
            if ev["type"] == "stream_done":
                full_text = ev["full_text"]
            elif ev["type"] == "meta":
                meta = ev

        return {
            "text": full_text,
            "candidate_sentiment": meta.get("candidate_sentiment", "neutre"),
            "interview_ended": meta.get("interview_ended", False),
            "phase": meta.get("phase", self.steps[self.current_step_index]),
            "time_left": meta.get("time_left", self.get_time_remaining()),
            "report": meta.get("report"),
        }

    def generate_response(self, user_text: str) -> Dict[str, Any]:
        import asyncio

        async def _collect():
            return await self.generate_response_async(user_text)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_collect())

        raise RuntimeError(
            "generate_response() ne doit pas etre appelee depuis un contexte async. "
            "Utilisez await generate_response_async(...)."
        )

    async def generate_response_stream(self, user_text: str):
        if self.ended:
            final = self._localized_closure("already_finished")
            for ev in self._yield_text_stream(final, candidate_sentiment="neutre", interview_ended=True):
                yield ev
            return

        self._ensure_llm_analysis()

        clean_user = self._clean_candidate_answer(user_text)
        candidate_sentiment = self._estimate_sentiment(clean_user)
        answer_quality = self._classify_answer_quality(clean_user)
        self._answer_quality_history.append(answer_quality)
        self._answer_quality_history = self._answer_quality_history[-10:]

        current_phase_for_claims = self.steps[self.current_step_index]
        if current_phase_for_claims != "OPENING":
            signal_validated = self._detect_and_register_claims(clean_user)
            weak_sigs = self._detect_weak_signals(clean_user)
        else:
            signal_validated = False
            weak_sigs = []

        behavioral_detected = self._detect_behavioral_story(clean_user)
        decision_detected = self._detect_decision_reasoning(clean_user)
        leadership_signals = self._detect_leadership_signals(clean_user)

        if behavioral_detected:
            self._behavioral_stories_count += 1
        if decision_detected:
            self._decision_shown_per_phase[self.steps[self.current_step_index]] = True
        if leadership_signals:
            self._leadership_signals.extend(leadership_signals)
            self._leadership_score = min(10, self._leadership_score + len(leadership_signals))

        self._turns_since_metrics_probe += 1

        self._append_turn(
            self.steps[self.current_step_index],
            self._candidate_speaker_label(),
            clean_user,
            answer_quality=answer_quality,
            signal_validated=signal_validated,
            behavioral_story_detected=behavioral_detected,
            decision_reasoning_detected=decision_detected,
            weak_signals_detected=weak_sigs,
        )
        self._update_dynamic_state(clean_user, candidate_sentiment)
        self._update_scores(clean_user)
        self._update_coverage_matrix(clean_user)

        current_phase = self.steps[self.current_step_index]
        if current_phase in ("CANDIDATE_QUESTIONS", "FINAL_CHECK"):
            if self._detecting_question_decline(clean_user):
                self._candidate_declined_questions = True
                # Forcer passage immédiat à CLOSING sans aucune autre question
                self.current_step_index = self.steps.index("CLOSING")
                self.ended = True

        if self._should_end_interview():
            self.ended = True
            next_text = self._build_closing()
        else:
            next_text = self._build_next_question(clean_user, candidate_sentiment, answer_quality)
            # Garde-fou : si entre-temps ended est devenu True, on écrase avec la conclusion seule
            if self.ended:
                next_text = self._build_closing()

        self._append_turn(
            self.steps[self.current_step_index],
            self._recruiter_speaker_label(),
            next_text,
            emotion=self.vision_emotion_label if self.vision_stress_flag else "neutre",
        )
        for ev in self._yield_text_stream(
            next_text,
            candidate_sentiment=candidate_sentiment,
            interview_ended=self.ended,
        ):
            yield ev


    # =========================================================================
    def _split_cv_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Découpe le CV en sections nommées.
        Retourne une liste de dicts : {"section": str, "text": str}
        """
        lines = text.splitlines()
        sections: List[Dict[str, str]] = []
        current_section = "header"
        current_lines: List[str] = []

        for line in lines:
            if CV_SECTION_PATTERNS.match(line.strip()):
                if current_lines:
                    sections.append({
                        "section": current_section,
                        "text": "\n".join(current_lines).strip()
                    })
                current_section = line.strip().lower()
                current_section = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", "", current_section).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "section": current_section,
                "text": "\n".join(current_lines).strip()
            })

        # Fallback : si aucune section détectée, tout comme "body"
        if not sections or all(s["text"] == "" for s in sections):
            return [{"section": "body", "text": text}]

        return [s for s in sections if s["text"].strip()]


    def _split_jd_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Découpe l'offre d'emploi en sections nommées.
        Retourne une liste de dicts : {"section": str, "text": str}
        """
        lines = text.splitlines()
        sections: List[Dict[str, str]] = []
        current_section = "intro"
        current_lines: List[str] = []

        for line in lines:
            if JD_SECTION_PATTERNS.match(line.strip()):
                if current_lines:
                    sections.append({
                        "section": current_section,
                        "text": "\n".join(current_lines).strip()
                    })
                current_section = line.strip().lower()
                current_section = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", "", current_section).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "section": current_section,
                "text": "\n".join(current_lines).strip()
            })

        if not sections or all(s["text"] == "" for s in sections):
            return [{"section": "body", "text": text}]

        return [s for s in sections if s["text"].strip()]


    def _extract_experience_header(self, block_text: str) -> Tuple[str, str]:
        """
        Extrait (entity, date_range) depuis le début d'un bloc d'expérience.
        Ex: "Google | Software Engineer | 2020-2023" → ("Google — Software Engineer", "2020-2023")
        """
        lines = [l.strip() for l in block_text.splitlines()[:4] if l.strip()]
        entity = ""
        date_range = ""

        date_pattern = re.compile(
            r"(\d{4}|\w+\s+\d{4})\s*[-–—/|]\s*(\d{4}|present|aujourd'hui|current|الآن|حتى الآن)",
            re.IGNORECASE
        )

        for line in lines:
            dm = date_pattern.search(line)
            if dm:
                date_range = dm.group(0).strip()
                # L'entité est souvent sur la même ligne avant la date, ou la ligne précédente
                before_date = line[:dm.start()].strip().strip("|•-— \t")
                if before_date and len(before_date) > 3:
                    entity = before_date[:60]
                elif lines[0] != line:
                    entity = lines[0][:60]
                break

        if not entity and lines:
            entity = lines[0][:60]

        return entity, date_range
    # RAG-IMPROVEMENT-3: Semantic Chunking
    # =========================================================================

    def _semantic_chunk_text(
        self, text: str, source_type: str, source_name: str
    ) -> List[ChunkRecord]:
        """
        RAG-IMPROVEMENT-3 + STRUCTURED CHUNKING:
        
        - CV       → découpage par sections + sous-chunks par expérience avec header contextuel
        - job_offer → découpage par sections + chunking sémantique à l'intérieur
        - autres   → chunking sémantique uniforme (comportement V3 original)
        """
        if not USE_SEMANTIC_CHUNKING:
            return self._chunk_text(text, source_type, source_name)

        text = text.strip()
        if not text:
            return []

        if source_type == "cv":
            return self._chunk_cv(text, source_name)
        elif source_type == "job_offer":
            return self._chunk_job_offer(text, source_name)
        else:
            # company_info et autres → sémantique uniforme
            return self._semantic_chunk_uniform(text, source_type, source_name)


    def _chunk_cv(self, text: str, source_name: str) -> List[ChunkRecord]:
        """
        CV : section → expérience-entry → sous-chunk si trop long.
        Chaque chunk préfixé par son contexte [Section | Entité | Dates].
        """
        sections = self._split_cv_sections(text)
        chunks: List[ChunkRecord] = []
        chunk_idx = 0

        for sec in sections:
            sec_name = sec["section"]
            sec_text = sec["text"]

            # Sections non-expérience → un seul chunk (ou sémantique si long)
            is_experience_section = any(k in sec_name.lower() for k in [
                "expérience", "experience", "تجارب", "خبرات", "projets", "projects", "مشاريع"
            ])

            if not is_experience_section:
                word_count = len(sec_text.split())
                if word_count <= SEMANTIC_MAX_CHUNK_TOKENS:
                    if word_count >= SEMANTIC_MIN_CHUNK_TOKENS:
                        chunks.append(ChunkRecord(
                            chunk_id=f"cv_{source_name}_sec_{chunk_idx}",
                            source_type="cv",
                            source_name=source_name,
                            text=f"[{sec_name.title()}]\n{sec_text}",
                            meta={"section": sec_name, "structured": True},
                            section=sec_name,
                        ))
                        chunk_idx += 1
                else:
                    # Section longue (ex: liste de skills dense) → sémantique
                    sub_chunks = self._semantic_chunk_uniform(sec_text, "cv", source_name, start_idx=chunk_idx)
                    for sc in sub_chunks:
                        sc.section = sec_name
                        sc.meta["section"] = sec_name
                        sc.text = f"[{sec_name.title()}]\n{sc.text}"
                        sc.chunk_id = f"cv_{source_name}_sec_{chunk_idx}"
                        chunks.append(sc)
                        chunk_idx += 1
                continue

            # Section expérience → découpage par entrée
            # Détecte les blocs d'expérience individuelle
            entry_blocks = self._split_experience_entries(sec_text)

            for block in entry_blocks:
                entity, date_range = self._extract_experience_header(block)
                header = f"[{sec_name.title()} | {entity} | {date_range}]" if (entity or date_range) else f"[{sec_name.title()}]"
                word_count = len(block.split())

                if word_count <= SEMANTIC_MAX_CHUNK_TOKENS:
                    if word_count >= SEMANTIC_MIN_CHUNK_TOKENS:
                        chunks.append(ChunkRecord(
                            chunk_id=f"cv_{source_name}_exp_{chunk_idx}",
                            source_type="cv",
                            source_name=source_name,
                            text=f"{header}\n{block}",
                            meta={"section": sec_name, "entity": entity,
                                "date_range": date_range, "structured": True},
                            section=sec_name,
                            entity=entity,
                            date_range=date_range,
                        ))
                        chunk_idx += 1
                else:
                    # Expérience trop longue → sous-chunks sémantiques avec header préfixé
                    sub_chunks = self._semantic_chunk_uniform(block, "cv", source_name, start_idx=chunk_idx)
                    for sc in sub_chunks:
                        sc.section = sec_name
                        sc.entity = entity
                        sc.date_range = date_range
                        sc.meta.update({"section": sec_name, "entity": entity, "date_range": date_range})
                        # Préfixer chaque sous-chunk avec le contexte de l'expérience
                        sc.text = f"{header}\n{sc.text}"
                        sc.chunk_id = f"cv_{source_name}_exp_{chunk_idx}"
                        chunks.append(sc)
                        chunk_idx += 1

        return chunks if chunks else self._chunk_text(text, "cv", source_name)


    def _chunk_job_offer(self, text: str, source_name: str) -> List[ChunkRecord]:
        """
        Offre d'emploi : section → chunking sémantique à l'intérieur.
        Chaque chunk taggé avec sa section JD pour le reranking.
        """
        sections = self._split_jd_sections(text)
        chunks: List[ChunkRecord] = []
        chunk_idx = 0

        for sec in sections:
            sec_name = sec["section"]
            sec_text = sec["text"]

            if not sec_text.strip():
                continue

            word_count = len(sec_text.split())

            if word_count <= SEMANTIC_MAX_CHUNK_TOKENS:
                if word_count >= SEMANTIC_MIN_CHUNK_TOKENS:
                    chunks.append(ChunkRecord(
                        chunk_id=f"job_offer_{source_name}_sec_{chunk_idx}",
                        source_type="job_offer",
                        source_name=source_name,
                        text=f"[{sec_name.title()}]\n{sec_text}",
                        meta={"section": sec_name, "structured": True},
                        section=sec_name,
                    ))
                    chunk_idx += 1
            else:
                sub_chunks = self._semantic_chunk_uniform(sec_text, "job_offer", source_name, start_idx=chunk_idx)
                for sc in sub_chunks:
                    sc.section = sec_name
                    sc.meta["section"] = sec_name
                    sc.text = f"[{sec_name.title()}]\n{sc.text}"
                    sc.chunk_id = f"job_offer_{source_name}_sec_{chunk_idx}"
                    chunks.append(sc)
                    chunk_idx += 1

        return chunks if chunks else self._chunk_text(text, "job_offer", source_name)


    def _split_experience_entries(self, text: str) -> List[str]:
        """
        Découpe un bloc expérience en entrées individuelles
        basé sur les patterns de dates ou séparateurs visuels.
        """
        # Cherche les positions des débuts d'entrée
        matches = list(CV_EXPERIENCE_ENTRY_PATTERNS.finditer(text))

        if len(matches) < 2:
            return [text]

        blocks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            if len(block.split()) >= SEMANTIC_MIN_CHUNK_TOKENS:
                blocks.append(block)

        return blocks if blocks else [text]


    def _semantic_chunk_uniform(
        self,
        text: str,
        source_type: str,
        source_name: str,
        start_idx: int = 0,
    ) -> List[ChunkRecord]:
        """
        Chunking sémantique uniforme (V3 original) — utilisé comme fallback
        et pour les sections non-structurées.
        """
        text = text.strip()
        if not text:
            return []

        sentences = _split_sentences(text)
        if len(sentences) < 2:
            return self._chunk_text(text, source_type, source_name)

        try:
            model = _get_embeddings()
            embeddings = model.encode(
                sentences,
                normalize_embeddings=True,
                batch_size=DEVICE_CONFIG.get("st_batch_size", 32),
                show_progress_bar=False
            )
        except Exception as e:
            print(f"WARNING _semantic_chunk_uniform embedding: {e} — fallback to fixed chunking.")
            return self._chunk_text(text, source_type, source_name)

        chunks: List[ChunkRecord] = []
        current_sentences: List[str] = []
        current_word_count: int = 0
        chunk_idx: int = start_idx

        def _flush(sents: List[str]) -> None:
            nonlocal chunk_idx
            chunk_text = " ".join(sents).strip()
            if len(chunk_text.split()) < SEMANTIC_MIN_CHUNK_TOKENS:
                return
            chunks.append(ChunkRecord(
                chunk_id=f"{source_type}_{source_name}_sem_{chunk_idx}",
                source_type=source_type,
                source_name=source_name,
                text=chunk_text,
                meta={"semantic": True, "n_sentences": len(sents)},
            ))
            chunk_idx += 1

        for i, sentence in enumerate(sentences):
            wc = len(sentence.split())
            current_sentences.append(sentence)
            current_word_count += wc

            is_last = (i == len(sentences) - 1)
            should_break = False

            if not is_last:
                sim = _cosine(embeddings[i], embeddings[i + 1])
                semantic_break = sim < (1.0 - SEMANTIC_BREAKPOINT_THRESHOLD)
                size_break = current_word_count >= SEMANTIC_MAX_CHUNK_TOKENS
                should_break = semantic_break or size_break

            if should_break or is_last:
                _flush(current_sentences)
                current_sentences = []
                current_word_count = 0

        return chunks

        def _flush(sents: List[str]) -> None:
            nonlocal chunk_idx
            chunk_text = " ".join(sents).strip()
            if len(chunk_text.split()) < SEMANTIC_MIN_CHUNK_TOKENS:
                return  # too small — discard (will be covered by neighbour)
            chunks.append(ChunkRecord(
                chunk_id=f"{source_type}_{source_name}_sem_{chunk_idx}",
                source_type=source_type,
                source_name=source_name,
                text=chunk_text,
                meta={"semantic": True, "n_sentences": len(sents)},
            ))
            chunk_idx += 1

        for i, sentence in enumerate(sentences):
            wc = len(sentence.split())
            current_sentences.append(sentence)
            current_word_count += wc

            # Decide whether to break after this sentence
            is_last = (i == len(sentences) - 1)
            should_break = False

            if not is_last:
                # Cosine similarity between this sentence and the next
                sim = _cosine(embeddings[i], embeddings[i + 1])
                semantic_break = sim < (1.0 - SEMANTIC_BREAKPOINT_THRESHOLD)
                size_break = current_word_count >= SEMANTIC_MAX_CHUNK_TOKENS
                should_break = semantic_break or size_break

            if should_break or is_last:
                _flush(current_sentences)
                current_sentences = []
                current_word_count = 0

        return chunks

    # ── Legacy fixed-size chunking (kept as fallback) ────────────────────────

    def _chunk_text(self, text: str, source_type: str, source_name: str) -> List[ChunkRecord]:
        text = text.strip()
        if not text:
            return []
        chunks, start, idx = [], 0, 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            chunks.append(ChunkRecord(
                chunk_id=f"{source_type}_{source_name}_{idx}",
                source_type=source_type, source_name=source_name,
                text=text[start:end], meta={"start": start, "end": end},
            ))
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
            idx += 1
        return chunks

    # =========================================================================
    # Core: Build next question
    # =========================================================================

    def _build_next_question(
        self, user_text: str, candidate_sentiment: str, answer_quality: str = "GOOD"
    ) -> str:
        phase = self.steps[self.current_step_index]
        # Si le candidat a décliné les questions → conclusion directe, sans aucune autre question
        if self._candidate_declined_questions:
            self.current_step_index = self.steps.index("CLOSING")
            self.ended = True
            return self._build_closing()

        # If candidate already said they have no more questions, skip to closing
        if self._candidate_declined_questions and phase in ("CANDIDATE_QUESTIONS", "FINAL_CHECK"):
            if phase == "CANDIDATE_QUESTIONS":
                self._advance_phase()
            self._advance_phase()
            return self._build_closing()

        # If candidate starts asking questions early, jump to candidate questions phase
        if (
            self._candidate_is_asking_questions(user_text)
            and phase not in ("CANDIDATE_QUESTIONS", "FINAL_CHECK", "CLOSING")
        ):
            self.candidate_questions_mode = True
            self.current_step_index = self.steps.index("CANDIDATE_QUESTIONS")
            phase = "CANDIDATE_QUESTIONS"

        # Hard switch to Block B once 2/3 of interview time is consumed
        if self._is_in_block_b_window() and phase in BLOCK_A_PHASES:
            self.current_step_index = self.steps.index("SOFT_SKILLS_BEHAVIORAL")
            phase = "SOFT_SKILLS_BEHAVIORAL"

        # Phase progression
        if self._needs_phase_progression(phase):
            self._advance_phase()
            phase = self.steps[self.current_step_index]

        # Hard closing guard
        if phase == "CLOSING":
            return {
                "Français": "Merci d'avoir été présent(e) aujourd'hui. Nous reviendrons vers vous avec la réponse finale dans quelques jours.",
                "Anglais": "Thank you for being with us today. We will get back to you with the final decision within the next few days.",
                "Arabe": "شكراً لحضورك اليوم. راح نرجع لك بالرد النهائي خلال الأيام الجاية.",
            }[self.target_lang]

        # =========================================================
        # CANDIDATE QUESTIONS PHASE — fully deterministic, no LLM
        # =========================================================

        # First entry: ask if the candidate has any questions
        if phase == "CANDIDATE_QUESTIONS" and not self.candidate_questions_started:
            self.candidate_questions_started = True
            return {
                "Français": "Avant de conclure cette partie, avez-vous des questions sur le poste, l'équipe ou le déroulement du processus ?",
                "Anglais": "Before we move to the end, do you have any questions about the role, the team, or the hiring process?",
                "Arabe": "قبل ما نكمل للنهاية، هل عندك أي أسئلة عن الدور أو الفريق أو خطوات التوظيف؟",
            }[self.target_lang]

        # Candidate questions phase is active — handle every turn here, never fall through to LLM
        if phase == "CANDIDATE_QUESTIONS" and self.candidate_questions_started:

            # 1. Le candidat décline explicitement → avancer
            if self._detecting_question_decline(user_text):
                self._candidate_declined_questions = True
                self._advance_phase()
                phase = self.steps[self.current_step_index]

                if phase == "FINAL_CHECK":
                    return {
                        "Français": "Avant de conclure, y a-t-il quelque chose que vous souhaitez ajouter et que nous n'avons pas couvert ?",
                        "Anglais": "Before we close, is there anything you'd like to add that we have not covered?",
                        "Arabe": "قبل ما نختم، هل فيه شيء تحب تضيفه وما غطيناه؟",
                    }[self.target_lang]

                return {
                    "Français": "Merci d'avoir été présent(e) aujourd'hui. Nous reviendrons vers vous avec la réponse finale dans quelques jours.",
                    "Anglais": "Thank you for being with us today. We will get back to you with the final decision within the next few days.",
                    "Arabe": "شكراً لحضورك اليوم. راح نرجع لك بالرد النهائي خلال الأيام الجاية.",
                }[self.target_lang]

            # 2. Le candidat pose une vraie question → répondre
            self._candidate_question_rounds += 1
            answer = self._build_candidate_question_answer(user_text)

            # Invite à d'autres questions
            followup_invite = {
                "Français": " Avez-vous d'autres questions ?",
                "Anglais":  " Do you have any other questions?",
                "Arabe":    " هل عندك أي أسئلة أخرى؟",
            }[self.target_lang]

            # Max rounds atteint → conclure après la réponse
            if self._candidate_question_rounds >= self._max_candidate_question_rounds:
                self.candidate_questions_completed = True
                self._advance_phase()
                transition = {
                    "Français": " Je propose que nous passions à la conclusion.",
                    "Anglais":  " I suggest we move on to wrap up.",
                    "Arabe":    " أقترح أن ننتقل للخاتمة.",
                }[self.target_lang]
                return answer.rstrip() + transition

            return answer.rstrip() + " " + followup_invite.strip()

        # =========================================================
        # FINAL CHECK — deterministic, no LLM
        # =========================================================
        if phase == "FINAL_CHECK":
            return {
                "Français": "Avant de conclure, y a-t-il quelque chose que vous souhaitez ajouter et que nous n'avons pas couvert ?",
                "Anglais": "Before we close, is there anything you'd like to add that we have not covered?",
                "Arabe": "قبل ما نختم، هل فيه شيء تحب تضيفه وما غطيناه؟",
            }[self.target_lang]

        # =========================================================
        # ALL OTHER PHASES — LLM-generated question
        # =========================================================
        briefing = self._build_agent_brief(phase, user_text, candidate_sentiment, answer_quality)

        # ── MULTI-AGENT : Orchestrateur / Traducteur pour l'arabe saoudien ──
        if self.target_lang == "Arabe":
            question = self._ot_agent.generate(briefing, phase=phase)
        else:
            llm_raw = self._call_llm(
                briefing,
                phase=phase,
                include_few_shot=True,
            )
            question = self._extract_final_question(llm_raw)

        question = self._postprocess_question(question, phase)
        question = self._strip_praise(question)
        question = self._humanize_question(question, user_text, candidate_sentiment)
        question = self._ensure_human_quality(question, phase, user_text)

        if phase == "JOB_ALIGNED_EXPLORATION" and self._current_tool_focus:
            self._tool_followup_count[self._current_tool_focus] = (
                self._tool_followup_count.get(self._current_tool_focus, 0) + 1
            )
            if self._tool_followup_count[self._current_tool_focus] >= MAX_TOOL_FOLLOWUPS:
                self._tools_fully_done.add(self._current_tool_focus)
                self._current_tool_focus = ""

        self._mark_jd_tool_why_asked(question)
        self._update_vague_retry_state(user_text, answer_quality)

        return question
    # =========================================================================
    # Central LLM prompt
    # =========================================================================

    def _build_agent_brief(
        self,
        phase: str,
        user_text: str,
        candidate_sentiment: str,
        answer_quality: str = "GOOD",
    ) -> str:
        retrieved = self._retrieve_context(user_text, phase)
        interview_state = self._format_state_summary(phase, candidate_sentiment)
        phase_directive = self._phase_directive(phase, user_text, answer_quality)
        quality_directive = self._answer_quality_directive(answer_quality)
        anti_repeat = self._anti_repeat_directive()
        signal_val_dir = self._build_signal_validation_directive()
        job_adapt_dir = self._build_job_adaptation_directive(phase)
        behavioral_dir = self._build_behavioral_assessment_directive(phase)
        decision_dir = self._build_decision_making_directive(phase)
        seniority_dir = self._build_seniority_directive(phase)
        metrics_dir = self._metrics_probe_directive(phase)
        challenge_dir = self._build_challenge_directive(answer_quality, user_text)
        time_pressure_dir = self._time_pressure_directive(phase)
        relance_dir = self._build_relance_directive(answer_quality, user_text, phase)
        weak_signal_dir = self._build_weak_signal_directive()
        coverage_gap_dir = self._coverage_gap_report(phase)

        is_opening = (phase == "OPENING")
        is_first_tool_question = (
            phase == "JOB_ALIGNED_EXPLORATION"
            and self._tool_followup_count.get(self._current_tool_focus, 0) == 0
        )

        end_phases = {"CANDIDATE_QUESTIONS", "FINAL_CHECK", "CLOSING"}

        # --- NOUVEAU : détection topic saturé ---
        saturated_topics = self._get_saturated_topics()
        next_topic = self._get_next_uncovered_topic(phase)
        topic_switch_block = ""
        if (
            self.current_topic_focus in saturated_topics
            and phase not in {"OPENING", "CANDIDATE_QUESTIONS", "FINAL_CHECK", "CLOSING"}
        ):
            topic_switch_block = (
                f"\n🚨 MANDATORY TOPIC SWITCH:\n"
                f"You have asked {self._topic_followup_count.get(self.current_topic_focus, 0)} questions "
                f"on '{self.current_topic_focus}' — this topic is SATURATED.\n"
                f"→ Do NOT ask another question on '{self.current_topic_focus}'.\n"
                f"→ Move IMMEDIATELY to: '{next_topic or 'a completely different angle'}'.\n"
                f"→ Saturated topics to avoid entirely: {saturated_topics}\n"
                f"→ Covered topics to avoid repeating: {sorted(self.covered_topics)}\n"
            )
        # ----------------------------------------

        priority_block = ""
        if phase not in end_phases:
            # Priorité absolue : switch forcé si topic saturé
            if topic_switch_block:
                priority_block = topic_switch_block
            elif not is_opening and not is_first_tool_question and signal_val_dir:
                priority_block = f"\n⚡ HIGHEST PRIORITY — CLAIM VALIDATION:\n{signal_val_dir}\n"
            elif not is_opening and weak_signal_dir:
                priority_block = f"\n⚡ HIGH PRIORITY — WEAK SIGNAL PROBE:\n{weak_signal_dir}\n"
            elif not is_opening and relance_dir:
                priority_block = f"\n⚡ HIGH PRIORITY — SMART RELANCE:\n{relance_dir}\n"
            elif not is_opening and challenge_dir:
                priority_block = f"\n⚡ HIGH PRIORITY — CHALLENGE STRONG SIGNAL:\n{challenge_dir}\n"
            elif behavioral_dir and phase == "SOFT_SKILLS_BEHAVIORAL":
                priority_block = f"\n⚡ HIGH PRIORITY — BEHAVIORAL ASSESSMENT:\n{behavioral_dir}\n"
            elif decision_dir and phase in {"PROJECT_DEEP_DIVE", "TECHNICAL_DEPTH"}:
                priority_block = f"\n⚡ HIGH PRIORITY — DECISION-MAKING PROBE:\n{decision_dir}\n"

        cv_signal_block = ""
        if is_opening:
            last_answer = self._last_candidate_message()
            if last_answer:
                cv_signals = self._extract_cv_strong_signals(last_answer)
                mentioned_projects = self._extract_mentioned_projects(last_answer)  # défini AVANT les if
                projects_str = ", ".join(f'"{p}"' for p in mentioned_projects[:4]) if mentioned_projects else "none"

                if cv_signals:
                    top = cv_signals[0]
                    target_project = mentioned_projects[0] if mentioned_projects else "the project they mentioned"
                    cv_signal_block = (
                        f"\n⚡ OPENING SIGNAL — EXPLOIT IMMEDIATELY:\n"
                        f"Candidate mentioned: \"{top['quote']}\"\n"
                        f"Projects named by candidate: {projects_str}\n\n"
                        f"CRITICAL FORMATTING RULE:\n"
                        f"→ Your question MUST start by naming a specific project.\n"
                        f"→ Correct format: 'On your {target_project} — [specific question]?'\n"
                        f"→ WRONG format: 'How did you integrate these components?' (no project named)\n"
                        f"→ WRONG format: 'Tell me about your experience with...' (generic)\n"
                        f"→ Do NOT say 'Noted', 'Great', or any filler.\n"
                    )
                elif mentioned_projects:
                    target_project = mentioned_projects[0]
                    cv_signal_block = (
                        f"\n⚡ OPENING — PROJECT ANCHOR REQUIRED:\n"
                        f"Projects named by candidate: {projects_str}\n"
                        f"→ Your question MUST name '{target_project}' explicitly.\n"
                        f"→ Correct format: 'On your {target_project} — [specific question]?'\n"
                    )

        return f"""
    You are evaluating the next best recruiter turn.

    INTERVIEW CONTEXT
    LANGUAGE          : {self.target_lang}
    CURRENT PHASE     : {phase} — {PHASE_LABELS[phase]}
    JOB DOMAIN        : {self._job_domain or "Not yet determined"}
    SENIORITY LEVEL   : {self._job_seniority_level}
    JD TOOLS TO COVER : {self._jd_tools[:10]}
    JD TOOLS EXPLORED : {sorted(self.explored_jd_tools)}
    JD TOOLS WHY ASKED: {sorted(self._jd_tool_why_asked)}
    TOOLS FULLY DONE  : {sorted(self._tools_fully_done)}
    CURRENT TOOL FOCUS: {self._current_tool_focus or "none"}
    TOOL FOLLOWUPS    : {dict(self._tool_followup_count)}
    PROJECTS DIVED    : {sorted(self.deep_dived_projects)}
    TIME REMAINING    : {self.get_time_remaining()} min
    COVERAGE GAPS     : {coverage_gap_dir}

    {priority_block}{cv_signal_block}

    PHASE DIRECTIVE
    {phase_directive}

    ANSWER QUALITY [{answer_quality}]
    {quality_directive}
    {metrics_dir if phase not in end_phases else ""}

    JOB-SPECIFIC ADAPTATION
    {job_adapt_dir or "No specific adaptation required."}
    {seniority_dir or ""}
    {time_pressure_dir or ""}

    ANTI-REPETITION
    {anti_repeat if phase not in end_phases else "Do not repeat previous technical questions. End-of-interview behavior only."}

    GLOBAL RULES
    {self._global_behavioral_rules()}

    INTERVIEW STATE
    {interview_state}

    DOCUMENT CONTEXT (CV + JD) — Retrieved via Hybrid BM25+Vector Search
    {retrieved}

    LAST CANDIDATE ANSWER
    {user_text}

    CRITICAL SOURCE SEPARATION RULE
    The DOCUMENT CONTEXT above is split into two clearly labeled sections:
    - "CV CONTEXT" = what the candidate has actually done and built.
    - "JOB DESCRIPTION CONTEXT" = what the role requires. This is NOT the candidate's experience.
    NEVER attribute information from the JOB DESCRIPTION to the candidate.
    Example of FORBIDDEN confusion: "you built an AI-based HR avatar" when that appears only in the JD.
    Example of CORRECT usage: "the role involves an AI-based HR avatar — have you worked on something similar?"

    OUTPUT FORMAT
    Return plain text only.
    Write exactly what the recruiter says next.
    1 focused, sharp follow-up question.
    1–2 sentences maximum.
    """.strip()

    # =========================================================================
    def _build_few_shot_block(self, phase: Optional[str]) -> str:
        if not phase:
            return ""

        lang_examples = FEW_SHOT_EXAMPLES.get(self.target_lang, {})
        examples = lang_examples.get(phase, [])

        if not examples:
            examples = FEW_SHOT_EXAMPLES.get("Anglais", {}).get(phase, [])

        if not examples:
            return ""

        selected = examples[:2]
        rendered = []

        for ex in selected:
            rendered.append(
                "[Recruiter question]\n"
                f"{ex['recruiter_question']}\n"
                "[Candidate answer]\n"
                f"{ex['candidate_answer']}\n"
                "[Your follow-up]\n"
                f"{ex['your_follow_up']}"
            )

        return "FEW-SHOT EXAMPLES:\n\n" + "\n\n".join(rendered)

    def _build_system_prompt(
        self,
        phase: Optional[str] = None,
        include_few_shot: bool = True,
        extra_rules: str = "",
    ) -> str:
        parts = [BASE_SYSTEM_PROMPT]

        if phase:
            parts.append(f"CURRENT INTERVIEW PHASE: {phase}")

        parts.append(
            "STRICT OUTPUT RULES:\n"
            "- Ask one focused question.\n"
            "- Maximum 2 sentences.\n"
            "- No bullets, no labels, no markdown.\n"
            "- Never start with filler like Noted, Good, Thank you, I see, Absolutely.\n"
            "- Never praise the candidate.\n"
            "- Never summarize the candidate answer before the question.\n"
            "- Sound human, direct, and evaluative."
        )

        if extra_rules:
            parts.append(extra_rules)

        if include_few_shot:
            few_shot_block = self._build_few_shot_block(phase)
            if few_shot_block:
                parts.append(few_shot_block)

        parts.append(
            OUTPUT_LANGUAGE_GATES.get(
                self.target_lang,
                OUTPUT_LANGUAGE_GATES["Anglais"]
            )
        )

        return "\n\n".join(parts).strip()
    # FIX-AR-3: Language enforcement block builder
    # =========================================================================

    def _build_language_enforcement_block(self) -> str:
        if self.target_lang == "Arabe":
            return (
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  🔴 إلزامي: اكتب ردك باللغة العربية السعودية حصراً.      ║\n"
                "║  المصطلحات التقنية الإنجليزية مسموحة (مثل: API، deploy). ║\n"
                "║  ممنوع تماماً استخدام الفرنسية أو الإنجليزية العادية.   ║\n"
                "║  أي كلمة فرنسية في الإجابة تُعدّ خطأً فادحاً.           ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )
        if self.target_lang == "Français":
            return (
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  🔴 OBLIGATOIRE : Réponds UNIQUEMENT en français.         ║\n"
                "║  Termes techniques anglais autorisés.                     ║\n"
                "║  Toute réponse en arabe ou autre langue est une erreur.   ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )
        return (
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  🔴 MANDATORY: Write your response in English ONLY.       ║\n"
            "║  Technical terms in any language are allowed.             ║\n"
            "║  Any French or Arabic response is an error.               ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )

    # =========================================================================
    # FIX-AR-2: Language rules
    # =========================================================================

    def _language_rules(self) -> str:
        if self.target_lang == "Français":
            return (
                "• Parle en français uniquement.\n"
                "• Ton naturel de recruteur, pas de formules robotiques.\n"
                "• Les termes techniques anglais (API, pipeline, deploy) sont autorisés.\n"
                "• N'utilise JAMAIS l'arabe, l'anglais courant, ou toute autre langue.\n"
                "• Si le prompt contient des instructions en arabe → exécute-les en français."
            )
        if self.target_lang == "Anglais":
            return (
                "• Speak in professional English only.\n"
                "• Concise, human recruiter tone.\n"
                "• Technical terms in any language are fine.\n"
                "• NEVER use French, Arabic, or any other non-English words.\n"
                "• If the prompt contains Arabic/French instructions → follow them but answer in English."
            )
        return (
            "• اكتب بالعربية السعودية الفصحى فقط، بنبرة محاور توظيف محترف.\n"
            "• المصطلحات التقنية الإنجليزية مسموحة: API, pipeline, deploy, backend,\n"
            "  frontend, ML, LLM, RAG, ETL, SLA, SLO, DAG, CI/CD, Docker, Kubernetes.\n"
            "• ممنوع منعاً باتاً استخدام أي كلمات فرنسية حتى لو ظهرت في الـ prompt.\n"
            "• ممنوع أيضاً استخدام الإنجليزية العادية بخلاف المصطلحات التقنية.\n"
            "• إذا وجدت تعليمات بالفرنسية في الـ prompt → نفّذ المعنى لكن اكتب العربية.\n"
            "• التحقق الذاتي قبل الإرسال: هل كل كلمة في ردي إما عربية أو مصطلح تقني؟"
        )

    # =========================================================================
    # IMPROVEMENT A: Smart relance directive
    # =========================================================================

    def _build_relance_directive(self, answer_quality: str, user_text: str, phase: str) -> str:
        if answer_quality not in ("VAGUE", "INCOMPLETE"):
            self._vague_retry_count[self.current_topic_focus] = 0
            return ""

        topic = self.current_topic_focus or phase
        retries = self._vague_retry_count.get(topic, 0)

        tier_key = f"{topic}:tier{min(retries+1, 3)}"
        if tier_key in self._relance_tier_used:
            return ""

        self._relance_tier_used.add(tier_key)

        ar_mandate = "⚠️ اكتب السؤال بالعربية فقط. "
        fr_mandate = "⚠️ Répondez en français uniquement. "
        en_mandate = "⚠️ Write in English only. "

        if retries == 0:
            templates = {
                "Français": (fr_mandate + "RELANCE TIER 1 — La réponse est trop vague. Demandez UN exemple concret et précis."),
                "Anglais":  (en_mandate + "RELANCE TIER 1 — Answer too vague. Ask for ONE concrete example: a specific project, a number, or a decision they personally made."),
                "Arabe":    (ar_mandate + "RELANCE TIER 1 — الإجابة مبهمة جداً. اطلب مثالاً واحداً ملموساً: مشروع محدد، رقم، أو قرار أخذه شخصياً."),
            }
        elif retries == 1:
            templates = {
                "Français": (fr_mandate + "RELANCE TIER 2 — Deuxième réponse vague. Proposez un cadre STAR."),
                "Anglais":  (en_mandate + "RELANCE TIER 2 — Second vague answer. Offer scaffolding: Situation → Action → Result."),
                "Arabe":    (ar_mandate + "RELANCE TIER 2 — إجابة مبهمة ثانية. قدم إطاراً: موقف → فعل → نتيجة."),
            }
        else:
            templates = {
                "Français": (fr_mandate + "RELANCE TIER 3 — Trois réponses vagues. Notez la LACUNE. Passez immédiatement à un nouveau sujet."),
                "Anglais":  (en_mandate + "RELANCE TIER 3 — Three vague answers. Mark as GAP. Move immediately to a new topic."),
                "Arabe":    (ar_mandate + "RELANCE TIER 3 — ثلاث إجابات مبهمة. سجّله كـ GAP وانتقل فوراً لموضوع جديد بالعربية."),
            }
            if topic not in self.session_notes.get("weaknesses", []):
                self.session_notes.setdefault("weaknesses", []).append(f"Consistently vague on: {topic}")

        return templates.get(self.target_lang, templates["Anglais"])
    def _get_saturated_topics(self) -> List[str]:
        """Retourne les topics ayant atteint ou dépassé _max_topic_followups."""
        return [
            t for t, count in self._topic_followup_count.items()
            if count >= self._max_topic_followups
        ]

    def _get_next_uncovered_topic(self, phase: str) -> str:
        """
        Retourne le prochain topic prioritaire non saturé et non couvert,
        selon la phase courante.
        """
        saturated = set(self._get_saturated_topics())
        covered = self.covered_topics | saturated

        phase_topic_priority = {
            "PROJECT_DEEP_DIVE": [
                "architecture", "problem_solving", "delivery",
                "cost_ownership", "observability", "quality"
            ],
            "TECHNICAL_DEPTH": [
                "problem_solving", "architecture", "observability",
                "delivery", "quality", "cost_ownership"
            ],
            "SOFT_SKILLS_BEHAVIORAL": [
                "conflict", "prioritization", "failure",
                "leadership", "pressure"
            ],
            "JOB_ALIGNED_EXPLORATION": [],  # géré par _tool_rotation_order
        }

        priorities = phase_topic_priority.get(phase, list(self._job_topic_keywords.keys()))
        for topic in priorities:
            if topic not in covered:
                return topic

        # Fallback : topic couvert mais non saturé (pour approfondir différemment)
        for topic in priorities:
            if topic not in saturated:
                return topic

        return ""

    def _update_vague_retry_state(self, user_text: str, answer_quality: str) -> None:
        topic = self.current_topic_focus or "general"
        if answer_quality in ("VAGUE", "INCOMPLETE"):
            self._vague_retry_count[topic] = self._vague_retry_count.get(topic, 0) + 1
            self._current_vague_topic = topic
        else:
            self._vague_retry_count[topic] = 0
            self._current_vague_topic = ""

    # =========================================================================
    # IMPROVEMENT B: Weak signal detection
    # =========================================================================

    def _detect_weak_signals(self, text: str) -> List[str]:
        detected = []
        lower = text.lower()
        for sig_type, patterns in WEAK_SIGNAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, lower, re.IGNORECASE)
                for m in matches:
                    excerpt = m if isinstance(m, str) else m[0]
                    idx = lower.find(excerpt.lower())
                    if idx >= 0:
                        window_start = max(0, idx - 30)
                        window_end = min(len(text), idx + len(excerpt) + 60)
                        context = text[window_start:window_end].strip()
                    else:
                        context = excerpt

                    key = f"{sig_type}:{context[:60]}"
                    if key not in self._probed_weak_signals:
                        self._weak_signal_queue.append({
                            "type": sig_type,
                            "excerpt": excerpt,
                            "context": context,
                            "turn_idx": len(self.turns),
                        })
                        detected.append(sig_type)
                        break
        return list(set(detected))

    def _build_weak_signal_directive(self) -> str:
        if not self._weak_signal_queue:
            return ""

        signal = self._weak_signal_queue.pop(0)
        key = f"{signal['type']}:{signal['context'][:60]}"
        self._probed_weak_signals.add(key)

        sig_type = signal["type"]
        excerpt = signal.get("context", signal["excerpt"])[:100]

        ar_mandate = "⚠️ اكتب السؤال بالعربية فقط. "
        fr_mandate = "⚠️ Question en français uniquement. "
        en_mandate = "⚠️ Write in English only. "

        templates = {
            "hedged_ownership": {
                "Français": fr_mandate + f"Le candidat a dit '{excerpt}' — collectif. Demandez ce que LUI a concrètement fait.",
                "Anglais":  en_mandate + f"Candidate used collective language: '{excerpt}'. Ask what THEY specifically did.",
                "Arabe":    ar_mandate + f"المرشح استخدم صيغة جماعية: '{excerpt}'. اسأله ماذا فعل هو تحديداً.",
            },
            "vague_impact": {
                "Français": fr_mandate + f"Le candidat a dit '{excerpt}' sans chiffre. Demandez: de combien exactement ?",
                "Anglais":  en_mandate + f"Candidate said '{excerpt}' with no number. Ask: by how much exactly?",
                "Arabe":    ar_mandate + f"المرشح قال '{excerpt}' بدون أرقام. اسأل: بقدر كم بالضبط؟",
            },
            "understatement": {
                "Français": fr_mandate + f"Le candidat a minimisé: '{excerpt}'. Challengez: était-ce vraiment mineur ?",
                "Anglais":  en_mandate + f"Candidate understated: '{excerpt}'. Push back: was that actually minor?",
                "Arabe":    ar_mandate + f"المرشح قلّل من إجابته: '{excerpt}'. تحدّاه: هل كان فعلاً صغيراً؟",
            },
            "unanchored_comparison": {
                "Français": fr_mandate + f"Le candidat a dit '{excerpt}' sans baseline. Demandez: par rapport à quoi ?",
                "Anglais":  en_mandate + f"Candidate said '{excerpt}' with no baseline. Ask: compared to what?",
                "Arabe":    ar_mandate + f"المرشح قال '{excerpt}' بدون مرجع. اسأل: مقارنة بماذا؟",
            },
        }

        fallback = {
            "Français": fr_mandate + f"Demandez une preuve concrète de: '{excerpt}'",
            "Anglais":  en_mandate + f"Ask for concrete evidence of: '{excerpt}'",
            "Arabe":    ar_mandate + f"اطلب دليلاً ملموساً على: '{excerpt}'",
        }

        chosen = templates.get(sig_type, fallback)
        return chosen.get(self.target_lang, chosen.get("Anglais", ""))

    # =========================================================================
    # IMPROVEMENT C: Enhanced deduplication
    # =========================================================================

    def _compute_question_fingerprint(self, text: str) -> frozenset:
        words = [w for w in _normalize_text(text).split() if len(w) > 4]
        if len(words) < NGRAM_SIZE:
            return frozenset(words)
        ngrams = set()
        for i in range(len(words) - NGRAM_SIZE + 1):
            ngrams.add(tuple(words[i:i + NGRAM_SIZE]))
        return frozenset(ngrams)

    def _extract_intent_verb_target(self, text: str) -> Optional[Tuple[str, str]]:
        lower = _normalize_text(text)
        for cat, verbs in INTENT_VERB_CATEGORIES.items():
            if any(v in lower for v in verbs):
                for verb in verbs:
                    pos = lower.find(verb)
                    if pos >= 0:
                        after = lower[pos + len(verb):pos + len(verb) + 60]
                        targets = [w for w in after.split() if len(w) > 5]
                        if targets:
                            return (cat, targets[0])
        return None

    def _is_semantic_duplicate(self, normalized: str) -> bool:
        fp = self._compute_question_fingerprint(normalized)
        for prev_fp in self._question_ngram_fingerprints:
            if len(fp) == 0 or len(prev_fp) == 0:
                continue
            overlap = len(fp & prev_fp) / max(len(fp), len(prev_fp))
            if overlap >= 0.50:
                return True
        ivt = self._extract_intent_verb_target(normalized)
        if ivt and ivt in self._intent_verb_targets:
            return True
        return False

    # =========================================================================
    # IMPROVEMENT D: Coverage matrix
    # =========================================================================

    def _update_coverage_matrix(self, user_text: str) -> None:
        lower = user_text.lower()
        for tool in self._jd_tools:
            tool_words = tool.lower().split()
            if any(w in lower for w in tool_words if len(w) > 3):
                self._coverage_matrix["TECH"][tool] = True
        for proj in self.session_notes.get("key_projects", []):
            if proj.lower()[:20] in lower:
                self._coverage_matrix["PROJECT"][proj] = True
        for dim, v in self._behavioral_coverage.items():
            self._coverage_matrix["BEHAVIORAL"][dim] = v
        for dim, v in self._decision_coverage.items():
            self._coverage_matrix["DECISION"][dim] = v

    def _coverage_gap_report(self, phase: str) -> str:
        if phase in ("OPENING", "CANDIDATE_QUESTIONS", "FINAL_CHECK", "CLOSING"):
            return "N/A"
        gaps = {}
        for quadrant, items in self._coverage_matrix.items():
            uncovered = [k for k, v in items.items() if not v]
            gaps[quadrant] = uncovered
        phase_quadrant_priority = {
            "JOB_ALIGNED_EXPLORATION": ["TECH"],
            "PROJECT_DEEP_DIVE":       ["PROJECT", "TECH"],
            "TECHNICAL_DEPTH":         ["TECH", "DECISION"],
            "SOFT_SKILLS_BEHAVIORAL":  ["BEHAVIORAL", "DECISION"],
        }
        priority_quadrants = phase_quadrant_priority.get(phase, COVERAGE_QUADRANTS)
        report_parts = []
        for q in priority_quadrants:
            unc = gaps.get(q, [])
            if unc:
                report_parts.append(f"{q}: {unc[:4]}")
        return " | ".join(report_parts) if report_parts else "All quadrants covered"

    # =========================================================================
    # IMPROVEMENT E: Human voice post-processing
    # =========================================================================

    def _humanize_question(self, question: str, user_text: str, candidate_sentiment: str) -> str:
        if self.target_lang == "Arabe":
            question = re.sub(r"\?+", "?", question)
            question = re.sub(r"\.+", ".", question)
            if not question.rstrip().endswith(("?", "؟")):
                ar_question_words = ["كيف", "ما", "لماذا", "متى", "أين", "من", "هل",
                                     "أخبرني", "صف", "حدثني", "خذني", "أعطني"]
                if any(kw in question for kw in ar_question_words):
                    question = question.rstrip(".") + "؟"
            question = re.sub(r"^(إذن،?\s*|حسناً،?\s*|بالتأكيد،?\s*|طبعاً،?\s*)", "", question, flags=re.I)
            return question.strip()

        question = re.sub(r"\?+", "?", question)
        question = re.sub(r"\.+", ".", question)

        if not question.rstrip().endswith(("?", "؟")):
            if any(kw in question.lower() for kw in [
                "what", "how", "why", "when", "tell me", "describe", "walk",
                "quel", "comment", "pourquoi", "décrivez",
            ]):
                question = question.rstrip(".") + "?"

        emotional_phrases = ["failed", "difficult", "hard", "stressful", "frustrated",
                             "échec", "difficile", "dur", "stressant"]
        if (
            candidate_sentiment == "stressed"
            or any(p in user_text.lower() for p in emotional_phrases)
        ):
            bridges = {
                "Français": ["Sur ce projet difficile, ", "Dans ce contexte, ", "Et malgré cela, "],
                "Anglais":  ["On that note, ", "Building on that, ", "Given that context, "],
            }
            lang_bridges = bridges.get(self.target_lang, bridges["Anglais"])
            first_word = question.split()[0].lower() if question.split() else ""
            generic_starters = ["on", "in", "and", "dans", "sur"]
            if first_word not in generic_starters:
                bridge = lang_bridges[len(self.turns) % len(lang_bridges)]
                question = bridge + question[0].lower() + question[1:]

        question = re.sub(r"^(so,?\s*)?let me ask you (this:?\s*|something:?\s*)", "", question, flags=re.I)
        question = re.sub(r"^(d'accord,?\s*)?permettez-moi de vous demander:?\s*", "", question, flags=re.I)

        return question.strip()

    # =========================================================================
    # FIX-AR-9: Language drift detection
    # =========================================================================

    def _detect_language_drift(self, question: str) -> bool:
        if self.target_lang != "Arabe":
            return False
        if re.search(r"[\u4e00-\u9fff]", question):
            return True
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", question))
        total_alpha = len(re.findall(r"[A-Za-z\u0600-\u06FF]", question))
        if total_alpha == 0:
            return False
        arabic_ratio = arabic_chars / total_alpha
        french_matches = FRENCH_DRIFT_PATTERN.findall(question)
        tech_exceptions = {"api", "ml", "ai", "data", "pipeline", "deploy", "log",
                           "commit", "merge", "branch", "cache", "server", "client",
                           "backend", "frontend", "docker", "cloud", "query", "index"}
        non_tech_french = [w for w in french_matches if w.lower() not in tech_exceptions]
        return len(non_tech_french) >= 2 or (arabic_ratio < 0.30 and len(question.split()) > 5)

    # =========================================================================
    # IMPROVEMENT G: Humanness quality gate
    # =========================================================================

    def _score_question_humanness(self, question: str) -> int:
        score = 0
        words = question.split()
        wc = len(words)
        if 8 <= wc <= 55:
            score += 1
        if question.rstrip().endswith(("?", "؟")):
            score += 1
        first_5 = " ".join(words[:5]).lower()
        forbidden_starts = [
            "noted", "good", "thank you", "i see", "understood",
            "absolutely", "certainly", "of course", "sure",
            "bien noté", "d'accord", "je vois", "parfait",
            "حسناً", "بالتأكيد", "طبعاً",
        ]
        if not any(first_5.startswith(f) for f in forbidden_starts):
            score += 1
        natural_starters = [
            "what", "how", "why", "when", "which", "who", "where", "could", "can", "did",
            "tell", "walk", "describe", "give", "help", "on that", "and —", "so —",
            "quel", "comment", "pourquoi", "quand", "décrivez", "dites", "donnez",
            "كيف", "ما", "لماذا", "متى", "أين", "أخبرني", "صف", "حدثني", "خذني",
            "وش", "إيش", "ليش", "احكيلي", "أعطني", "في",
        ]
        if any(question.lower().startswith(s) for s in natural_starters):
            score += 1
        if question.count("?") + question.count("؟") == 1:
            score += 1
        if wc > 0 and len(set(words)) / wc > 0.6:
            score += 1
        return score

    def _ensure_human_quality(self, question: str, phase: str, user_text: str) -> str:
        score = self._score_question_humanness(question)
        drift = self._detect_language_drift(question)

        if score >= MIN_HUMANNESS_SCORE and not drift:
            return question

        if self.target_lang == "Arabe":
            lang_instruction = (
                "اكتب بالعربية السعودية فقط. "
                "المصطلحات التقنية الإنجليزية مسموحة. "
                "ممنوع تماماً استخدام الفرنسية أو الإنجليزية العادية."
            )
        elif self.target_lang == "Français":
            lang_instruction = "Écris en français uniquement. Les termes techniques anglais sont autorisés."
        else:
            lang_instruction = "Write in English only. Technical terms in any language are fine."

        rewrite_prompt = (
            f"Rewrite the following interview question to sound more natural and human.\n"
            f"Language instruction: {lang_instruction}\n"
            f"Max 2 sentences. End with a single question mark. "
            f"No preamble, no praise, no filler words.\n\n"
            f"ORIGINAL: {question}\n\nREWRITTEN:"
        )
        try:
            raw = self._call_llm(rewrite_prompt)
            rewritten = self._extract_final_question(raw)
            new_score = self._score_question_humanness(rewritten)
            new_drift = self._detect_language_drift(rewritten)
            if rewritten and new_score > score and not new_drift:
                return rewritten
        except Exception:
            pass
        return question

    # =========================================================================
    # FIX-AR-1: _call_llm — inject system-level language message
    # =========================================================================

    def _call_llm(
        self,
        prompt: str,
        system_override: Optional[str] = None,
        phase: Optional[str] = None,
        include_few_shot: bool = True,
    ) -> str:
        system_msg = system_override or self._build_system_prompt(
            phase=phase,
            include_few_shot=include_few_shot,
        )

        opts = {
            "temperature":    0.35,
            "top_p":          0.9,
            "num_predict":    300,
            "repeat_penalty": 1.1,
            "top_k":          40,
            "num_gpu": DEVICE_CONFIG.get("ollama_num_gpu", 20) if DEVICE_CONFIG["use_gpu"] else 0,
            "num_ctx": DEVICE_CONFIG.get("ollama_num_ctx", 4096),
        }

        def _collect_stream(stream_iter) -> str:
            parts = []
            deadline = time.time() + 90
            for chunk in stream_iter:
                if time.time() > deadline:
                    print("WARNING _call_llm: timeout 90s — reponse partielle retournee")
                    break
                if hasattr(chunk, "message"):
                    parts.append(chunk.message.content or "")
                elif isinstance(chunk, dict):
                    parts.append(
                        chunk.get("message", {}).get("content", "")
                        or chunk.get("response", "")
                    )
            return "".join(parts).strip()

        if system_msg:
            try:
                stream = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": prompt},
                    ],
                    options=opts,
                    stream=True,
                )
                return _collect_stream(stream)
            except Exception as e:
                print(f"WARNING _call_llm chat stream: {e} — fallback generate")
                combined = f"{system_msg}\n\n{prompt}"
                try:
                    stream = self.client.generate(
                        model=self.model_name,
                        prompt=combined,
                        options=opts,
                        stream=True,
                    )
                    return _collect_stream(stream)
                except Exception as e2:
                    print(f"WARNING _call_llm generate stream: {e2}")
                    return ""
        else:
            try:
                stream = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=opts,
                    stream=True,
                )
                return _collect_stream(stream)
            except Exception as e:
                print(f"WARNING _call_llm generate stream: {e}")
                return ""
    # =========================================================================
    # FIX-AR-10: _postprocess_question — adds drift detection
    # =========================================================================

    def _postprocess_question(self, text: str, phase: str) -> str:
        if len(text) < 8:
            return self._fallback_question(phase)

        normalized = self._normalize_question(text)

        is_duplicate = normalized in self.asked_question_signatures

        if not is_duplicate:
            new_words = {w for w in normalized.split() if len(w) > 4}
            if new_words:
                for prev in self.last_questions_normalized[-8:]:
                    prev_words = {w for w in prev.split() if len(w) > 4}
                    if prev_words:
                        overlap = len(new_words & prev_words) / len(new_words)
                        if overlap >= 0.55:
                            is_duplicate = True
                            break

        if not is_duplicate:
            is_duplicate = self._is_semantic_duplicate(normalized)

        if is_duplicate:
            text = self._fallback_question(phase)
            normalized = self._normalize_question(text)

        if self._detect_language_drift(text):
            text = self._ensure_human_quality(text, phase, "")
            if self._detect_language_drift(text):
                text = self._fallback_question(phase)
                normalized = self._normalize_question(text)

        self.asked_question_signatures.add(normalized)
        self.last_questions_normalized.append(normalized)
        self.last_questions_normalized = self.last_questions_normalized[-20:]

        fp = self._compute_question_fingerprint(normalized)
        self._question_ngram_fingerprints.add(fp)

        ivt = self._extract_intent_verb_target(normalized)
        if ivt:
            self._intent_verb_targets.add(ivt)

        if self.vision_stress_flag and phase in {"TECHNICAL_DEPTH", "SOFT_SKILLS_BEHAVIORAL"}:
            text = self._soften_if_needed(text)
        return text

    # =========================================================================
    # Phase directives
    # =========================================================================

    def _phase_directive(self, phase: str, user_text: str, answer_quality: str) -> str:
        if phase == "CANDIDATE_QUESTIONS":
            return self._directive_candidate_questions(user_text)

        if phase == "FINAL_CHECK":
            return self._directive_final_check()

        if phase == "CLOSING":
            return self._directive_closing()
        if phase == "OPENING":
            return self._directive_opening()
        if phase == "JOB_ALIGNED_EXPLORATION":
            return self._directive_jd_exploration()
        if phase == "PROJECT_DEEP_DIVE":
            return self._directive_project_deep_dive()
        if phase == "TECHNICAL_DEPTH":
            return self._directive_technical_depth(user_text, answer_quality)
        if phase == "SOFT_SKILLS_BEHAVIORAL":
            return self._directive_soft_skills()
    def _directive_opening(self) -> str:
        last_answer = self._last_candidate_message()
        if last_answer:
            cv_signals = self._extract_cv_strong_signals(last_answer)
            # Extraire aussi les projets mentionnés explicitement
            mentioned_projects = self._extract_mentioned_projects(last_answer)
            # Exclure tout projet qui vient de la JD et non du candidat
            jd_text_lower = self.doc_texts.get("job_offer", "").lower()
            candidate_projects = [
                p for p in mentioned_projects
                if p.lower()[:30] not in jd_text_lower or p.lower()[:30] in self.doc_texts.get("cv", "").lower()
            ]
            projects_str = ", ".join(f'"{p}"' for p in candidate_projects[:4]) if candidate_projects else "none detected"

            if cv_signals:
                signal = cv_signals[0]
                return (
                    f"The candidate just introduced themselves and mentioned: {signal['quote']}.\n"
                    f"Projects explicitly mentioned: {projects_str}\n\n"
                    f"MANDATORY RULES:\n"
                    f"1. Do NOT say 'Noted', 'Great', or any passive acknowledgement.\n"
                    f"2. You MUST name a specific project from this list in your question: {projects_str}\n"
                    f"   Example: 'On your RAG conversational system — ...' or 'In your sales prediction project — ...'\n"
                    f"3. Ask ONE sharp follow-up anchored to that named project.\n"
                    f"4. NEVER ask a generic question like 'how did you integrate these components' without naming the project first.\n"
                    f"5. Write in {self.target_lang}."
                )
            return (
                f"The candidate just introduced themselves.\n"
                f"Projects explicitly mentioned: {projects_str}\n\n"
                f"MANDATORY RULES:\n"
                f"1. You MUST reference a specific project by name in your question.\n"
                f"2. Ask ONE focused follow-up anchored to that named project.\n"
                f"3. NEVER ask a generic question without naming the project.\n"
                f"4. Write in {self.target_lang}."
            )

        jd_tools_str = ", ".join(self._jd_tools[:3]) if self._jd_tools else "the role's technologies"
        return (
            f"The candidate hasn't introduced themselves yet.\n"
            f"Ask them to briefly walk you through their background and what drew them to this role.\n"
            f"Anchor to: {jd_tools_str}.\n"
            f"Write in {self.target_lang}."
        )

    def _directive_jd_exploration(self) -> str:
        current = self._current_tool_focus
        followups_used = self._tool_followup_count.get(current, 0)

        if current and followups_used >= MAX_TOOL_FOLLOWUPS:
            self._tools_fully_done.add(current)
            current = ""
            self._current_tool_focus = ""

        unexplored = [t for t in self._tool_rotation_order
                      if t not in self._tools_fully_done and t != current]
        needs_why = [t for t in self.explored_jd_tools
                     if t not in self._jd_tool_why_asked
                     and t not in self._tools_fully_done
                     and t != current]

        if not current:
            if needs_why:
                current = needs_why[0]
            elif unexplored:
                current = unexplored[0]
            else:
                current = self._jd_tools[0] if self._jd_tools else ""
            self._current_tool_focus = current
            self._tool_followup_count.setdefault(current, 0)

        followups_used = self._tool_followup_count.get(current, 0)
        linked_projects = self._tool_project_map.get(current, [])
        fully_done_list = sorted(self._tools_fully_done)
        upcoming = [t for t in self._tool_rotation_order
                    if t not in self._tools_fully_done and t != current][:3]

        if linked_projects and followups_used >= 1:
            proj = linked_projects[0]
            if self.target_lang == "Arabe":
                project_hint = f"PROJECT BRIDGE: 'في مشروعك «{proj}»، استخدمت {current} — وش اللي بنيته بالضبط؟'"
            elif self.target_lang == "Français":
                project_hint = f"PROJECT BRIDGE: 'Dans votre projet «{proj}», vous avez utilisé {current} — qu'avez-vous concrètement réalisé ?'"
            else:
                project_hint = f"PROJECT BRIDGE: 'In your project «{proj}», you used {current} — what specifically did you build or solve with it ?'"
        else:
            project_hint = f"Ask a deeper technical question about «{current}»: a failure mode, a scale constraint, or a tuning decision."

        if followups_used == 0:
            step_instruction = f"STEP 1 — PRACTICAL USAGE: Ask about the candidate's real experience with «{current}». How did they use it? At what scale?"
        elif followups_used == 1:
            step_instruction = f"STEP 2 — WHY DECISION: Ask WHY «{current}» was chosen over alternatives."
        elif followups_used == 2:
            step_instruction = f"STEP 3 — PROJECT BRIDGE OR DEEPER:\n  {project_hint}"
        else:
            next_tool = upcoming[0] if upcoming else "next available tool"
            step_instruction = f"STEP 4 — MANDATORY SWITCH to «{next_tool}»."

        return (
            "PHASE: JOB-ALIGNED TECHNICAL EXPLORATION\n\n"
            f"ALL JD tools: {self._tool_rotation_order[:12]}\n"
            f"Fully done: {fully_done_list}\n"
            f"CURRENT TOOL: «{current}» (follow-ups used: {followups_used}/{MAX_TOOL_FOLLOWUPS})\n"
            f"Next in queue: {upcoming}\n\n"
            f"{step_instruction}\n\n"
            f"Write your question in {self.target_lang}.\n"
        )

    def _directive_project_deep_dive(self) -> str:
        projects = self.session_notes.get("key_projects", [])[:5]
        dived = sorted(self.deep_dived_projects)
        remaining = [p for p in projects if p not in self.deep_dived_projects]
        matched = self.session_notes.get("matched_skills", [])
        return (
            "PHASE: PROJECT DEEP DIVE\n"
            f"Key projects: {projects}\n"
            f"Already dived: {dived}\n"
            f"Remaining: {remaining[:3]}\n"
            f"JD-matched skills: {matched[:6]}\n\n"
            "Structured angles: architecture → challenges → trade-offs → metrics → ownership.\n"
            "ANY metric mentioned → ask HOW measured and the baseline.\n"
            f"Write your question in {self.target_lang}."
        )

    def _directive_technical_depth(self, user_text: str, answer_quality: str) -> str:
        return (
            "PHASE: TECHNICAL EVALUATION DEPTH\n"
            "Evaluation angles: CHALLENGE → WHY+HOW → METRICS → FAILURES → OWNERSHIP.\n"
            f"Answer quality: {answer_quality}\n"
            f"Write your question in {self.target_lang}."
        )

    def _directive_soft_skills(self) -> str:
        uncovered = [d for d, v in self._behavioral_coverage.items() if not v]
        return (
            "PHASE: SOFT SKILLS & BEHAVIORAL ASSESSMENT\n"
            f"Dimensions not yet assessed: {uncovered}\n"
            "Ask for CONCRETE STAR stories. Generic answer → push back.\n"
            f"Write your question in {self.target_lang}."
        )


    def _build_candidate_question_answer(self, user_text: str) -> str:
        company_info = self.doc_texts.get("company_info", "").strip()
        job_offer = self.doc_texts.get("job_offer", "").strip()

        context = "\n\n".join([
            f"COMPANY:\n{company_info[:2000]}" if company_info else "",
            f"JOB:\n{job_offer[:2000]}" if job_offer else "",
        ]).strip()

        if self.target_lang == "Français":
            no_info = (
                "Je n'ai pas les détails précis sur ce point pour le moment, "
                "mais je serai ravi(e) de vous revenir avec une réponse complète. "
                "Avez-vous d'autres questions ?"
            )
        elif self.target_lang == "Arabe":
            no_info = (
                "ما عندي التفاصيل الدقيقة على هذه النقطة الآن، "
                "لكن بقدر أرجع لك بإجابة كاملة. "
                "هل عندك أي أسئلة أخرى؟"
            )
        else:
            no_info = (
                "I don't have the specific details on that right now, "
                "but I'll be happy to follow up with a complete answer. "
                "Do you have any other questions?"
            )

        if not context:
            return no_info

        prompt = f"""
    Candidate question:
    {user_text}

    Available context:
    {context}

    Instructions:
    - Answer the question directly
    - Use ONLY the context
    - If not enough info → say you don't know
    - DO NOT ask a new question except: "any other questions?"
    - DO NOT reverse the question
    - Keep it short and clear
    """

        raw = self._call_llm(
            prompt,
            system_override="You are a recruiter answering candidate questions. Never guess. Never reverse the question.",
            include_few_shot=False,
        )

        answer = raw.strip()

        bad_patterns = [
            "what do you think",
            "could you",
            "can you",
            "would you",
            "how would you",
        ]

        if any(p in answer.lower() for p in bad_patterns):
            return no_info

        return answer if answer else no_info

    def _directive_candidate_questions(self, user_text: str) -> str:
        has_context = bool(
            self.doc_texts.get("company_info", "").strip()
            or self.doc_texts.get("job_offer", "").strip()
        )

        if not self.candidate_questions_started:
            return (
                "PHASE: CANDIDATE QUESTIONS\n"
                "This phase is ONLY for candidate questions.\n"
                "Ask explicitly whether the candidate has any questions about the role, team, scope, KPIs, or hiring process.\n"
                "Do NOT ask a technical follow-up.\n"
                "Do NOT challenge the candidate.\n"
                "Do NOT introduce a new topic.\n"
                "Do not answer anything yet. Just ask that question."
            )

        if self._detecting_question_decline(user_text):
            self._candidate_declined_questions = True
            return (
                "PHASE: CANDIDATE QUESTIONS\n"
                "The candidate has no questions.\n"
                "Do NOT ask any new interview question.\n"
                "Move to the next phase."
            )

        return (
            "PHASE: CANDIDATE QUESTIONS\n"
            "This phase is ONLY for answering the candidate's question.\n"
            f"Available context: {'YES' if has_context else 'NO / LIMITED'}.\n"
            "Answer directly.\n"
            "If you don't know → say you don't know.\n"
            "Do NOT reverse the question.\n"
            "After answering → ask if they have other questions."
        )



    def _directive_final_check(self) -> str:
        if self.target_lang == "Arabe":
            phrase = "قبل ما نختم، هل فيه شيء تحب تضيفه وما غطيناه؟"
        elif self.target_lang == "Français":
            phrase = "Avant de conclure, y a-t-il quelque chose que vous souhaitez ajouter et que nous n'avons pas couvert ?"
        else:
            phrase = "Before we close, is there anything you'd like to add that we have not covered?"

        return (
            "PHASE: FINAL CHECK\n"
            "This phase is only for one final wrap-up question.\n"
            "Do NOT ask a technical question.\n"
            "Do NOT ask about tools, metrics, projects, campaigns, or decisions.\n"
            f"Ask exactly this: {phrase}"
        )
    def _directive_closing(self) -> str:
        mandatory_close = {
            "Français": "Merci d'avoir été présent(e) aujourd'hui. Nous reviendrons vers vous avec la réponse finale dans quelques jours.",
            "Anglais":  "Thank you for being with us today. We will get back to you with the final decision within the next few days.",
            "Arabe":    "شكراً لحضورك اليوم. راح نرجع لك بالرد النهائي خلال الأيام الجاية.",
        }[self.target_lang]

        return (
            "PHASE: CLOSING\n"
            "This phase is closing only.\n"
            "Do NOT ask any new question.\n"
            "Do NOT introduce any new topic.\n"
            f"Say exactly this closing message: {mandatory_close}"
        )

    # =========================================================================
    # Supporting directives
    # =========================================================================

    def _global_behavioral_rules(self) -> str:
        covered_str = ", ".join(sorted(self.covered_topics)) or "none yet"
        consecutive_vague = 0
        for q in reversed(self._answer_quality_history):
            if q == "VAGUE":
                consecutive_vague += 1
            else:
                break
        vague_warn = (
            f"\nWARNING: {consecutive_vague} consecutive VAGUE answers. Use Tier 2+ relance."
            if consecutive_vague >= 3 else ""
        )
        return (
            "1.  ONE question per turn.\n"
            "2.  Anchor to CV and JD.\n"
            "3.  Verify experience, not definitions.\n"
            "4.  Personal ownership: what THEY designed, decided, implemented.\n"
            f"5.  NEVER repeat. Covered: [{covered_str}].\n"
            "6.  NO praise, NO fillers.\n"
            "7.  1–2 sentences max.\n"
            f"8.  ALWAYS write in {self.target_lang}.\n"
            "9.  The candidate may pause, hesitate, or repeat themselves. This is natural. "
            "Do not interpret normal speech disfluencies as lack of competence.\n"
            "10. When the candidate takes a long pause, wait for them to finish. "
            "    Your response will be triggered only after 10 seconds of silence.\n"
            + vague_warn
        )

    def _anti_repeat_directive(self) -> str:
        recent = self.last_questions_normalized[-8:]
        return (
            f"RECENT QUESTIONS (last 8 — do NOT rephrase):\n  {recent}\n\n"
            f"TOOLS EXPLORED: {sorted(self.explored_jd_tools)}\n"
            f"PROJECTS DIVED: {sorted(self.deep_dived_projects)}\n"
            f"TOPICS COVERED: {sorted(self.covered_topics)}\n\n"
            f"Write the new question in {self.target_lang}."
        )

    def _answer_quality_directive(self, quality: str) -> str:
        covered_str = ", ".join(sorted(self.covered_topics)) or "none yet"
        force_switch = (
            quality in ("STRONG", "GOOD") and self.followup_depth >= 2
        ) or (
            self.current_topic_focus in self._get_saturated_topics())
        return {
            "VAGUE": f"VAGUE: Push hard for specificity. Write in {self.target_lang}.",
            "INCOMPLETE": f"INCOMPLETE: Probe for ownership, rationale, or metric. Write in {self.target_lang}.",
            "GOOD": (
                "GOOD — do NOT praise. "
                + (f"SWITCH to NEW uncovered topic (covered: [{covered_str}])." if force_switch else "ONE targeted deep-dive.")
                + f" Write in {self.target_lang}."
            ),
            "STRONG": (
                f"STRONG — pivot IMMEDIATELY to DIFFERENT topic (covered: [{covered_str}]). "
                f"Write in {self.target_lang}."
            ),
        }.get(quality, "")

    def _build_challenge_directive(self, answer_quality: str, user_text: str) -> str:
        if answer_quality not in ("STRONG", "GOOD"):
            return ""
        metric_match = re.search(
            r"(\d+[\s]?(?:%|percent|ms|tb|gb|k\b|m\b|x\b)|\d+[\s]?(?:times|fold|reduction|improvement|faster|slower))",
            user_text.lower()
        )
        if not metric_match:
            return ""
        if self._metrics_probed and self._turns_since_metrics_probe < 3:
            return ""
        metric_snippet = user_text[max(0, user_text.lower().find(metric_match.group()) - 20):][:80].strip()
        templates = {
            "Français": f"Candidat a mentionné : '...{metric_snippet}...'. Challengez ce chiffre. Comment mesuré ? Quelle baseline ?",
            "Anglais":  f"Candidate mentioned: '...{metric_snippet}...'. Challenge this metric. How measured? What was the baseline?",
            "Arabe":    f"المرشح ذكر: '...{metric_snippet}...'. تحدّ هذا الرقم. كيف قياسه؟ ما الأساس؟ اكتب بالعربية.",
        }
        return templates.get(self.target_lang, templates["Anglais"])

    def _metrics_probe_directive(self, phase: str) -> str:
        if phase not in ("JOB_ALIGNED_EXPLORATION", "PROJECT_DEEP_DIVE", "TECHNICAL_DEPTH"):
            return ""
        if self._metrics_probed:
            return ""
        relevant_turns = [
            t for t in self.turns
            if t.phase == phase and self._is_candidate_speaker(t.speaker)
        ]
        if len(relevant_turns) < METRICS_PROBE_AFTER_TURNS:
            return ""
        return (
            f"\nMETRICS PROBE (MANDATORY): Ask for a specific number and HOW it was measured. "
            f"Write in {self.target_lang}.\n"
        )

    def _build_seniority_directive(self, phase: str) -> str:
        if phase not in ("JOB_ALIGNED_EXPLORATION", "PROJECT_DEEP_DIVE", "TECHNICAL_DEPTH", "SOFT_SKILLS_BEHAVIORAL"):
            return ""
        if self._job_seniority_level == "senior":
            return "SENIOR: Architecture ownership, strategic decisions, mentoring, ambiguity."
        if self._job_seniority_level == "junior":
            return "JUNIOR: Fundamentals, learning agility, initiative, growth mindset."
        return ""

    def _build_job_adaptation_directive(self, phase: str) -> str:
        directives = []
        tech_phases = ("JOB_ALIGNED_EXPLORATION", "PROJECT_DEEP_DIVE", "TECHNICAL_DEPTH")
        if self._job_domain_directive and phase in tech_phases:
            directives.append(f"DOMAIN FOCUS: {self._job_domain_directive}")
        if "team_leadership" in self._job_required_behaviors and phase == "SOFT_SKILLS_BEHAVIORAL":
            directives.append("LEADERSHIP REQUIRED: Probe team management and feedback.")
        return "\n".join(directives)

    def _build_signal_validation_directive(self) -> str:
        if not self._unvalidated_claims:
            return ""
        priority_order = ["ownership_vague", "impact_claim", "leadership_claim", "scale_claim", "credential_claim"]
        target = None
        for pt in priority_order:
            for claim in self._unvalidated_claims:
                if claim["type"] == pt:
                    target = claim
                    break
            if target:
                break
        if not target:
            target = self._unvalidated_claims[0]

        ctype = target["type"]
        excerpt = target["excerpt"]
        ar_mandate = " اكتب السؤال بالعربية فقط."
        fr_mandate = " Question en français."
        en_mandate = " Write in English."

        templates = {
            "ownership_vague": {
                "Français": f"'{excerpt}' (collectif). Demandez ce que LUI SEUL a conçu ou livré." + fr_mandate,
                "Anglais":  f"'{excerpt}' (collective). Ask what THEY personally owned or delivered." + en_mandate,
                "Arabe":    f"'{excerpt}' (جماعي). اسأله إيش اللي هو شخصياً صمّمه أو قرره." + ar_mandate,
            },
            "impact_claim": {
                "Français": f"'{excerpt}'. Comment mesuré ? Quelle baseline ?" + fr_mandate,
                "Anglais":  f"'{excerpt}'. How measured? What was the baseline?" + en_mandate,
                "Arabe":    f"'{excerpt}'. كيف قياسه؟ ما الأساس؟" + ar_mandate,
            },
            "leadership_claim": {
                "Français": f"'{excerpt}'. Comment géré une sous-performance ?" + fr_mandate,
                "Anglais":  f"'{excerpt}'. How did they handle underperformance?" + en_mandate,
                "Arabe":    f"'{excerpt}'. كيف تعامل مع ضعف أداء أحد؟" + ar_mandate,
            },
            "scale_claim": {
                "Français": f"'{excerpt}'. Quels défis concrets ça a créés ?" + fr_mandate,
                "Anglais":  f"'{excerpt}'. What concrete problems did that create?" + en_mandate,
                "Arabe":    f"'{excerpt}'. وش المشاكل الملموسة اللي سببها؟" + ar_mandate,
            },
            "credential_claim": {
                "Français": f"'{excerpt}'. Dans quel contexte et sur quels critères ?" + fr_mandate,
                "Anglais":  f"'{excerpt}'. In what context and against what criteria?" + en_mandate,
                "Arabe":    f"'{excerpt}'. في أي سياق وبناءً على أي معايير؟" + ar_mandate,
            },
        }
        fallback = {
            "Français": f"'{excerpt}'. Demandez une preuve concrète." + fr_mandate,
            "Anglais":  f"'{excerpt}'. Ask for concrete evidence." + en_mandate,
            "Arabe":    f"'{excerpt}'. اطلب دليلاً ملموساً." + ar_mandate,
        }
        chosen = templates.get(ctype, fallback)
        directive = chosen.get(self.target_lang, chosen.get("Anglais", ""))
        if directive:
            self._unvalidated_claims = [c for c in self._unvalidated_claims if c["type"] != ctype]
            self._challenged_claim_types.add(ctype)
        return directive

    def _build_behavioral_assessment_directive(self, phase: str) -> str:
        if phase != "SOFT_SKILLS_BEHAVIORAL":
            return ""
        soft_turns = [
            t for t in self.turns
            if t.phase == "SOFT_SKILLS_BEHAVIORAL" and self._is_candidate_speaker(t.speaker)
        ]
        directives = []
        if len(soft_turns) >= PROBE_DEPTH_CONFIG["behavioral_min_turns"] and self._behavioral_stories_count == 0:
            uncovered = [d for d, v in self._behavioral_coverage.items() if not v]
            if uncovered:
                dim = uncovered[0]
                if dim not in self._behavioral_asked:
                    # Récupérer la question en anglais pour l'arabe, sinon dans la langue cible
                    q_lang = "Anglais" if self.target_lang == "Arabe" else self.target_lang
                    q = BEHAVIORAL_QUESTIONS.get(dim, {}).get(q_lang, "")
                    if q:
                        self._behavioral_asked.add(dim)
                        self._behavioral_coverage[dim] = True
                        if self.target_lang == "Arabe":
                            q = self._ot_agent.generate_from_english(q, phase=phase)
                        directives.append(f"No behavioral story yet. Ask STAR question about '{dim}':\n{q}")
        return "\n".join(directives)

    def _build_decision_making_directive(self, phase: str) -> str:
        if phase not in ("PROJECT_DEEP_DIVE", "TECHNICAL_DEPTH"):
            return ""
        if self._decision_shown_per_phase.get(phase, False):
            return ""
        relevant_turns = [
            t for t in self.turns
            if t.phase == phase and self._is_candidate_speaker(t.speaker)
        ]
        if len(relevant_turns) < PROBE_DEPTH_CONFIG["decision_probe_after_turns"]:
            return ""
        uncovered = [d for d, v in self._decision_coverage.items() if not v and d not in self._decision_asked]
        if not uncovered:
            return ""
        priority = next(
            (d for d in ["tradeoff", "why_technology", "incident_decision", "rollback"] if d in uncovered),
            uncovered[0]
        )
        # Récupérer la question en anglais pour l'arabe, sinon dans la langue cible
        q_lang = "Anglais" if self.target_lang == "Arabe" else self.target_lang
        q = DECISION_MAKING_QUESTIONS.get(priority, {}).get(q_lang, "")
        if q:
            self._decision_asked.add(priority)
            self._decision_coverage[priority] = True
            if self.target_lang == "Arabe":
                q = self._ot_agent.generate_from_english(q, phase=phase)
            return f"No decision reasoning yet. Ask:\n{q}"
        return ""

    def _time_pressure_directive(self, phase: str) -> str:
        time_left = self.get_time_remaining()
        time_fraction = time_left / max(1, self.duration_minutes)
        if time_fraction > 0.25:
            return ""
        if phase in ("CLOSING", "FINAL_CHECK"):
            return ""
        return (
            f"⏱ TIME PRESSURE: Only {time_left:.1f} min remaining. "
            f"Ask ONE final consolidating question. Write in {self.target_lang}."
        )

    # =========================================================================
    # Decline detection
    # =========================================================================

    def _detecting_question_decline(self, text: str) -> bool:
        lower = text.lower().strip()

        # Si le texte contient une vraie question → jamais un déclin
        has_real_question = (
            text.count("?") + text.count("؟") >= 1
            or any(k in lower for k in [
                "what", "how", "why", "when", "where", "who", "which",
                "quel", "comment", "pourquoi", "quand", "où",
                "كيف", "ما", "لماذا", "متى", "أين", "هل عندكم", "هل يمكن",
                "i'd like to know", "i would like to know", "i'm curious",
                "can you tell me", "could you tell me", "j'aimerais savoir",
            ])
        )
        if has_real_question:
            return False

        # Patterns explicites de refus
        decline_cues = [
            "no thank", "not really", "nothing", "nope",
            "i'm good", "i'm fine", "that's all", "that's it", "all good",
            "all is clear", "all clear", "no more", "rien d'autre",
            "c'est tout", "ça me suffit", "pas de question",
            "لا", "ما عندي", "كل شيء واضح", "لا شيء",
            "don't have", "do not have", "no other", "no more questions",
            "no further", "i have no", "i've got no", "that's everything",
            "that is all", "nothing else", "no additional",
            "je n'ai pas", "je n'ai plus", "pas d'autres", "aucune autre",
            "ما عندي أسئلة", "ما في أسئلة", "خلاص", "ما عندي شيء",
        ]

        if any(c in lower for c in decline_cues):
            return True

        # Phrases très courtes sans question → probablement un déclin
        if len(lower.split()) <= 5:
            short_cues = ["no", "non", "لا", "nope", "nothing"]
            if any(lower.startswith(c) for c in short_cues):
                return True

        return False

    # =========================================================================
    # Deterministic closing
    # =========================================================================

    def _build_closing(self) -> str:
        skills = [s for s in list(self.covered_skills)[:3] if len(s) > 2]
        projects = list(self.covered_projects)[:2]

        personal_sentence = ""
        if skills or projects:
            refs = []
            if projects:
                if self.target_lang == "Anglais":
                    refs.append(f"your work on {', '.join(projects[:1])}")
                elif self.target_lang == "Français":
                    refs.append(f"votre travail sur {', '.join(projects[:1])}")
                else:
                    refs.append(f"عملك على {', '.join(projects[:1])}")
            if skills:
                if self.target_lang == "Anglais":
                    refs.append(f"your expertise in {', '.join(skills[:2])}")
                elif self.target_lang == "Français":
                    refs.append(f"votre maîtrise de {', '.join(skills[:2])}")
                else:
                    refs.append(f"خبرتك في {', '.join(skills[:2])}")
            ref_str = (
                " and ".join(refs) if self.target_lang == "Anglais" else
                " et ".join(refs) if self.target_lang == "Français" else
                " و".join(refs)
            )
            if self.target_lang == "Arabe":
                prompt = f"اكتب جملة واحدة دافئة بالعربية السعودية تُشير إلى: {ref_str}. بحد أقصى 20 كلمة. نص عادي فقط."
            elif self.target_lang == "Français":
                prompt = f"Write ONE warm neutral sentence in French referencing: {ref_str}. Max 20 words. Plain text only."
            else:
                prompt = f"Write ONE warm neutral sentence in English referencing: {ref_str}. Max 20 words. Plain text only."
            try:
                raw = self._call_llm(prompt)
                candidate = self._extract_final_question(raw)
                if candidate and 5 <= len(candidate.split()) <= 25:
                    # Rejeter toute phrase qui ressemble à une question
                    if "?" not in candidate and "؟" not in candidate:
                        if self.target_lang == "Arabe" and not self._detect_language_drift(candidate):
                            personal_sentence = candidate.rstrip(".") + ". "
                        elif self.target_lang != "Arabe":
                            personal_sentence = candidate.rstrip(".") + ". "
            except Exception:
                pass

        mandatory_close = {
            "Français": "Merci pour cet échange. Nous reviendrons vers vous avec un retour d'ici quelques jours.",
            "Anglais":  "Thank you for your time. We will get back to you with feedback within the next few days.",
            "Arabe":    "شكراً لوقتك. راح نرجع لك بالتغذية الراجعة خلال الأيام القادمة.",
        }[self.target_lang]

        result = (personal_sentence + mandatory_close).strip()
        # Supprimer tout ce qui précède la conclusion obligatoire si une question s'est glissée
        if mandatory_close in result:
            result = result[result.index(mandatory_close):]
        # Sécurité finale : s'assurer qu'aucune phrase-question ne subsiste
        sentences = re.split(r"(?<=[.!?؟])\s+", result)
        clean_sentences = [s for s in sentences if "?" not in s and "؟" not in s]
        if not clean_sentences:
            return mandatory_close
        return " ".join(clean_sentences).strip()

    # =========================================================================
    # Phase progression
    # =========================================================================

    def _needs_phase_progression(self, phase: str) -> bool:
        if phase == "CLOSING":
            return False

        n = self._candidate_turns_in_phase(phase)

        if self._is_in_block_b_window() and phase in BLOCK_A_PHASES:
            return True

        if phase == "OPENING":
            return n >= 1

        if phase == "JOB_ALIGNED_EXPLORATION":
            if self._is_in_block_b_window():
                return True
            required = min(2, max(1, len(self._jd_tools)))
            return n >= 2 and len(self._tools_fully_done) >= required

        if phase == "PROJECT_DEEP_DIVE":
            if self._is_in_block_b_window():
                return True
            return n >= 2 and (len(self.deep_dived_projects) >= 1 or n >= 3)

        if phase == "TECHNICAL_DEPTH":
            if self._is_in_block_b_window():
                return True
            return n >= 2 and (
                self._decision_shown_per_phase.get("TECHNICAL_DEPTH", False)
                or n >= 3
            )

        if phase == "SOFT_SKILLS_BEHAVIORAL":
            min_soft_turns = 2 if self.block_b_budget_minutes >= 6 else 1
            behavioral_ok = self._behavioral_stories_count >= 1 or n >= min_soft_turns
            return behavioral_ok

        if phase == "CANDIDATE_QUESTIONS":
            if self._candidate_declined_questions:
                self.candidate_questions_completed = True
                return True
            if self._candidate_question_rounds >= self._max_candidate_question_rounds:
                self.candidate_questions_completed = True
                return True
            return False

        if phase == "FINAL_CHECK":
            if self._candidate_declined_questions:
                self.final_check_completed = True
                return True
            return n >= 1

        return False

    def _advance_phase(self) -> None:
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.followup_depth = 0
            self.current_topic_focus = ""

    def _should_end_interview(self) -> bool:
        # Déclin candidat → fin immédiate, indépendamment du temps restant
        if self._candidate_declined_questions:
            if self.steps[self.current_step_index] != "CLOSING":
                self.current_step_index = self.steps.index("CLOSING")
            return True

        phase = self.steps[self.current_step_index]
        if self.get_time_remaining() <= 0.5:
            if phase != "CLOSING":
                self.current_step_index = self.steps.index("CLOSING")
            return True
        if phase == "CLOSING":
            return True
        if phase == "FINAL_CHECK" and self._candidate_declined_questions:
            self.current_step_index = self.steps.index("CLOSING")
            return True
        if phase == "FINAL_CHECK" and self._needs_phase_progression("FINAL_CHECK"):
            self.current_step_index = self.steps.index("CLOSING")
            return True
        return False

    # =========================================================================
    # Dedup + praise stripping
    # =========================================================================

    def _strip_praise(self, text: str) -> str:
        lower = text.lower()
        for token in FORBIDDEN_PRAISE_TOKENS:
            if token in lower:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                sentences = [s for s in sentences if token not in s.lower()]
                text = " ".join(sentences).strip()
                if not text:
                    return self._fallback_question()
                break
        return text

    def _soften_if_needed(self, text: str) -> str:
        prefix = {
            "Français": "Prenons cela simplement : ",
            "Anglais":  "Let's keep it simple: ",
            "Arabe":    "خلّنا ناخذها ببساطة: ",
        }[self.target_lang]
        if not text.lower().startswith(prefix.lower()):
            return prefix + text[0].lower() + text[1:] if text else text
        return text

    # =========================================================================
    # Signal detection
    # =========================================================================

    def _detect_and_register_claims(self, text: str) -> bool:
        if not PROBE_DEPTH_CONFIG["signal_validation_enabled"]:
            return False
        already = self._challenged_claim_types
        prompt = (
            "Identify unverifiable assertions in the candidate answer below.\n"
            "Return ONLY a JSON array [{type, excerpt}]. "
            "Types: ownership_vague, impact_claim, scale_claim, leadership_claim, credential_claim.\n"
            "Empty array [] if none. No markdown.\n\n"
            f"ANSWER:\n{text[:1500]}"
        )
        try:
            raw = self._call_llm(prompt, system_override="You are an analytical assistant. Return only valid JSON.")
            clean = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
            m = re.search(r"\[.*?\]", clean, re.DOTALL)
            if not m:
                return False
            claims = json.loads(m.group())
            if not isinstance(claims, list):
                return False
            found = False
            for item in claims:
                if not isinstance(item, dict):
                    continue
                ctype = str(item.get("type", "")).strip()
                excerpt = str(item.get("excerpt", "")).strip()[:120]
                if not ctype or not excerpt or ctype in already:
                    continue
                existing = {c["type"] for c in self._unvalidated_claims}
                if ctype in existing:
                    continue
                self._unvalidated_claims.append({"type": ctype, "excerpt": excerpt, "full_text": text[:300]})
                found = True
            return found
        except Exception:
            return False

    def _detect_behavioral_story(self, text: str) -> bool:
        lower = text.lower()
        has_situation = any(k in lower for k in [
            "when", "quand", "once", "there was", "il y avait", "faced",
            "لما", "كانت عندي", "واجهت", "في موقف"
        ])
        has_action = any(k in lower for k in [
            "i decided", "j'ai décidé", "i built", "j'ai mis", "i escalated",
            "i convinced", "j'ai convaincu", "قررت", "بنيت", "أقنعت", "تصرفت", "i proposed"
        ])
        has_result = (
            any(k in lower for k in [
                "result", "résultat", "outcome", "ended up", "at the end", "finalement",
                "النتيجة", "في النهاية", "we achieved", "نجحنا", "the outcome"
            ])
            or bool(re.search(r"\b\d+[\s]?(?:%|days|weeks|hours|jours|semaines)\b", lower))
        )
        return has_situation and has_action and has_result and len(text.split()) >= 40

    def _detect_decision_reasoning(self, text: str) -> bool:
        lower = text.lower()
        has_choice = any(k in lower for k in [
            "chose", "selected", "decided to use", "opted for", "went with",
            "j'ai choisi", "on a opté", "اخترنا", "قررنا نستخدم"
        ])
        has_alternative = any(k in lower for k in [
            "instead of", "rather than", "compared to", "alternative", " vs ",
            "plutôt que", "au lieu de", "بدل", "مقارنة بـ",
        ])
        has_rationale = any(k in lower for k in [
            "because", "since", "given that", "trade-off", "constraint", "due to",
            "parce que", "étant donné", "عشان", "بسبب", "نظراً"
        ])
        return has_choice and (has_alternative or has_rationale)

    def _detect_leadership_signals(self, text: str) -> List[str]:
        lower = text.lower()
        signals = []
        patterns = [
            (r"led?\b.{0,30}\b(\d+).{0,20}\b(engineer|developer|person|people)", "team_size"),
            (r"(mentored|coached|onboard).{0,30}\b(junior|new|intern)", "mentoring"),
            (r"(set|defined|established).{0,30}\b(direction|roadmap|standard|process)", "direction_setting"),
            (r"(hired|interviewed|recruited).{0,30}\b(engineer|developer|candidate)", "hiring"),
            (r"(performance review|1:1|one.on.one|feedback session)", "people_management"),
        ]
        for pattern, stype in patterns:
            if re.search(pattern, lower):
                signals.append(stype)
        return list(set(signals))

    # =========================================================================
    # IMPROVEMENT F: Enhanced skill extraction
    # =========================================================================

    def _normalise_skill(self, skill: str) -> str:
        lower = skill.lower().strip()
        for canonical, variants in SKILL_SYNONYMS.items():
            if lower == canonical or lower in variants:
                return canonical
            for v in variants:
                if v in lower or lower in v:
                    return canonical
        return lower

    def _match_skills_fuzzy(self, text_lower: str, keyword: str) -> bool:
        canonical = self._normalise_skill(keyword)
        variants = SKILL_SYNONYMS.get(canonical, [canonical])
        all_forms = {canonical} | set(variants)
        return any(v in text_lower for v in all_forms)

    def _extract_skills(self, text: str) -> List[str]:
        if not text:
            return []
        lower = text.lower()
        found = set()

        if self._dynamic_skill_keywords:
            for kw in self._dynamic_skill_keywords:
                if self._match_skills_fuzzy(lower, kw):
                    found.add(self._normalise_skill(kw))
        else:
            cap_tokens = re.findall(r"\b([A-Z][a-zA-Z0-9+#._/-]{1,30})\b", text)
            abbrevs = re.findall(
                r"\b(sql|api|crm|erp|kpi|seo|ml|ai|nlp|llm|rag|etl|qa|oop)\b", lower
            )
            for t in cap_tokens:
                found.add(self._normalise_skill(t))
            found.update(abbrevs)

        multi_word_pattern = re.compile(
            r"\b(apache\s+(?:spark|airflow|kafka|flink|hive|beam)|"
            r"google\s+(?:cloud|bigquery|dataflow|pubsub)|"
            r"amazon\s+(?:redshift|emr|glue|s3|ec2)|"
            r"azure\s+(?:datafactory|blob|databricks)|"
            r"great\s+expectations|dbt\s+(?:core|cloud)|scikit[\-\s]learn|"
            r"deep\s+learning|machine\s+learning|"
            r"data\s+(?:quality|engineering|science|pipeline|lake|warehouse|mesh))\b",
            re.I
        )
        for m in multi_word_pattern.finditer(lower):
            found.add(self._normalise_skill(m.group().strip()))

        soft_markers = [
            "cross-functional", "stakeholder", "okr", "roadmap", "sprint",
            "agile", "scrum", "kanban", "sla", "slo", "on-call", "incident management",
        ]
        for marker in soft_markers:
            if marker in lower:
                found.add(marker)

        return sorted(found)

    # =========================================================================
    # Answer quality classification
    # =========================================================================

    def _extract_cv_strong_signals(self, text: str) -> list:
        signals = []
        metric_pattern = re.compile(
            r'(\d+[\.,]?\d*\s*%|\d+\s*(?:terabyte|TB|GB|million|billion|k\b)[\w\s]*)',
            re.IGNORECASE
        )
        for m in metric_pattern.finditer(text):
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 60)
            quote = text[start:end].strip()
            signals.append({
                "type": "metric",
                "topic": f"the claim '{m.group()}' — ask HOW it was measured and their personal contribution",
                "quote": quote,
                "priority": 1,
            })
        lead_pattern = re.compile(
            r'(?:led|lead|owned|architected|designed|built|implemented)\s+(?:the\s+)?[\w\s\-]{5,40}',
            re.IGNORECASE
        )
        for m in lead_pattern.finditer(text):
            signals.append({
                "type": "ownership",
                "topic": f"their ownership of '{m.group().strip()}'",
                "quote": m.group().strip(),
                "priority": 2,
            })
        found_techs: List[str] = []
        if self._jd_tools:
            for jd_tool in self._jd_tools:
                if self._match_skills_fuzzy(text.lower(), jd_tool):
                    found_techs.append(jd_tool)
        else:
            cap_tokens = re.findall(r'\b([A-Z][a-zA-Z0-9+#._/-]{2,30})\b', text)
            found_techs = list(dict.fromkeys(cap_tokens))[:6]
        for tech in found_techs[:3]:
            signals.append({
                "type": "technology",
                "topic": f"their use of {tech} — a specific architectural decision",
                "quote": tech,
                "priority": 3,
            })
        signals.sort(key=lambda x: x["priority"])
        return signals

    def _classify_answer_quality(self, text: str) -> str:
        words = text.split()
        wc = len(words)
        sc = len(self._extract_skills(text))
        mc = len(re.findall(
            r"\b\d+(?:[.,]\d+)?[\s]*(?:%|ms|s\b|tb|gb|mb|fps|req|k\b|m\b)\b"
            r"|\b(?:accuracy|f1|latency|throughput|precision|recall|auc|rmse)\b",
            text.lower()
        ))
        has_decision = self._detect_decision_reasoning(text)
        has_behavioral = self._detect_behavioral_story(text)
        t = ANSWER_QUALITY_THRESHOLDS
        if wc >= t["STRONG"]["min_words"] and sc >= t["STRONG"]["min_skills"] and mc >= t["STRONG"]["min_metrics"]:
            return "STRONG"
        if wc >= t["GOOD"]["min_words"] and (sc >= t["GOOD"]["min_skills"] or has_decision or has_behavioral):
            return "GOOD"
        if wc >= t["INCOMPLETE"]["min_words"]:
            return "INCOMPLETE"
        return "VAGUE"

    # =========================================================================
    # Static analysis & dynamic state
    # =========================================================================

    def _refresh_static_analysis(self) -> None:
        cv_text    = self.doc_texts.get("cv", "")
        offer_text = self.doc_texts.get("job_offer", "")
        self.session_notes["key_projects"] = self._extract_projects(cv_text)
        offer_skills = self._extract_skills(offer_text)
        cv_skills    = self._extract_skills(cv_text)
        self.session_notes["matched_skills"]  = sorted(set(offer_skills) & set(cv_skills))
        self.session_notes["missing_skills"]  = sorted(set(offer_skills) - set(cv_skills))
        if offer_text:
            self._job_required_skills = list(offer_skills)
        self._static_analysis_pending = True

    def _ensure_llm_analysis(self) -> None:
        if not getattr(self, "_static_analysis_pending", False):
            return
        self._static_analysis_pending = False

        cv_text    = self.doc_texts.get("cv", "")
        offer_text = self.doc_texts.get("job_offer", "")

        try:
            self._refresh_dynamic_keywords(cv_text, offer_text)
        except Exception as e:
            print(f"⚠️  _refresh_dynamic_keywords: {e}")

        if offer_text:
            try:
                self._analyze_job_offer_llm(offer_text)
            except Exception as e:
                print(f"⚠️  _analyze_job_offer_llm: {e}")
            try:
                self._jd_tools = self._extract_jd_tools(offer_text)
            except Exception as e:
                print(f"⚠️  _extract_jd_tools: {e}")

        if self._jd_tools and (cv_text or offer_text):
            try:
                self._build_tool_project_map()
                self._tool_rotation_order = list(self._jd_tools[:15])
            except Exception as e:
                print(f"⚠️  _build_tool_project_map: {e}")

        for tool in self._jd_tools:
            self._coverage_matrix["TECH"].setdefault(tool, False)
        for proj in self.session_notes.get("key_projects", []):
            self._coverage_matrix["PROJECT"].setdefault(proj, False)

        print(f"✅  Analyse LLM lazy terminée — {len(self._jd_tools)} outils JD détectés")

    def _build_tool_project_map(self) -> None:
        cv_text = self.doc_texts.get("cv", "").lower()
        projects = self.session_notes.get("key_projects", [])
        self._tool_project_map = {}
        for tool in self._jd_tools:
            tool_words = [w for w in tool.lower().split() if len(w) > 2]
            canonical = self._normalise_skill(tool)
            for variant in SKILL_SYNONYMS.get(canonical, []):
                tool_words.extend([w for w in variant.split() if len(w) > 2])
            tool_words = list(set(tool_words))
            if not tool_words:
                self._tool_project_map[tool] = []
                continue
            matching_projects = []
            for proj in projects:
                proj_lower = proj.lower()
                if any(w in proj_lower for w in tool_words):
                    matching_projects.append(proj)
            if not matching_projects:
                for proj in projects:
                    proj_key = proj[:30].lower()
                    idx = cv_text.find(proj_key)
                    if idx != -1:
                        window = cv_text[idx: idx + 400]
                        if any(w in window for w in tool_words):
                            matching_projects.append(proj)
            self._tool_project_map[tool] = list(dict.fromkeys(matching_projects))[:3]

    def _extract_jd_tools(self, offer_text: str) -> List[str]:
        prompt = (
            "Extract all specific tools, technologies, frameworks, and platforms "
            "from this job description. Return ONLY a JSON array of lowercase strings, "
            "most critical first. No duplicates. No generic words. No markdown.\n\n"
            f"JOB DESCRIPTION:\n{offer_text[:3000]}"
        )
        try:
            raw = self._call_llm(prompt, system_override="You are a technical recruiter. Return only valid JSON arrays.")
            clean = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
            m = re.search(r"\[.*?\]", clean, re.DOTALL)
            if m:
                tools = json.loads(m.group())
                if isinstance(tools, list):
                    normalised = []
                    for t in tools:
                        if isinstance(t, str) and 1 < len(t.strip()) < 50:
                            normalised.append(self._normalise_skill(t.strip()))
                    return list(dict.fromkeys(normalised))
        except Exception:
            pass
        return list(self._dynamic_skill_keywords)[:15]

    def _refresh_dynamic_keywords(self, cv_text: str, offer_text: str) -> None:
        combined = ""
        if offer_text:
            combined += f"JOB OFFER:\n{offer_text[:2500]}\n\n"
        if cv_text:
            combined += f"CV:\n{cv_text[:2500]}"
        if not combined.strip():
            return
        prompt = (
            "Extract all specific skills, tools, technologies, and domain terms. "
            "Return ONLY a JSON array of lowercase strings. No generic words. No markdown.\n\n"
            f"{combined}"
        )
        try:
            raw = self._call_llm(prompt, system_override="You are a technical recruiter. Return only valid JSON arrays.")
            clean = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
            m = re.search(r"\[.*?\]", clean, re.DOTALL)
            if m:
                kws = json.loads(m.group())
                if isinstance(kws, list):
                    self._dynamic_skill_keywords = {
                        self._normalise_skill(str(k).strip())
                        for k in kws
                        if isinstance(k, str) and 2 <= len(k.strip()) <= 50
                    }
        except Exception:
            pass

    def _analyze_job_offer_llm(self, offer_text: str) -> None:
        prompt = (
            "Analyse this job description and return ONLY a JSON object.\n\n"
            "Required fields:\n"
            "  seniority: one of ['junior', 'mid', 'senior']\n"
            "  domain: short lowercase label\n"
            "  required_behaviors: list of lowercase strings\n"
            "  domain_directive: 1–3 sentences on technical sub-topics to prioritise\n"
            "  topic_keywords: dict mapping topic_name to list of 3–6 keywords\n"
            "  skill_synonyms: dict mapping canonical_tool_name to variant spellings\n\n"
            f"JOB DESCRIPTION:\n{offer_text[:3000]}"
        )
        try:
            raw = self._call_llm(prompt, system_override="You are a technical recruiter analyst. Return only valid JSON.")
            clean = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if not m:
                return
            data = json.loads(m.group())
            self._job_seniority_level = str(data.get("seniority", "mid")).strip()
            self._job_domain          = str(data.get("domain", "general")).strip()
            self._job_required_behaviors = [
                str(b).strip() for b in data.get("required_behaviors", [])
                if isinstance(b, str)
            ]
            self._job_domain_directive = str(data.get("domain_directive", "")).strip()
            syns = data.get("skill_synonyms", {})
            if isinstance(syns, dict):
                for canonical, variants in syns.items():
                    if isinstance(variants, list):
                        SKILL_SYNONYMS[str(canonical).lower().strip()] = [
                            str(v).lower().strip() for v in variants if isinstance(v, str)
                        ]
            topic_kws = data.get("topic_keywords", {})
            if isinstance(topic_kws, dict):
                self._job_topic_keywords = {
                    str(k).strip(): [str(v).strip() for v in vals if isinstance(v, str)]
                    for k, vals in topic_kws.items()
                    if isinstance(vals, list)
                }
        except Exception:
            pass

    def _update_dynamic_state(self, user_text: str, candidate_sentiment: str) -> None:
        if not self.session_notes["candidate_name"]:
            name = self._extract_candidate_name(user_text)
            if name:
                self.session_notes["candidate_name"] = name

        skill_hits = self._extract_skills(user_text)
        project_hits = self._extract_projects(user_text)
        # Compléter avec les projets mentionnés oralement
        oral_projects = self._extract_mentioned_projects(user_text)
        for op in oral_projects:
            if op not in project_hits:
                project_hits.append(op)
        # Mémoriser dans session_notes pour que les directives y accèdent
        for p in project_hits:
            if p not in self.session_notes.get("key_projects", []):
                self.session_notes.setdefault("key_projects", []).append(p)
        for s in skill_hits:
            self.covered_skills.add(s)
            self.current_focus_skill = s
            for jd_tool in self._jd_tools:
                if self._match_skills_fuzzy(s, jd_tool):
                    self.explored_jd_tools.add(jd_tool)

        for p in project_hits:
            self.covered_projects.add(p)
            self.current_focus_project = p
            phase = self.steps[self.current_step_index]
            if phase in ("PROJECT_DEEP_DIVE", "TECHNICAL_DEPTH"):
                self.deep_dived_projects.add(p)

        lower = user_text.lower()
        UNIVERSAL_TOPIC_MAP = {
            "teamwork":        ["team", "stakeholder", "collaboration", "manager", "colleague"],
            "problem_solving": ["incident", "bug", "debug", "failure", "issue", "error", "crash"],
            "architecture":    ["architecture", "design", "layer", "structure", "pattern"],
            "delivery":        ["deploy", "release", "launch", "ship", "production", "rollout"],
            "cost_ownership":  ["cost", "saving", "budget", "billing", "efficiency", "reduce"],
            "observability":   ["monitor", "alert", "log", "metric", "trace", "dashboard"],
            "quality":         ["quality", "test", "review", "validation", "standard"],
        }

        effective_topic_map = {**UNIVERSAL_TOPIC_MAP, **self._job_topic_keywords}
        for topic, kws in effective_topic_map.items():
            if any(k in lower for k in kws):
                self.covered_topics.add(topic)

        if self._detect_decision_reasoning(user_text):
            self.covered_topics.add("decision_reasoning")
        if self._detect_behavioral_story(user_text):
            self.covered_topics.add("behavioral_story")

        beh_map = {
            "conflict":       ["conflict", "disagreement", "dispute"],
            "prioritization": ["prioriti", "deadline", "competing"],
            "failure":        ["failure", "failed", "mistake", "learned"],
            "leadership":     ["led", "managed", "lead a team", "mentored"],
            "pressure":       ["pressure", "stress", "crisis", "urgent"],
        }
        for dim, kws in beh_map.items():
            if any(k in lower for k in kws):
                self._behavioral_coverage[dim] = True
                self._coverage_matrix["BEHAVIORAL"][dim] = True

        detected_topic = next(
            (t for t, kws in effective_topic_map.items() if any(k in lower for k in kws)), ""
        )
        if detected_topic:
            if detected_topic == self.current_topic_focus:
                self.followup_depth += 1
                self._topic_followup_count[detected_topic] = (
                    self._topic_followup_count.get(detected_topic, 0) + 1
                )
            else:
                self.current_topic_focus = detected_topic
                self.followup_depth = 0
                self._topic_followup_count.setdefault(detected_topic, 0)

    def _mark_metrics_probed(self) -> None:
        self._metrics_probed = True
        self._turns_since_metrics_probe = 0
        self.covered_topics.add("metrics")

    def _mark_jd_tool_why_asked(self, question_text: str) -> None:
        lower = question_text.lower()
        is_why = any(k in lower for k in [
            "why", "pourquoi", "ليش", "instead of", "alternative", "over", "trade-off",
            "plutôt que", "rather than", "compared to", "بدلاً", "عوض"
        ])
        if not is_why:
            return
        for jd_tool in self.explored_jd_tools:
            tool_words = set(jd_tool.lower().split())
            if any(w in lower for w in tool_words if len(w) > 3):
                self._jd_tool_why_asked.add(jd_tool)

    def _update_scores(self, user_text: str) -> None:
        phase = self.steps[self.current_step_index]
        if phase in ("FINAL_CHECK", "CLOSING"):
            return
        length_score = min(10, max(2, len(user_text.split()) // 8))
        specificity = 0
        if re.search(r"\b\d+(?:[.,]\d+)?\b", user_text):
            specificity += 2
        if any(k in user_text.lower() for k in ["because", "parce que", "trade-off", "architecture"]):
            specificity += 2
        if len(self._extract_skills(user_text)) >= 2:
            specificity += 2
        if self._detect_behavioral_story(user_text):
            specificity += 2
        if self._detect_decision_reasoning(user_text):
            specificity += 2
        value = min(10, length_score + specificity)
        if phase in self.scores:
            self.scores[phase] = min(10, max(self.scores.get(phase, 0), value))
        if phase in ("JOB_ALIGNED_EXPLORATION", "PROJECT_DEEP_DIVE", "TECHNICAL_DEPTH"):
            self._hs_score += value
            self._hs_count += 1
        elif phase in ("SOFT_SKILLS_BEHAVIORAL", "CANDIDATE_QUESTIONS"):
            self._ss_score += value
            self._ss_count += 1

    # =========================================================================
    # RAG retrieval — V3: Hybrid + Cross-Encoder
    # =========================================================================

    def _retrieve_context(self, query: str, phase: str) -> str:
        """
        RAG-IMPROVEMENT-1+2: Uses hybrid BM25+vector search with cross-encoder re-ranking.
        Source boosts are passed to both BM25 and dense search during fusion.
        """
        self.ensure_embeddings_ready()
        source_boost = {
            "cv":           1.15 if phase in {"OPENING", "PROJECT_DEEP_DIVE"} else 1.0,
            "job_offer":    1.20 if phase in {"JOB_ALIGNED_EXPLORATION", "TECHNICAL_DEPTH"} else 1.0,
            "company_info": 1.10 if phase == "CANDIDATE_QUESTIONS" else 1.0,
        }
        q = self._craft_retrieval_query(query, phase)
        # SimpleVectorStore.search() now does hybrid BM25+dense + cross-encoder re-rank
        chunks = self.vector_store.search(q, top_k=RETRIEVAL_TOP_K, source_boost=source_boost)
        reranked = self._rerank_chunks(chunks, phase, query)[:MAX_RETRIEVED_CHUNKS]
        cv_blocks = []
        jd_blocks = []
        other_blocks = []

        for i, ch in enumerate(reranked, 1):
            label_header = {
                "cv":           "── CANDIDATE CV (what the candidate has done) ──",
                "job_offer":    "── JOB DESCRIPTION (what the role requires — NOT candidate info) ──",
                "company_info": "── COMPANY INFO ──",
            }.get(ch.source_type, "── OTHER ──")

            block = f"[{i}] {label_header}\n{ch.text[:900]}"

            if ch.source_type == "cv":
                cv_blocks.append(block)
            elif ch.source_type == "job_offer":
                jd_blocks.append(block)
            else:
                other_blocks.append(block)

        sections = []
        if cv_blocks:
            sections.append("=== CV CONTEXT (candidate's actual experience) ===\n" + "\n\n".join(cv_blocks))
        if jd_blocks:
            sections.append("=== JOB DESCRIPTION CONTEXT (role requirements — do NOT attribute to candidate) ===\n" + "\n\n".join(jd_blocks))
        if other_blocks:
            sections.append("=== OTHER CONTEXT ===\n" + "\n\n".join(other_blocks))

        return "\n\n".join(sections) if sections else "No context available."

    def _craft_retrieval_query(self, user_text: str, phase: str) -> str:
        tokens = []
        if self.current_focus_project:
            tokens.append(self.current_focus_project)
        if self.current_focus_skill:
            tokens.append(self.current_focus_skill)
        if self._current_tool_focus:
            tokens.append(self._current_tool_focus)
        phase_queries = {
            "OPENING":                 "candidate background education career path",
            "JOB_ALIGNED_EXPLORATION": "job requirements tools technologies skills why decision",
            "PROJECT_DEEP_DIVE":       "project architecture decisions trade-offs impact metrics",
            "TECHNICAL_DEPTH":         "technical depth metrics scale production decisions failures",
            "SOFT_SKILLS_BEHAVIORAL":  "leadership teamwork conflict communication STAR behavioral",
            "CANDIDATE_QUESTIONS":     "company role team process roadmap culture",
            "FINAL_CHECK":             "wrap up",
            "CLOSING":                 "closing",
        }
        tokens.append(phase_queries.get(phase, ""))
        tokens.append(user_text)
        return " | ".join(t for t in tokens if t)

    def _rerank_chunks(self, chunks: List[ChunkRecord], phase: str, user_text: str) -> List[ChunkRecord]:
        """
        Reranking secondaire : boost par source-type ET par section JD/CV.
        """
        words = set(re.findall(r"[a-zA-Z0-9_+.#-]+", user_text.lower()))

        # Sections JD prioritaires selon la phase
        PHASE_JD_SECTION_BOOST: Dict[str, List[str]] = {
            "JOB_ALIGNED_EXPLORATION": ["requirements", "required skills", "tech stack", "technologies"],
            "TECHNICAL_DEPTH":         ["requirements", "required skills", "tech stack", "experience required"],
            "PROJECT_DEEP_DIVE":       ["responsibilities", "missions", "role"],
            "SOFT_SKILLS_BEHAVIORAL":  ["requirements", "nice to have", "preferred", "responsibilities"],
            "CANDIDATE_QUESTIONS":     ["about", "benefits", "what we offer", "company"],
        }

        # Sections CV prioritaires selon la phase
        PHASE_CV_SECTION_BOOST: Dict[str, List[str]] = {
            "OPENING":            ["header", "formations", "education"],
            "PROJECT_DEEP_DIVE":  ["expérience", "experience", "projets", "projects"],
            "TECHNICAL_DEPTH":    ["expérience", "experience", "compétences", "skills"],
            "SOFT_SKILLS_BEHAVIORAL": ["expérience", "experience"],
        }

        priority_jd_sections = PHASE_JD_SECTION_BOOST.get(phase, [])
        priority_cv_sections  = PHASE_CV_SECTION_BOOST.get(phase, [])

        def score(ch: ChunkRecord) -> float:
            bonus = 0.0
            txt = ch.text.lower()
            section_lower = (ch.section or "").lower()

            # Boost source-type (comportement original)
            if phase in ("PROJECT_DEEP_DIVE", "OPENING") and ch.source_type == "cv":
                bonus += 0.20
            if phase in ("JOB_ALIGNED_EXPLORATION", "TECHNICAL_DEPTH") and ch.source_type == "job_offer":
                bonus += 0.25
            if phase == "CANDIDATE_QUESTIONS" and ch.source_type == "company_info":
                bonus += 0.30

            # Boost section JD
            if ch.source_type == "job_offer" and priority_jd_sections:
                if any(s in section_lower for s in priority_jd_sections):
                    bonus += 0.20

            # Boost section CV
            if ch.source_type == "cv" and priority_cv_sections:
                if any(s in section_lower for s in priority_cv_sections):
                    bonus += 0.15

            # Boost entité/projet mentionné dans la question
            if ch.entity:
                entity_words = set(ch.entity.lower().split())
                if entity_words & words:
                    bonus += 0.25

            # Boost mots-clés partagés
            bonus += sum(0.02 for w in words if len(w) > 2 and w in txt)

            return bonus

        return sorted(chunks, key=score, reverse=True)
    # =========================================================================
    # Question extraction (FIX-AR-6)
    # =========================================================================

    _PREAMBLE_PATTERNS = re.compile(
        r"^("
        r"here'?s?\s+(my\s+)?((first|next|following|last)\s+)?question\s*[:—\-]*\s*|"
        r"let'?s?\s+(get\s+started|begin|start)\s*(with\s+(the\s+)?interview\s*)?[.,:—\-]*\s*|"
        r"on\s+that\s+note\s*[,.:—\-]*\s*|"
        r"with\s+that\s+(in\s+mind\s*)?[,.:—\-]*\s*|"
        r"so\s*[,.:—\-]+\s*|"
        r"alright\s*[,.:—\-]*\s*|"
        r"great\s*[,.:—\-]*\s*|"
        r"perfect\s*[,.:—\-]*\s*|"
        r"moving\s+(on|forward)\s*[,.:—\-]*\s*|"
        r"(my\s+)?(next|first)\s+question\s+(is\s+|for\s+you\s+)?[,:—\-]*\s*|"
        r"(voici\s+)?(ma\s+)?(première\s+)?question\s*[:—\-]*\s*|"
        r"passons\s+(à\s+)?(la\s+suite|maintenant)\s*[,.:—\-]*\s*|"
        r"donc\s*[,.:—\-]+\s*|"
        r"d'accord\s*[,.:—\-]+\s*|"
        r"bien\s+sûr\s*[,.:—\-]*\s*|"
        r"bien\s*[,.:—\-]+\s*|"
        r"parfait\s*[,.:—\-]*\s*|"
        r"très\s+bien\s*[,.:—\-]*\s*|"
        r"السؤال\s+(الأول\s+|التالي\s+)?[,:—\-:]*\s*|"
        r"إليك\s+سؤالي?\s*[,:—\-:]*\s*|"
        r"حسناً\s*[,،.:—\-]*\s*|"
        r"بالتأكيد\s*[,،.:—\-]*\s*|"
        r"طبعاً\s*[,،.:—\-]*\s*|"
        r"شكراً\s+(?:على\s+)?(?:هذا|ذلك|إجابتك)?\s*[,،.:—\-]*\s*|"
        r"ممتاز\s*[,،.:—\-]*\s*|"
        r"رائع\s*[,،.:—\-]*\s*|"
        r"جيد\s*[,،.:—\-]*\s*|"
        r"فهمت\s*[,،.:—\-]*\s*|"
        r"واضح\s*[,،.:—\-]*\s*|"
        r"إذن\s*[,،.:—\-]*\s*|"
        r"دعني\s+أسألك\s*[,:—\-:]*\s*|"
        r"سؤالي\s+(التالي|الأول|هو)\s*[,:—\-:]*\s*|"
        r"لننتقل\s+إلى\s*[,:—\-:]*\s*|"
        r"انتقل\s+إلى\s*[,:—\-:]*\s*"
        r")",
        re.I
    )

    def _extract_final_question(self, raw: str) -> str:
        text = raw.strip()
        text = re.sub(r"^```[a-zA-Z]*", "", text).replace("```", "").strip()
        text = re.sub(
            r"^(Question|Interviewer|Recruiter|Recruteur|Avatar|Hiring Manager|المحاور|المقابلة)\s*:\s*",
            "", text, flags=re.I
        )
        text = re.sub(r"\s+", " ", text).strip()

        prev = None
        while prev != text:
            prev = text
            text = self._PREAMBLE_PATTERNS.sub("", text).strip()
            if text:
                text = text[0].upper() + text[1:]

        if not text:
            return self._fallback_question()

        sentences = re.split(r"(?<=[.?!؟])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return self._fallback_question()

        question_sentences = [
            s for s in sentences
            if s.endswith(("?", "؟")) or re.search(
                r"\b(what|how|why|when|where|who|which|could|can|did|do|does|"
                r"est-ce|quel|quelle|comment|pourquoi|quand|où|qui|"
                r"هل|كيف|ماذا|ما|لماذا|متى|أين|من|وش|إيش|ليش)\b", s, re.I
            )
        ]
        if question_sentences:
            last_q = question_sentences[-1]
            idx = sentences.index(last_q)
            if idx > 0:
                trans = sentences[idx - 1]
                trans_lower = trans.lower()
                is_preamble = self._PREAMBLE_PATTERNS.match(trans_lower) is not None
                is_filler = any(tok in trans_lower for tok in [
                    "here's", "let's", "alright", "great", "perfect",
                    "my question", "first question", "next question",
                    "voici", "passons", "السؤال", "حسناً", "إذن",
                ])
                if not is_preamble and not is_filler and len(trans.split()) <= 10:
                    return f"{trans} {last_q}"
            return last_q

        result = " ".join(sentences[:2]).strip()
        return result if len(result) >= 8 else self._fallback_question()

    # =========================================================================
    # Logging & streaming
    # =========================================================================

    def _append_turn(
        self, phase: str, speaker: str, text: str,
        emotion: str = "neutre", answer_quality: str = "N/A",
        signal_validated: bool = False,
        behavioral_story_detected: bool = False,
        decision_reasoning_detected: bool = False,
        weak_signals_detected: Optional[List[str]] = None,
    ) -> None:
        turn = TurnRecord(
            phase=phase, speaker=speaker, text=text.strip(),
            timestamp=_now_str(), emotion=emotion, answer_quality=answer_quality,
            signal_validated=signal_validated,
            behavioral_story_detected=behavioral_story_detected,
            decision_reasoning_detected=decision_reasoning_detected,
            weak_signals_detected=weak_signals_detected or [],
        )
        self.turns.append(turn)
        role = "candidate" if self._is_candidate_speaker(speaker) else "assistant"
        self.conversation_history.append({"role": role, "content": turn.text})
        self.conversation_history = self.conversation_history[-(MAX_HISTORY_TURNS * 2):]
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(
                f"[{turn.timestamp}] [{phase}] {speaker} "
                f"[quality={answer_quality}] [behavioral={behavioral_story_detected}] "
                f"[decision={decision_reasoning_detected}] "
                f"[weak_signals={weak_signals_detected}]: {turn.text}\n"
            )

    def _write_log_header(self) -> None:
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"HR INTERVIEW — {self.target_lang}\n")
            f.write(f"Date    : {_now_str()}\n")
            f.write(f"Duration: {self.duration_minutes} minutes\n")
            f"RAG     : Hybrid BM25+Vector (alpha={HYBRID_ALPHA}) + CrossEncoder={USE_CROSS_ENCODER} "
            f"+ StructuredChunking=CV_sections+JD_sections + SemanticFallback={USE_SEMANTIC_CHUNKING}\n\n"
    def _yield_text_stream(self, text: str, candidate_sentiment: str, interview_ended: bool):
        tokens = re.findall(r"\S+\s*", text) or [text]
        sentence_buffer = []
        sentence_idx = 0
        full_text = ""
        for tok in tokens:
            full_text += tok
            sentence_buffer.append(tok)
            yield {"type": "token", "token": tok}
            if tok.strip().endswith((".", "?", "!", "؟")):
                sentence = "".join(sentence_buffer).strip()
                if sentence:
                    yield {"type": "sentence", "text": sentence, "index": sentence_idx}
                    sentence_idx += 1
                sentence_buffer = []
        if sentence_buffer:
            sentence = "".join(sentence_buffer).strip()
            if sentence:
                yield {"type": "sentence", "text": sentence, "index": sentence_idx}
        yield {"type": "stream_done", "full_text": full_text.strip()}
        yield {
            "type": "meta",
            "candidate_sentiment": candidate_sentiment,
            "phase": self.steps[self.current_step_index],
            "time_left": self.get_time_remaining(),
            "interview_ended": interview_ended,
            "report": self._build_inline_report() if interview_ended else None,
        }

    # =========================================================================
    # State summary
    # =========================================================================

    def _format_state_summary(self, phase: str, candidate_sentiment: str) -> str:
        history = self.turns[-MAX_HISTORY_TURNS:]
        hist_text = "\n".join(
            f"- [{t.phase}] {t.speaker} [quality={t.answer_quality}]: {t.text[:200]}"
            for t in history
        ) or "No history yet."
        time_left = self.get_time_remaining()
        time_pct = int((time_left / max(1, self.duration_minutes)) * 100)
        return (
            f"Phase              : {phase}\n"
            f"Time left          : {time_left} min ({time_pct}%)\n"
            f"Sentiment          : {candidate_sentiment}\n"
            f"JD tools           : {self._jd_tools[:8]}\n"
            f"Tools explored     : {sorted(self.explored_jd_tools)}\n"
            f"Tool focus         : {self._current_tool_focus or 'none'} (followups: {self._tool_followup_count})\n"
            f"Tools fully done   : {sorted(self._tools_fully_done)}\n"
            f"Projects dived     : {sorted(self.deep_dived_projects)}\n"
            f"Covered skills     : {sorted(self.covered_skills)}\n"
            f"Covered topics     : {sorted(self.covered_topics)}\n"
            f"Behavioral         : {[k for k, v in self._behavioral_coverage.items() if v]}\n"
            f"Pending claims     : {len(self._unvalidated_claims)}\n"
            f"Answer history     : {self._answer_quality_history[-6:]}\n"
            f"RAG mode           : Hybrid BM25+Dense (α={HYBRID_ALPHA}) + CrossEncoder={USE_CROSS_ENCODER} + SemanticChunks={USE_SEMANTIC_CHUNKING}\n\n"
            f"Conversation:\n{hist_text}"
        )

    # =========================================================================
    # Report
    # =========================================================================

    def _build_inline_report(self) -> Dict[str, Any]:
        total = 0
        phases_out = []
        for phase in [p for p in KNOWN_PHASES if p not in ("FINAL_CHECK", "CLOSING")]:
            raw = self.scores.get(phase, 0)
            max_pts = PHASE_WEIGHTS.get(phase, 0)
            got = round((raw / 10) * max_pts)
            total += got
            phases_out.append({"phase": phase, "label": PHASE_LABELS[phase], "score": got, "max": max_pts})

        strengths, weaknesses = [], []
        if self.session_notes.get("matched_skills"):
            strengths.append("Skills aligned with JD: " + ", ".join(self.session_notes["matched_skills"][:6]))
        if self.covered_projects:
            strengths.append("Projects discussed: " + ", ".join(list(self.covered_projects)[:4]))
        if self._behavioral_stories_count > 0:
            strengths.append(f"Concrete STAR stories: {self._behavioral_stories_count}")
        if self._leadership_score >= 5:
            strengths.append(f"Leadership signals: {', '.join(self._leadership_signals[:3])}")
        if "decision_reasoning" in self.covered_topics:
            strengths.append("Decision-making reasoning demonstrated")
        if self._metrics_probed:
            strengths.append("Metrics probed and verified")

        if self.session_notes.get("missing_skills"):
            weaknesses.append("Skills not demonstrated: " + ", ".join(self.session_notes["missing_skills"][:6]))
        if not self._metrics_probed:
            weaknesses.append("No quantified metrics probed or verified")
        if self._behavioral_stories_count == 0:
            weaknesses.append("No concrete STAR story provided")

        pct = round((total / 100) * 100)
        recommendation = (
            "Strong profile — recommend further evaluation." if pct >= 70 else
            "Interesting profile with gaps to explore." if pct >= 50 else
            "Significant reservations about fit or depth."
        )
        hs_avg = round(self._hs_score / self._hs_count, 1) if self._hs_count else 0
        ss_avg = round(self._ss_score / self._ss_count, 1) if self._ss_count else 0
        quality_dist: Dict[str, int] = {"STRONG": 0, "GOOD": 0, "INCOMPLETE": 0, "VAGUE": 0}
        for q in self._answer_quality_history:
            if q in quality_dist:
                quality_dist[q] += 1

        # ── ÉVALUATION DÉTAILLÉE TECHNIQUE PAR POINT ─────────────────────────
        # Chaque outil JD est évalué : répondu / partiel / manquant
        evaluation_technique_points = []
        all_candidate_text = " ".join(
            t.text.lower() for t in self.turns
            if self._is_candidate_speaker(t.speaker)
        )
        for tool in self._jd_tools[:20]:
            tool_lower = tool.lower()
            tool_words = [w for w in tool_lower.split() if len(w) > 2]
            # Vérifier si l'outil a été exploré
            is_explored = tool in self.explored_jd_tools or tool in self._tools_fully_done
            # Vérifier si le candidat en a parlé dans ses réponses
            mentioned_in_answers = any(w in all_candidate_text for w in tool_words) if tool_words else False
            # Vérifier si fully done (deep dive avec métriques)
            is_fully_done = tool in self._tools_fully_done

            if is_fully_done or (is_explored and mentioned_in_answers):
                status = "answered"  # Répondu complètement
            elif is_explored or mentioned_in_answers:
                status = "partial"   # Partiellement abordé
            else:
                status = "missing"   # Non abordé

            evaluation_technique_points.append({
                "point": tool,
                "status": status,
                "explored_by_interviewer": is_explored,
                "mentioned_by_candidate": mentioned_in_answers,
            })

        # ── ÉVALUATION COMMUNICATION PAR POINT ────────────────────────────────
        evaluation_communication_points = []

        # Point 1 : Structure des réponses (STAR)
        star_status = "answered" if self._behavioral_stories_count >= 2 else (
            "partial" if self._behavioral_stories_count == 1 else "missing"
        )
        evaluation_communication_points.append({
            "point": "Réponses structurées (méthode STAR / exemples concrets)",
            "status": star_status,
            "detail": f"{self._behavioral_stories_count} histoire(s) STAR détectée(s)",
        })

        # Point 2 : Métriques et chiffres
        metrics_status = "answered" if self._metrics_probed else "missing"
        metrics_count = sum(
            1 for t in self.turns
            if self._is_candidate_speaker(t.speaker)
            and re.search(r"\d+\s*(%|percent|ms|s\b|min|ko|mo|go|tb|k\b|m\b)", t.text, re.IGNORECASE)
        )
        evaluation_communication_points.append({
            "point": "Utilisation de métriques et chiffres précis",
            "status": "answered" if metrics_count >= 3 else ("partial" if metrics_count >= 1 else "missing"),
            "detail": f"{metrics_count} réponse(s) avec chiffres détectée(s)",
        })

        # Point 3 : Ownership et prise de décision personnelle
        ownership_turns = sum(
            1 for t in self.turns
            if self._is_candidate_speaker(t.speaker) and t.decision_reasoning_detected
        )
        evaluation_communication_points.append({
            "point": "Ownership et prise de décision personnelle",
            "status": "answered" if ownership_turns >= 2 else ("partial" if ownership_turns == 1 else "missing"),
            "detail": f"{ownership_turns} tour(s) avec raisonnement décisionnel",
        })

        # Point 4 : Clarté et précision des réponses
        quality_counts = quality_dist
        total_answers = sum(quality_counts.values())
        strong_good = quality_counts.get("STRONG", 0) + quality_counts.get("GOOD", 0)
        clarity_ratio = (strong_good / total_answers) if total_answers > 0 else 0
        evaluation_communication_points.append({
            "point": "Clarté et précision des réponses",
            "status": "answered" if clarity_ratio >= 0.6 else ("partial" if clarity_ratio >= 0.3 else "missing"),
            "detail": f"{strong_good}/{total_answers} réponses de bonne qualité ({round(clarity_ratio*100)}%)",
        })

        # Point 5 : Leadership et influence
        leadership_status = "answered" if self._leadership_score >= 5 else (
            "partial" if self._leadership_score >= 2 else "missing"
        )
        evaluation_communication_points.append({
            "point": "Signaux de leadership et d'influence",
            "status": leadership_status,
            "detail": f"Score leadership : {self._leadership_score}",
        })

        # Point 6 : Questions posées par le candidat
        candidate_questions = [
            t for t in self.turns
            if self._is_candidate_speaker(t.speaker) and self._candidate_is_asking_questions(t.text)
        ]
        evaluation_communication_points.append({
            "point": "Curiosité et questions posées sur le poste/l'entreprise",
            "status": "answered" if len(candidate_questions) >= 2 else (
                "partial" if len(candidate_questions) == 1 else "missing"
            ),
            "detail": f"{len(candidate_questions)} question(s) posée(s)",
        })

        # ── POINTS DÉTECTÉS ET POINTS MANQUANTS ──────────────────────────────
        # Points détectés = couverts par le candidat (answered ou partial)
        # Points manquants = non abordés du tout
        points_detectes = []
        points_manquants = []

        for pt in evaluation_technique_points:
            if pt["status"] in ("answered", "partial"):
                points_detectes.append(f"[Tech] {pt['point']}" + (" ✓" if pt["status"] == "answered" else " (~)"))
            else:
                points_manquants.append(f"[Tech] {pt['point']}")

        for pt in evaluation_communication_points:
            if pt["status"] in ("answered", "partial"):
                points_detectes.append(f"[Comm] {pt['point']}" + (" ✓" if pt["status"] == "answered" else " (~)"))
            else:
                points_manquants.append(f"[Comm] {pt['point']}")

        # ── SCORE COMMUNICATION (0–100) ───────────────────────────────────────
        comm_points_answered = sum(1 for p in evaluation_communication_points if p["status"] == "answered")
        comm_points_partial  = sum(1 for p in evaluation_communication_points if p["status"] == "partial")
        comm_total = len(evaluation_communication_points)
        score_communication_calc = round(
            ((comm_points_answered + comm_points_partial * 0.5) / max(1, comm_total)) * 100
        )

        # ── SCORE TECHNIQUE (0–100) basé sur les outils JD ───────────────────
        tech_answered = sum(1 for p in evaluation_technique_points if p["status"] == "answered")
        tech_partial  = sum(1 for p in evaluation_technique_points if p["status"] == "partial")
        tech_total    = max(1, len(evaluation_technique_points))
        score_technique_calc = round(
            ((tech_answered + tech_partial * 0.5) / tech_total) * 100
        )

        return {
            "date": _now_str(),
            "language": self.target_lang,
            "duration_minutes": self.duration_minutes,
            "phases": phases_out,
            "score_total": total,
            "score_max": 100,
            "percentage": pct,
            "hard_skills_avg": hs_avg,
            "soft_skills_avg": ss_avg,
            "answer_quality_distribution": quality_dist,
            "rag_config": {
                "hybrid_alpha": HYBRID_ALPHA,
                "rrf_k": RRF_K,
                "use_bm25": USE_BM25,
                "bm25_available": _bm25_available,
                "use_cross_encoder": USE_CROSS_ENCODER,
                "reranker_model": RERANKER_MODEL,
                "use_semantic_chunking": USE_SEMANTIC_CHUNKING,
                "semantic_breakpoint_threshold": SEMANTIC_BREAKPOINT_THRESHOLD,
                "semantic_max_chunk_tokens": SEMANTIC_MAX_CHUNK_TOKENS,
            },
            "jd_tools_explored": sorted(self.explored_jd_tools),
            "jd_tools_fully_done": sorted(self._tools_fully_done),
            "projects_deep_dived": sorted(self.deep_dived_projects),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendation": recommendation,
            "sources_rag": self.ingested_docs,
            # ── Nouvelles sections évaluation détaillée ──────────────────────
            "evaluation_technique_points": evaluation_technique_points,
            "evaluation_communication_points": evaluation_communication_points,
            "score_technique_calc": score_technique_calc,
            "score_communication_calc": score_communication_calc,
            "points_detectes": points_detectes,
            "points_manquants": points_manquants,
            "behavioral_stories_count": self._behavioral_stories_count,
            "metrics_probed": self._metrics_probed,
            "leadership_score": self._leadership_score,
        }

    # =========================================================================
    # Text helpers
    # =========================================================================

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _clean_candidate_answer(self, text: str) -> str:
        return self._clean_text(text)[:4000]

    def _extract_projects(self, text: str) -> List[str]:
        if not text:
            return []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        projects = []
        for ln in lines:
            if any(tag in ln.lower() for tag in ["project", "projet", "experience", "mission", "client", "مشروع", "تجربة"]):
                cleaned = re.sub(r"^[\-•*]\s*", "", ln)[:120]
                if len(cleaned.split()) >= 3:
                    projects.append(cleaned)
        return list(dict.fromkeys(projects))[:12]
    def _extract_mentioned_projects(self, text: str) -> List[str]:
        """
        Extrait les projets mentionnés naturellement dans une réponse orale ou textuelle.
        Complémentaire à _extract_projects() qui cible les CVs structurés.
        """
        projects = []

        # Pattern : "I built/developed/created/worked on a [description]"
        built_pattern = re.compile(
            r"(?:i built|i developed|i created|i designed|i implemented|i worked on|"
            r"j'ai construit|j'ai développé|j'ai créé|j'ai travaillé sur|"
            r"بنيت|طورت|أنشأت|عملت على)\s+(?:a\s+|an\s+|the\s+)?([^,.!?؟\n]{8,60})",
            re.IGNORECASE
        )
        for m in built_pattern.finditer(text):
            proj = m.group(1).strip().rstrip(".,;")
            if len(proj.split()) >= 2:
                projects.append(proj[:80])

        # Pattern : noms de projets avec technologies connues
        tech_project_pattern = re.compile(
            r"(?:system|platform|pipeline|model|application|tool|bot|assistant|engine)\s+"
            r"(?:using|with|based on|powered by|utilisant|avec|باستخدام)\s+"
            r"([A-Za-z0-9,\s+#.-]{3,40})",
            re.IGNORECASE
        )
        for m in tech_project_pattern.finditer(text):
            tech_ref = m.group(0).strip()[:80]
            projects.append(tech_ref)

        # Pattern : "a [adjective] [noun] system/project" avec capitalisation
        named_pattern = re.compile(
            r"\b(?:a|an|the)\s+([a-z]+(?:\s+[a-z]+){0,3}\s+"
            r"(?:system|project|platform|pipeline|model|application|assistant|bot|tool|engine))\b",
            re.IGNORECASE
        )
        for m in named_pattern.finditer(text):
            proj = m.group(1).strip()
            if len(proj.split()) >= 2:
                projects.append(proj[:60])

        # Déduplique en gardant l'ordre
        seen = set()
        unique = []
        for p in projects:
            key = p.lower()[:40]
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique[:6]

    def _extract_candidate_name(self, text: str) -> str:
        for p in [
            r"\bje suis\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"\bi am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"\bانا\s+([\u0621-\u064A]+(?:\s+[\u0621-\u064A]+){0,2})",
            r"\bأنا\s+([\u0621-\u064A]+(?:\s+[\u0621-\u064A]+){0,2})",
            r"\bاسمي\s+([\u0621-\u064A]+(?:\s+[\u0621-\u064A]+){0,2})",
        ]:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return ""

    def _estimate_sentiment(self, text: str) -> str:
        lower = text.lower()
        if len(text.split()) < 6 or any(k in lower for k in ["not sure", "maybe", "peut-être", "مش متأكد", "مو متأكد"]):
            return "hesitant"
        if any(k in lower for k in ["stress", "difficult", "hard", "صعب", "متوتر", "failed", "failure", "فشل"]):
            return "stressed"
        if any(k in lower for k in ["excited", "motivated", "interested", "enthousiaste", "متحمس", "مهتم"]):
            return "positive"
        return "neutral"

    def _candidate_is_asking_questions(self, text: str) -> bool:
        qmarks = text.count("?") + text.count("؟")
        cues = [
            "can you tell me", "could you explain", "j'ai une question",
            "pouvez-vous", "هل ممكن", "ودي أعرف", "عندي سؤال", "أريد أن أعرف"
        ]
        return qmarks >= 2 or any(c in text.lower() for c in cues)

    def _normalize_question(self, text: str) -> str:
        return _normalize_text(text)

    def _last_candidate_message(self) -> str:
        for t in reversed(self.turns):
            if self._is_candidate_speaker(t.speaker):
                return t.text
        return ""

    def _fallback_question(self, phase: Optional[str] = None, avoid_repeat: bool = False) -> str:
        import random

        phase = phase or self.steps[self.current_step_index]
        idx = self._fallback_counter.get(phase, 0)
        self._fallback_counter[phase] = idx + 1
        lang = getattr(self, "target_lang", "Anglais") or "Anglais"

        banks = {
            "OPENING": {
                "Anglais":  ["Walk me through the experience in your background most relevant to this role.",
                             "Which project from your past work best shows your technical depth?"],
                "Français": ["Décrivez l'expérience de votre parcours la plus pertinente pour ce poste.",
                             "Quel projet illustre le mieux votre niveau technique ?"],
                "Arabe":    ["وش الخبرة في مسارك اللي تشوفها الأقرب لهذا الدور وليش؟",
                             "وش المشروع اللي يعكس أكثر عمقك التقني؟"],
            },
            "JOB_ALIGNED_EXPLORATION": {
                "Anglais":  ["Give me a concrete example of how you used one of the core technologies for this role.",
                             "Which of the role's key tools do you know best, and what have you built with it?"],
                "Français": ["Donnez-moi un exemple concret d'utilisation d'une technologie clé de ce poste.",
                             "Quelle technologie du poste maîtrisez-vous le mieux ?"],
                "Arabe":    ["أعطني مثالاً عملياً على استخدامك لإحدى التقنيات الأساسية في هذا الدور.",
                             "وش التقنية اللي تعرفها أكثر من بين تقنيات الدور، وإيش بنيت فيها؟"],
            },
            "PROJECT_DEEP_DIVE": {
                "Anglais":  ["What architectural decision did you own personally on that project?",
                             "What was the hardest technical challenge you solved on that project?"],
                "Français": ["Quelle décision d'architecture avez-vous portée vous-même sur ce projet ?",
                             "Quel a été le défi technique le plus difficile de ce projet ?"],
                "Arabe":    ["وش قرار architecture امتلكته أنت شخصياً في هذا المشروع؟",
                             "وش أصعب تحدٍ تقني واجهته في هذا المشروع؟"],
            },
            "TECHNICAL_DEPTH": {
                "Anglais":  ["What concrete metric can you defend with numbers from that work?",
                             "What broke in production and what did you do about it?"],
                "Français": ["Quel résultat chiffré pouvez-vous défendre sur ce travail ?",
                             "Qu'est-ce qui a cassé en production et comment l'avez-vous résolu ?"],
                "Arabe":    ["وش النتيجة الرقمية اللي تقدر تدافع عنها من هذا العمل؟",
                             "وش اللي تعطّل في production وإيش سويت؟"],
            },
            "SOFT_SKILLS_BEHAVIORAL": {
                "Anglais":  ["Give me a specific example of a technical disagreement with a colleague.",
                             "Tell me about a time you delivered under serious time pressure."],
                "Français": ["Donnez-moi un exemple précis d'un désaccord technique avec un collègue.",
                             "Parlez-moi d'une livraison sous forte pression."],
                "Arabe":    ["أعطني مثالاً محدداً على خلاف تقني مع زميل.",
                             "حدثني عن موقف سلّمت فيه تحت ضغط شديد."],
            },
        }

        default_bank = {
            "Anglais":  "Could you give me a concrete example related to this?",
            "Français": "Pouvez-vous me donner un exemple concret ?",
            "Arabe":    "هل ممكن تعطيني مثالاً عملياً على ذلك؟",
        }

        phase_bank = banks.get(phase, {})
        options = phase_bank.get(lang, [])
        if options:
            return options[idx % len(options)]
        return default_bank.get(lang, default_bank["Anglais"])

    def _localized_closure(self, kind: str) -> str:
        if self.target_lang == "Français":
            if kind == "already_finished":
                return "L'entretien est déjà terminé. Merci encore pour votre temps."
            return "Merci pour cet échange. Nous reviendrons vers vous d'ici quelques jours."
        if self.target_lang == "Anglais":
            if kind == "already_finished":
                return "The interview is already complete. Thank you again for your time."
            return "Thank you for your time. We will get back to you with feedback within the next few days."
        if kind == "already_finished":
            return "المقابلة انتهت بالفعل، وشكراً مرة ثانية على وقتك."
        return "شكراً لوقتك، راح نرجع لك بالتغذية الراجعة خلال الأيام القادمة."

    def _candidate_speaker_label(self) -> str:
        return {"Français": "Candidat", "Anglais": "Candidate", "Arabe": "Candidate"}[self.target_lang]

    def _recruiter_speaker_label(self) -> str:
        return {"Français": "Recruteur", "Anglais": "Recruiter", "Arabe": "Recruiter"}[self.target_lang]

    @staticmethod
    def _is_candidate_speaker(speaker: str) -> bool:
        return "Candidate" in speaker or "Candidat" in speaker