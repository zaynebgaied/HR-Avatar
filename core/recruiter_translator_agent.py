"""
recruiter_translator_agent.py
─────────────────────────────────────────────────────────────────────────────
Pattern  : Orchestrateur / Traducteur (Multi-Agents)
Objectif : Séparer la *réflexion* (anglais) de la *traduction* (dialecte saoudien)
           pour garantir une qualité naturelle en arabe sans dégrader le raisonnement.

V2 — Correctifs drift :
  • Détection chinois/japonais/coréen → dérive immédiate
  • _TRANSLATOR_SYSTEM et _DRIFT_RETRY_SYSTEM interdisent explicitement les caractères CJK
  • MAX_DRIFT_RETRIES porté à 3
"""

from __future__ import annotations

import re
import time
from typing import Optional

from ollama import Client


# ── Tokens techniques anglais autorisés dans la sortie arabe ─────────────────
_ALLOWED_TECH_TOKENS = frozenset({
    "api", "apis", "pipeline", "pipelines", "deploy", "deployment",
    "backend", "frontend", "ml", "llm", "rag", "etl", "dag", "ci", "cd",
    "docker", "kubernetes", "k8s", "sql", "nosql", "redis", "kafka",
    "spark", "airflow", "terraform", "git", "github", "cloud", "sla",
    "slo", "rpc", "rest", "graphql", "microservice", "microservices",
    "log", "logs", "logging", "monitoring", "alert", "alerts", "cache",
    "caching", "queue", "queues", "batch", "streaming", "latency",
    "throughput", "benchmark", "commit", "merge", "branch", "rollback",
    "rollout", "canary", "ab", "a/b", "pr", "devops", "mlops",
    "data", "dataset", "feature", "model", "inference", "training",
    "fine-tuning", "prompt", "embedding", "embeddings", "vector",
    "index", "retrieval", "reranking", "chunking", "token", "tokens",
    "server", "client", "load", "balancer", "cdn", "s3", "ec2",
    "bigquery", "dbt", "flink", "beam",
    # Marketing / domaine métier
    "ctr", "cpc", "cpa", "roas", "cac", "roi", "kpi", "seo", "sem",
    "ads", "meta", "google", "ga4", "pixel", "conversion", "funnel",
    "canva", "capcut", "reels", "stories", "carousel", "creative",
    "testing", "retargeting", "lookalike", "audience",
    "email", "crm", "erp", "saas", "b2b", "b2c", "e-commerce",
})

# ── Patterns de dérive — Français ────────────────────────────────────────────
_FRENCH_DRIFT_RE = re.compile(
    r"\b(le|la|les|un|une|des|je|vous|nous|ils|est|sont|avez|pour|dans|sur|avec"
    r"|que|qui|comment|pourquoi|quel|quelle|voici|passons|donnez|dites|décrivez"
    r"|très|bien|merci|bonjour|donc|mais|aussi|encore|pouvez|voulez|devez)\b",
    re.IGNORECASE,
)

# ── Patterns de dérive — Anglais courant ─────────────────────────────────────
_ENGLISH_DRIFT_RE = re.compile(
    r"\b(the|a\b|an\b|is|are|was|were|have|has|do|does|did|will|would|could|should"
    r"|tell|walk|give|describe|what|how|why|when|where|and|or|but|so|if|then"
    r"|great|good|noted|thank|sure|absolutely|certainly|of course|your|you|me|my)\b",
    re.IGNORECASE,
)

# ── Patterns de dérive — Chinois / Japonais / Coréen (CJK) ───────────────────
_CJK_DRIFT_RE = re.compile(
    r"[\u4e00-\u9fff"      # CJK Unified Ideographs (chinois simplifié/traditionnel)
    r"\u3400-\u4dbf"       # CJK Extension A
    r"\uff00-\uffef"       # Halfwidth and Fullwidth Forms
    r"\u3040-\u309f"       # Hiragana (japonais)
    r"\u30a0-\u30ff"       # Katakana (japonais)
    r"\uac00-\ud7af"       # Hangul Syllables (coréen)
    r"\u3000-\u303f]"      # CJK Symbols and Punctuation
)


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT GUARD
# ─────────────────────────────────────────────────────────────────────────────

def _is_drift_free(text: str) -> bool:
    """
    Retourne True si le texte est du dialecte saoudien propre.

    Échecs immédiats :
      - Présence de caractères CJK (chinois/japonais/coréen)
      - Ratio arabe < 25 % sur un texte de plus de 6 mots
      - 2+ mots français non techniques
      - 3+ mots anglais courants non techniques
    """
    # ── 1. Chinois / Japonais / Coréen → dérive immédiate ────────────────────
    if _CJK_DRIFT_RE.search(text):
        return False

    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    total_alpha   = len(re.findall(r"[A-Za-z\u0600-\u06FF]", text))

    if total_alpha == 0:
        return True  # texte vide ou uniquement chiffres/ponctuation

    # ── 2. Ratio arabe trop faible ────────────────────────────────────────────
    if arabic_chars / total_alpha < 0.25 and len(text.split()) > 6:
        return False

    # ── 3. Mots français non techniques ──────────────────────────────────────
    fr_hits = _FRENCH_DRIFT_RE.findall(text)
    non_tech_fr = [w for w in fr_hits if w.lower() not in _ALLOWED_TECH_TOKENS]
    if len(non_tech_fr) >= 2:
        return False

    # ── 4. Mots anglais courants non techniques ───────────────────────────────
    en_hits = _ENGLISH_DRIFT_RE.findall(text)
    non_tech_en = [w for w in en_hits if w.lower() not in _ALLOWED_TECH_TOKENS]
    if len(non_tech_en) >= 3:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_RECRUITER_SYSTEM = """
You are a senior hiring manager conducting a structured technical interview.

OUTPUT LANGUAGE: English only — even if the briefing contains Arabic, French, or Chinese text.

STRICT RULES:
- Produce exactly ONE sharp, focused follow-up question.
- Maximum 2 sentences.
- Never praise the candidate.
- Never start with filler (Noted, Good, I see, Absolutely…).
- Never summarize the candidate's answer before the question.
- Ask about ownership, metrics, decisions, or concrete evidence.
- Output plain text only. No bullets, no markdown, no labels.
- NEVER write Chinese, Japanese, or Korean characters.
""".strip()

_TRANSLATOR_SYSTEM = """
You are a specialist translator from English to Saudi Arabic dialect (العامية السعودية / اللهجة السعودية).

YOUR ONLY JOB: translate the recruiter's question into natural Saudi Arabic dialect.

STRICT RULES:
1. Preserve 100 % of the original intent — do NOT soften, change, or omit any part of the question.
2. Write in Saudi colloquial Arabic (عامية), NOT Modern Standard Arabic (فصحى).
   - Use: وش / إيش / ليش / إيش سويت / أخذنا / خذني / ودي أفهم…
   - NOT: ما الذي / كيف يمكن / هل يمكنك / حدثني…
3. Technical English terms are ALLOWED and expected:
   pipeline, deploy, backend, API, ML, LLM, RAG, ETL, rollback, CTR, ROAS, GA4…
4. Keep it natural: 1–2 sentences, ending with ؟
5. DO NOT add praise, filler, or acknowledgement before the question.
6. DO NOT start with: حسناً / بالتأكيد / طبعاً / ممتاز / رائع / شكراً
7. NEVER write Chinese, Japanese, or Korean characters under any circumstances.
   If you feel the urge to write CJK characters, write Arabic instead.
8. Output ONLY the translated question — nothing else.
""".strip()

_DRIFT_RETRY_SYSTEM = """
أنت مترجم متخصص للعامية السعودية.

المهمة: أعد صياغة السؤال التالي بالعامية السعودية فقط.
- استخدم: وش / إيش / ليش / خذني / أخبرني / ودي أفهم
- المصطلحات التقنية الإنجليزية مسموحة (API, pipeline, deploy, CTR, ROAS, GA4…)
- ممنوع: أي كلمة فرنسية أو إنجليزية عادية
- ممنوع تماماً: أي حرف صيني أو ياباني أو كوري — إذا ظهرت هذه الحروف في الإدخال، تجاهلها كلياً
- ممنوع: حسناً / بالتأكيد / رائع / طبعاً في البداية
- انتهِ بعلامة ؟
- أخرج النص فقط، بلا شرح.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OrchestrateurTraducteur:
    """
    Orchestrateur multi-agents pour la génération de questions en dialecte saoudien.

    Workflow :
        1. Agent Recruteur  → relance en anglais (meilleure qualité de raisonnement)
        2. Agent Traducteur → dialecte saoudien naturel
        3. Drift guard      → re-traduction si dérive détectée (max 3 tentatives)
                             Dérive CJK (chinois/japonais/coréen) → échec immédiat

    Usage dans llm_chain.py :
        from recruiter_translator_agent import OrchestrateurTraducteur
        # Dans __init__ (après self.client et self.model_name) :
        self._ot_agent = OrchestrateurTraducteur(self.client, self.model_name)
        # Dans _build_next_question :
        if self.target_lang == "Arabe":
            question = self._ot_agent.generate(briefing, phase=phase)
    """

    MAX_DRIFT_RETRIES: int = 3  # porté à 3 pour mieux absorber les dérives CJK

    def __init__(self, client: Client, model_name: str):
        self.client     = client
        self.model_name = model_name
        self._base_opts = {
            "temperature":    0.3,
            "top_p":          0.9,
            "num_predict":    300,
            "repeat_penalty": 1.1,
            "top_k":          40,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _stream_collect(self, stream_iter, timeout: float = 90.0) -> str:
        """Collecte une réponse streamée en une seule chaîne."""
        parts: list[str] = []
        deadline = time.time() + timeout
        for chunk in stream_iter:
            if time.time() > deadline:
                break
            if hasattr(chunk, "message"):
                parts.append(chunk.message.content or "")
            elif isinstance(chunk, dict):
                parts.append(
                    chunk.get("message", {}).get("content", "")
                    or chunk.get("response", "")
                )
        return "".join(parts).strip()

    def _call(self, system: str, user: str) -> str:
        """Appel LLM bas niveau avec streaming."""
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options=self._base_opts,
                stream=True,
            )
            return self._stream_collect(stream)
        except Exception as exc:
            print(f"[OrchestrateurTraducteur] _call error: {exc}")
            return ""

    @staticmethod
    def _strip_cjk(text: str) -> str:
        """Supprime les caractères CJK d'une chaîne (nettoyage de secours)."""
        return _CJK_DRIFT_RE.sub("", text).strip()

    @staticmethod
    def _clean_output(text: str) -> str:
        """Supprime les fences markdown, labels et caractères CJK résiduels."""
        text = re.sub(r"^```[a-zA-Z]*", "", text).replace("```", "").strip()
        text = re.sub(
            r"^(Question|Recruiter|Recruteur|المحاور)\s*:\s*",
            "", text, flags=re.I,
        ).strip()
        # Supprimer les caractères CJK résiduels
        text = _CJK_DRIFT_RE.sub("", text).strip()
        return text

    # ── Agent 1 : Recruiter ───────────────────────────────────────────────────

    def _agent_recruiter(self, briefing: str, phase: Optional[str]) -> str:
        """
        Raisonne sur le contexte RAG + l'état de l'entretien et produit
        une relance en ANGLAIS uniquement.
        """
        phase_hint = f"[PHASE: {phase}]" if phase else ""
        user_prompt = f"{phase_hint}\n\n{briefing}"
        raw = self._call(_RECRUITER_SYSTEM, user_prompt)
        return self._clean_output(raw)

    # ── Agent 2 : Translator ──────────────────────────────────────────────────

    def _agent_translator(self, english_question: str) -> str:
        """
        Traduit la relance anglaise en dialecte saoudien naturel.
        Ne reçoit QUE la question — pas le contexte RAG ni la logique de
        l'entretien, pour éviter toute confusion de registre.
        """
        user_prompt = (
            "Translate this interviewer question into natural Saudi Arabic dialect.\n"
            "IMPORTANT: Output Arabic text only. Never output Chinese, Japanese, or Korean.\n\n"
            f"{english_question}"
        )
        raw = self._call(_TRANSLATOR_SYSTEM, user_prompt)
        return self._clean_output(raw)

    # ── Drift guard ───────────────────────────────────────────────────────────

    def _drift_retry(self, corrupted_text: str) -> str:
        """
        Re-traduit en cas de dérive linguistique détectée.
        Nettoie d'abord les caractères CJK du texte corrompu avant de renvoyer.
        """
        cleaned_input = self._strip_cjk(corrupted_text)
        if not cleaned_input:
            return ""

        user_prompt = (
            "هذا السؤال فيه مشكلة في اللغة، أعد صياغته بالعامية السعودية فقط.\n"
            "لا تكتب أي حرف صيني أو ياباني أو كوري.\n\n"
            f"{cleaned_input}"
        )
        raw = self._call(_DRIFT_RETRY_SYSTEM, user_prompt)
        return self._clean_output(raw)

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(phase: Optional[str]) -> str:
        """Question de secours en dialecte saoudien si tous les agents échouent."""
        banks = {
            "OPENING":                 "وش الخبرة في مسارك اللي تشوفها الأقرب لهذا الدور وليش؟",
            "JOB_ALIGNED_EXPLORATION": "أعطني مثالاً عملياً على استخدامك لإحدى التقنيات الأساسية في هذا الدور.",
            "PROJECT_DEEP_DIVE":       "وش قرار architecture امتلكته أنت شخصياً في هذا المشروع؟",
            "TECHNICAL_DEPTH":         "وش النتيجة الرقمية اللي تقدر تدافع عنها من هذا العمل؟",
            "SOFT_SKILLS_BEHAVIORAL":  "أعطني مثالاً محدداً على خلاف تقني مع زميل.",
        }
        return banks.get(phase or "", "أعطني مثالاً ملموساً على ذلك؟")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        briefing: str,
        phase: Optional[str] = None,
        english_override: Optional[str] = None,
    ) -> str:
        """
        Point d'entrée principal.

        Args:
            briefing         : le briefing complet (output de _build_agent_brief)
            phase            : la phase courante de l'entretien (str ou None)
            english_override : si fourni, saute l'Agent 1 et traduit directement
                               (utile pour les questions déterministes comme les
                               questions comportementales pré-rédigées)

        Returns:
            str : question en dialecte saoudien, drift-free.
        """
        # ── Étape 1 : Agent Recruteur ─────────────────────────────────────────
        if english_override:
            english_q = english_override.strip()
        else:
            english_q = self._agent_recruiter(briefing, phase)

        if not english_q:
            return self._fallback(phase)

        # Sécurité : si l'agent Recruteur a glissé en CJK, on ne traduit pas du garbage
        if _CJK_DRIFT_RE.search(english_q):
            print("[OrchestrateurTraducteur] CJK detected in recruiter output — using fallback")
            return self._fallback(phase)

        # ── Étape 2 : Agent Traducteur ────────────────────────────────────────
        arabic_q = self._agent_translator(english_q)

        if not arabic_q:
            return self._fallback(phase)

        # ── Étape 3 : Drift guard ─────────────────────────────────────────────
        for attempt in range(self.MAX_DRIFT_RETRIES):
            if _is_drift_free(arabic_q):
                break

            drift_type = "CJK" if _CJK_DRIFT_RE.search(arabic_q) else "lang"
            print(
                f"[OrchestrateurTraducteur] {drift_type} drift detected "
                f"(attempt {attempt + 1}/{self.MAX_DRIFT_RETRIES}) — retrying"
            )
            arabic_q = self._drift_retry(arabic_q)
            if not arabic_q:
                return self._fallback(phase)
        else:
            # Après MAX_DRIFT_RETRIES tentatives toujours avec dérive → fallback
            if not _is_drift_free(arabic_q):
                print("[OrchestrateurTraducteur] drift persists after retries — using fallback")
                return self._fallback(phase)

        return arabic_q

    def generate_from_english(
        self,
        english_question: str,
        phase: Optional[str] = None,
    ) -> str:
        """
        Raccourci : traduit directement une question anglaise pré-rédigée.
        Utile pour les questions comportementales statiques (BEHAVIORAL_QUESTIONS,
        DECISION_MAKING_QUESTIONS) afin de les passer par le même pipeline de qualité.
        """
        return self.generate(
            briefing="",
            phase=phase,
            english_override=english_question,
        )