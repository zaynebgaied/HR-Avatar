# stt_engine.py  —  Whisper STT  (RTX 2050 optimisé — float16 GPU / int8 CPU fallback)
# v4 : accumulation correcte + prompt candidat + VAD assoupli + log_prob ajusté
import torch
import numpy as np
from faster_whisper import WhisperModel
from device_config import DEVICE_CONFIG, log_device_config

# ── Constantes de classification des silences ─────────────────────────────────
# Un silence < SILENCE_SHORT_MS est ignoré (micro bruit, respiration)
SILENCE_SHORT_MS        = 300

# Entre SHORT et REFLECTION_MS → potentiellement fin de réplique
# Au-delà de REFLECTION_MS   → silence de réflexion (ou stress L99)
REFLECTION_THRESHOLD_MS = 1500

# Si l'énergie RMS du chunk de silence dépasse ce seuil,
# on considère qu'il y a encore du contenu (souffle, hésitation sonore "euh")
RMS_BREATH_THRESHOLD    = 0.008   # empirique, 16 kHz float32

# Ratio silence/parole au-delà duquel on penche vers "réflexion"
SILENCE_SPEECH_RATIO_REFLEX = 0.35

# Taux de mots par seconde en-dessous duquel on signale un débit lent (hésitation)
SLOW_SPEECH_WPS = 1.2


class SilenceEvent:
    """
    Représente un silence détecté entre deux segments de parole.
    Classifié comme 'end_of_turn' ou 'reflection' (avec sous-type stress possible).
    """
    END_OF_TURN = "end_of_turn"
    REFLECTION  = "reflection"
    STRESSED    = "stressed_reflection"   # reflection + flag vision stress actif

    def __init__(
        self,
        start_s: float,
        end_s: float,
        rms: float,
        speech_duration_s: float,
        vision_stress: bool = False,
    ):
        self.start_s            = start_s
        self.end_s              = end_s
        self.duration_ms        = round((end_s - start_s) * 1000)
        self.rms                = rms
        self.speech_duration_s  = speech_duration_s
        self.vision_stress      = vision_stress
        self.kind               = self._classify()

    def _classify(self) -> str:
        # 1. Silence court → fin de réplique par défaut
        if self.duration_ms < REFLECTION_THRESHOLD_MS:
            return self.END_OF_TURN

        # 2. Présence de souffle/hésitation sonore dans le silence
        has_breath = self.rms > RMS_BREATH_THRESHOLD

        # 3. Ratio silence/parole élevé → le candidat parle peu, hésite
        total = self.speech_duration_s + (self.duration_ms / 1000)
        ratio = (self.duration_ms / 1000) / total if total > 0 else 0
        high_ratio = ratio >= SILENCE_SPEECH_RATIO_REFLEX

        # 4. Vision stress
        if self.vision_stress:
            return self.STRESSED

        if has_breath or high_ratio:
            return self.REFLECTION

        # 5. Silence long mais propre et sans signal stress → fin de réplique longue
        return self.END_OF_TURN

    def to_dict(self) -> dict:
        return {
            "start_s":      round(self.start_s, 3),
            "end_s":        round(self.end_s, 3),
            "duration_ms":  self.duration_ms,
            "rms":          round(self.rms, 5),
            "vision_stress": self.vision_stress,
            "kind":         self.kind,
        }

    def __repr__(self):
        icon = {"end_of_turn": "✅", "reflection": "🤔", "stressed_reflection": "😰"}.get(self.kind, "?")
        return f"{icon} SilenceEvent({self.kind}, {self.duration_ms}ms, rms={self.rms:.4f})"


# ─────────────────────────────────────────────────────────────────────────────

def _compute_rms(audio: np.ndarray, start_s: float, end_s: float, sr: int = 16000) -> float:
    """RMS de la portion audio [start_s, end_s]. Retourne 0.0 si hors bornes."""
    i_start = int(start_s * sr)
    i_end   = int(end_s   * sr)
    i_start = max(0, i_start)
    i_end   = min(len(audio), i_end)
    if i_end <= i_start:
        return 0.0
    chunk = audio[i_start:i_end].astype(np.float32)
    return float(np.sqrt(np.mean(chunk ** 2)))


def _detect_silences(
    audio: np.ndarray,
    segments: list,
    sr: int = 16000,
    vision_stress: bool = False,
) -> list[SilenceEvent]:
    """
    À partir des segments Whisper (qui ne couvrent que les zones de parole),
    infère les silences inter-segments et les classifie.
    """
    if not segments:
        return []

    events: list[SilenceEvent] = []

    # Durée totale de parole dans cette réplique
    total_speech_s = sum(max(0.0, s.end - s.start) for s in segments)

    # Silence avant le premier segment (délai d'amorce)
    if segments[0].start > (SILENCE_SHORT_MS / 1000):
        sil_start = 0.0
        sil_end   = segments[0].start
        rms = _compute_rms(audio, sil_start, sil_end, sr)
        events.append(SilenceEvent(sil_start, sil_end, rms, total_speech_s, vision_stress))

    # Silences entre segments consécutifs
    for i in range(len(segments) - 1):
        sil_start = segments[i].end
        sil_end   = segments[i + 1].start
        dur_ms    = (sil_end - sil_start) * 1000
        if dur_ms < SILENCE_SHORT_MS:
            continue
        rms = _compute_rms(audio, sil_start, sil_end, sr)
        events.append(SilenceEvent(sil_start, sil_end, rms, total_speech_s, vision_stress))

    # Silence après le dernier segment (queue de fin)
    audio_duration_s = len(audio) / sr
    tail_start = segments[-1].end
    tail_dur_ms = (audio_duration_s - tail_start) * 1000
    if tail_dur_ms >= SILENCE_SHORT_MS:
        rms = _compute_rms(audio, tail_start, audio_duration_s, sr)
        events.append(SilenceEvent(tail_start, audio_duration_s, rms, total_speech_s, vision_stress))

    return events


def _compute_speech_stats(segments: list) -> dict:
    """Calcule quelques métriques de débit à partir des segments Whisper."""
    if not segments:
        return {"wps": 0.0, "total_words": 0, "total_speech_s": 0.0, "slow_speech": False}

    total_words = sum(len(s.text.split()) for s in segments)
    total_s     = sum(max(0.0, s.end - s.start) for s in segments)
    wps         = round(total_words / total_s, 2) if total_s > 0 else 0.0
    return {
        "wps":            wps,
        "total_words":    total_words,
        "total_speech_s": round(total_s, 2),
        "slow_speech":    wps < SLOW_SPEECH_WPS and total_words > 3,
    }


# ─────────────────────────────────────────────────────────────────────────────

class STTEngine:
    def __init__(self):
        self.model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"
        self.model      = None
        log_device_config("STT")
        self._load_model()

        self.multilingual_prompt = (
            "This is a job interview conversation in English, French, "
            "and Arabic (Saudi dialect - اللهجة السعودية). "
            "The speaker discusses professional experience and skills."
        )
        self.supported_languages = {"en", "fr", "ar"}

        # Référence optionnelle vers VisionEngine (injectée depuis main.py)
        self.vision_engine = None

        # ── FIX : Nom du candidat pour améliorer la transcription des noms propres
        # À injecter depuis main.py : stt_engine.candidate_name = "Zayneb Gaied"
        self.candidate_name: str | None = None

    # ── Chargement du modèle ──────────────────────────────────────────────────
    def _load_model(self):
        device       = DEVICE_CONFIG["whisper_device"]
        compute_type = DEVICE_CONFIG["whisper_compute_type"]
        vram         = DEVICE_CONFIG["vram_gb"]

        if device == "cuda":
            print(f"🚀 STT → CUDA float16  ({DEVICE_CONFIG['gpu_name']}, {vram} Go VRAM)")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16",
                    device_index=0,
                    num_workers=1,
                )
                self.device       = "cuda"
                self.compute_type = "float16"
                print("✅ STTEngine chargé sur CUDA (float16)")
                return
            except Exception as e:
                err = str(e)
                if any(k in err.lower() for k in ("cublas", "cudnn", "dll", "cuda")):
                    print(f"⚠️  CUDA indisponible ({err.split(chr(10))[0]}) → fallback CPU")
                else:
                    print(f"⚠️  Erreur CUDA inattendue : {e} → fallback CPU")

        print("⚙️  STT → CPU int8 (plus lent mais fonctionnel)")
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                num_workers=2,
            )
            self.device       = "cpu"
            self.compute_type = "int8"
            print("✅ STTEngine chargé sur CPU (int8)")
        except Exception as e:
            raise RuntimeError(f"❌ Chargement Whisper impossible (CPU) : {e}")

    # ── Normalisation audio ───────────────────────────────────────────────────
    def _normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        if audio_array is None or len(audio_array) == 0:
            raise ValueError("❌ Audio vide ou None reçu.")
        audio = np.array(audio_array, dtype=np.float32)
        if audio.dtype == np.int16 or audio.max() > 1.0:
            audio = audio.astype(np.float32) / 32768.0
        return np.clip(audio, -1.0, 1.0)

    # ── Construction du prompt enrichi avec le nom du candidat ───────────────
    def _build_prompt(self, language: str | None) -> str:
        """
        Construit le prompt initial Whisper.
        Si candidate_name est défini, il est injecté pour guider la transcription
        des noms propres (évite les substitutions phonétiques comme "Dynacade").
        """
        name_hint = ""
        if self.candidate_name:
            name_hint = f" Le candidat s'appelle {self.candidate_name}."

        prompt_map = {
            "fr": f"Entretien professionnel en français.{name_hint}",
            "en": f"Professional job interview in English.{name_hint}",
            "ar": f"مقابلة عمل باللغة العربية.{name_hint}",
        }
        return prompt_map.get(language or "", self.multilingual_prompt + name_hint)

    # ── Transcription ─────────────────────────────────────────────────────────
    # Phrases typiques d'hallucination Whisper — rejetées silencieusement
    _HALLUCINATION_PATTERNS = [
        "sous-titres", "sous titres", "subtitles", "subscribed", "subscribe",
        "merci d'avoir regardé", "thanks for watching", "thank you for watching",
        "...", "…", "so...", "so…", ". . .", "음", "ん", "um...", "uh...",
        "[musique]", "[music]", "[silence]", "[bruit]", "[noise]",
        "transcription", "translation", "translator",
    ]

    def _is_hallucination(self, text: str, audio: np.ndarray, sr: int = 16000) -> bool:
        """
        Retourne True si le texte ressemble à une hallucination Whisper.
        Critères :
          1. Texte trop court (< 2 mots) ET audio court (< 1.5s)
          2. Correspond à un pattern connu d'hallucination
          3. RMS global de l'audio trop faible (silence réel)
        """
        stripped = text.strip().lower().rstrip(".,!?…")
        words    = stripped.split()
        dur_s    = len(audio) / sr
        rms      = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

        # Silence réel : RMS très faible → hallucination certaine
        if rms < 0.005:
            print(f"[STT] 🚫 Hallucination (RMS={rms:.4f} < 0.005) : '{text}'")
            return True

        # Pattern connu
        for pat in self._HALLUCINATION_PATTERNS:
            if pat in stripped:
                print(f"[STT] 🚫 Hallucination (pattern '{pat}') : '{text}'")
                return True

        # Trop court ET audio bref → suspect
        if len(words) <= 1 and dur_s < 1.5:
            print(f"[STT] 🚫 Hallucination (1 mot, {dur_s:.1f}s) : '{text}'")
            return True

        return False

    def transcribe_stream(self, audio_array, language: str = None) -> tuple[list, str]:
        audio = self._normalize_audio(audio_array)

        if language and language not in self.supported_languages:
            print(f"⚠️ Langue '{language}' non supportée → auto-détection")
            language = None

        # ── FIX : prompt enrichi avec le nom du candidat
        prompt = self._build_prompt(language)

        beam_size = DEVICE_CONFIG.get("whisper_beam_size", 1)

        try:
            segments_gen, info = self.model.transcribe(
                audio,
                language=language,
                initial_prompt=prompt,
                beam_size=beam_size,
                best_of=1,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.45,              # FIX : était 0.55, moins agressif
                    min_speech_duration_ms=200,  # FIX : était 250
                    min_silence_duration_ms=300, # FIX : était 500, capture mieux les pauses naturelles
                    speech_pad_ms=200,           # FIX : était 100, plus de contexte autour des mots
                ),
                temperature=0.0,
                word_timestamps=True,
                condition_on_previous_text=True,  # FIX : était False, améliore la cohérence inter-segments
                no_speech_threshold=0.65,          # FIX : était 0.6, légèrement plus tolérant
                compression_ratio_threshold=2.0,
                log_prob_threshold=-1.2,           # FIX : était -1.0, accepte plus de segments incertains
            )

            segments      = list(segments_gen)
            detected_lang = info.language
            confidence    = round(info.language_probability, 3)

            if detected_lang not in self.supported_languages:
                print(f"⚠️ Langue détectée '{detected_lang}' hors périmètre (conf: {confidence})")
            else:
                print(f"🌐 Langue : {detected_lang} (conf: {confidence})  |  device: {self.device}")

            # ── Filtre anti-hallucination ──────────────────────────────────
            full_text = " ".join(s.text.strip() for s in segments)
            if full_text.strip() and self._is_hallucination(full_text, audio):
                return [], detected_lang   # retourner liste vide = rien à envoyer

            return segments, detected_lang

        except RuntimeError as e:
            err = str(e)
            if self.device == "cuda" and any(k in err.lower() for k in ("cublas", "dll", "oom", "out of memory")):
                print("⚠️  Erreur CUDA runtime → basculement CPU…")
                self._force_cpu_reload()
                return self.transcribe_stream(audio_array, language)
            raise RuntimeError(f"❌ Transcription échouée : {e}")

        except Exception as e:
            raise RuntimeError(f"❌ Transcription échouée : {e}")

    def _force_cpu_reload(self):
        try:
            print("⚙️  Rechargement Whisper sur CPU…")
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8", num_workers=2)
            self.device       = "cpu"
            self.compute_type = "int8"
            print("✅ STT rebasculé CPU (int8)")
        except Exception as e:
            raise RuntimeError(f"❌ Rechargement CPU impossible : {e}")

    # ── API principale enrichie ───────────────────────────────────────────────
    def get_full_text(self, audio_array, language: str = None) -> dict:
        """
        Retourne le texte transcrit + analyse complète des silences.

        Champs nouveaux dans le retour :
          silences        : liste de SilenceEvent.to_dict()
          silence_summary : {"end_of_turn": N, "reflection": N, "stressed_reflection": N}
          speech_stats    : {"wps": float, "total_words": int, "slow_speech": bool, ...}
          has_reflection  : bool  — au moins un silence de réflexion détecté
          has_stress      : bool  — au moins un silence de réflexion stressé détecté
        """
        audio    = self._normalize_audio(audio_array)
        segments, detected_lang = self.transcribe_stream(audio_array, language)

        # Lire le flag stress depuis VisionEngine si disponible
        vision_stress = False
        if self.vision_engine is not None:
            vision_stress = bool(getattr(self.vision_engine, "vision_stress_flag", False))

        # Analyse des silences
        silence_events = _detect_silences(audio, segments, sr=16000, vision_stress=vision_stress)
        speech_stats   = _compute_speech_stats(segments)

        # Résumé
        summary: dict[str, int] = {
            SilenceEvent.END_OF_TURN: 0,
            SilenceEvent.REFLECTION:  0,
            SilenceEvent.STRESSED:    0,
        }
        for ev in silence_events:
            summary[ev.kind] = summary.get(ev.kind, 0) + 1

        has_reflection = summary[SilenceEvent.REFLECTION] > 0 or summary[SilenceEvent.STRESSED] > 0
        has_stress     = summary[SilenceEvent.STRESSED] > 0

        # Log compact
        if silence_events:
            icons = {"end_of_turn": "✅", "reflection": "🤔", "stressed_reflection": "😰"}
            parts = [f"{icons.get(ev.kind,'?')}{ev.duration_ms}ms" for ev in silence_events]
            slow  = " | 🐢 débit lent" if speech_stats["slow_speech"] else ""
            print(f"🔇 Silences : {' '.join(parts)}{slow}")

        full_text = " ".join(seg.text.strip() for seg in segments)

        return {
            "text":     full_text,
            "language": detected_lang,
            "segments": [
                {
                    "start": seg.start,
                    "end":   seg.end,
                    "text":  seg.text.strip(),
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in (seg.words or [])
                    ],
                }
                for seg in segments
            ],
            # ── Nouveau ──────────────────────────────────────────────────────
            "silences":        [ev.to_dict() for ev in silence_events],
            "silence_summary": summary,
            "speech_stats":    speech_stats,
            "has_reflection":  has_reflection,
            "has_stress":      has_stress,
        }