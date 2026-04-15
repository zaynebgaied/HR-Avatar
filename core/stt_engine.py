# stt_engine.py  —  Whisper STT  (RTX 2050 optimisé — float16 GPU / int8 CPU fallback)
# v5 : VAD Silero intégré + transcription multilingue renforcée (FR / AR-SA / EN)
#      Pipeline : VAD pré-segmentation → Whisper par segment → fusion → analyse silences
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from transcript_cleaner import (
    clean_transcript,
    join_cleaned_segments,
    is_probable_hallucination_text_only,
)

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Literal

import numpy as np
import torch
from faster_whisper import WhisperModel
from device_config import DEVICE_CONFIG, log_device_config


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE             = 16_000    # Hz — Whisper natif

# ── Silences ──────────────────────────────────────────────────────────────────
SILENCE_SHORT_MS        = 300       # < ce seuil : ignoré (respiration, bruit)
REFLECTION_THRESHOLD_MS = 1500      # ≥ ce seuil : silence de réflexion
RMS_BREATH_THRESHOLD    = 0.008     # énergie résiduelle = souffle/hésitation sonore
SILENCE_SPEECH_RATIO_REFLEX = 0.35  # ratio silence/parole → hésitation
SLOW_SPEECH_WPS         = 1.2       # mots/sec en-dessous = débit lent

# ── VAD Silero (pré-segmentation) ─────────────────────────────────────────────
VAD_THRESHOLD           = 0.30      # plus sensible pour ne pas couper le début de parole
VAD_MIN_SPEECH_MS       = 80        # plus tolérant pour capter les attaques de phrase
VAD_MIN_SILENCE_MS      = 180       # évite de casser trop tôt la parole
VAD_SPEECH_PAD_MS       = 450       # marge plus large pour garder le début de réplique
VAD_WINDOW_SAMPLES      = 512       # fenêtre Silero (512 ou 1536 pour 16 kHz)

# ── Whisper ───────────────────────────────────────────────────────────────────
BEAM_SIZE_DEFAULT       = 5         # qualité maximale (GPU)
NO_SPEECH_THRESHOLD     = 0.72
COMPRESSION_THRESHOLD   = 2.0
LOG_PROB_THRESHOLD      = -0.8
MAX_SEGMENT_WORDS       = 7         # garde-fou réaliste sur le débit humain


# ══════════════════════════════════════════════════════════════════════════════
#  SILENCE EVENT
# ══════════════════════════════════════════════════════════════════════════════

class SilenceEvent:
    """
    Silence détecté entre deux segments de parole.
    Classifié comme 'end_of_turn', 'reflection' ou 'stressed_reflection'.
    """
    END_OF_TURN = "end_of_turn"
    REFLECTION  = "reflection"
    STRESSED    = "stressed_reflection"

    def __init__(
        self,
        start_s: float,
        end_s: float,
        rms: float,
        speech_duration_s: float,
        vision_stress: bool = False,
    ):
        self.start_s           = start_s
        self.end_s             = end_s
        self.duration_ms       = round((end_s - start_s) * 1000)
        self.rms               = rms
        self.speech_duration_s = speech_duration_s
        self.vision_stress     = vision_stress
        self.kind              = self._classify()

    def _classify(self) -> str:
        if self.duration_ms < REFLECTION_THRESHOLD_MS:
            return self.END_OF_TURN
        has_breath = self.rms > RMS_BREATH_THRESHOLD
        total      = self.speech_duration_s + (self.duration_ms / 1000)
        ratio      = (self.duration_ms / 1000) / total if total > 0 else 0
        high_ratio = ratio >= SILENCE_SPEECH_RATIO_REFLEX
        if self.vision_stress:
            return self.STRESSED
        if has_breath or high_ratio:
            return self.REFLECTION
        return self.END_OF_TURN

    def to_dict(self) -> dict:
        return {
            "start_s":       round(self.start_s, 3),
            "end_s":         round(self.end_s, 3),
            "duration_ms":   self.duration_ms,
            "rms":           round(self.rms, 5),
            "vision_stress": self.vision_stress,
            "kind":          self.kind,
        }

    def __repr__(self):
        icon = {"end_of_turn": "✅", "reflection": "🤔", "stressed_reflection": "😰"}.get(self.kind, "?")
        return f"{icon} SilenceEvent({self.kind}, {self.duration_ms}ms, rms={self.rms:.4f})"


# ══════════════════════════════════════════════════════════════════════════════
#  VAD SILERO — pré-segmentation audio
# ══════════════════════════════════════════════════════════════════════════════

class SileroVAD:
    """
    Encapsule le modèle Silero VAD pour détecter les segments de parole
    avant de les envoyer à Whisper — améliore la rigueur pour les 3 langues.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._load()

    def _load(self):
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model.to(self.device)
            self._model.eval()
            self._get_speech_timestamps = utils[0]
            print(f"✅ Silero VAD chargé ({self.device})")
        except Exception as e:
            print(f"⚠️  Silero VAD indisponible : {e} — fallback Whisper VAD interne")
            self._model = None
            self._get_speech_timestamps = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def get_speech_segments(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
    ) -> list[dict] | None:
        if not self.available or len(audio) == 0:
            return None

        try:
            tensor = torch.from_numpy(audio).float()
            if self.device == "cuda":
                tensor = tensor.to("cuda")

            speech_ts = self._get_speech_timestamps(
                tensor,
                self._model,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=VAD_MIN_SPEECH_MS,
                min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                speech_pad_ms=VAD_SPEECH_PAD_MS,
                window_size_samples=VAD_WINDOW_SAMPLES,
                return_seconds=True,
                sampling_rate=sr,
            )
            if not speech_ts:
                return []
            return speech_ts

        except Exception as e:
            print(f"⚠️  VAD segmentation échouée ({e}) — fallback Whisper vad_filter")
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES AUDIO & MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def _compute_rms(audio: np.ndarray, start_s: float, end_s: float, sr: int = SAMPLE_RATE) -> float:
    i0 = max(0, int(start_s * sr))
    i1 = min(len(audio), int(end_s * sr))
    if i1 <= i0:
        return 0.0
    chunk = audio[i0:i1].astype(np.float32)
    return float(np.sqrt(np.mean(chunk ** 2)))


def _detect_silences(
    audio: np.ndarray,
    segments: list,
    sr: int = SAMPLE_RATE,
    vision_stress: bool = False,
) -> list[SilenceEvent]:
    if not segments:
        return []

    events: list[SilenceEvent] = []
    total_speech_s = sum(max(0.0, s.end - s.start) for s in segments)

    if segments[0].start > (SILENCE_SHORT_MS / 1000):
        rms = _compute_rms(audio, 0.0, segments[0].start, sr)
        events.append(SilenceEvent(0.0, segments[0].start, rms, total_speech_s, vision_stress))

    for i in range(len(segments) - 1):
        sil_start = segments[i].end
        sil_end   = segments[i + 1].start
        dur_ms    = (sil_end - sil_start) * 1000
        if dur_ms < SILENCE_SHORT_MS:
            continue
        rms = _compute_rms(audio, sil_start, sil_end, sr)
        events.append(SilenceEvent(sil_start, sil_end, rms, total_speech_s, vision_stress))

    audio_dur_s = len(audio) / sr
    tail_start  = segments[-1].end
    if (audio_dur_s - tail_start) * 1000 >= SILENCE_SHORT_MS:
        rms = _compute_rms(audio, tail_start, audio_dur_s, sr)
        events.append(SilenceEvent(tail_start, audio_dur_s, rms, total_speech_s, vision_stress))

    return events


def _compute_speech_stats(segments: list) -> dict:
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


def _expand_and_merge_segments(segments: list[dict], audio_len_s: float) -> list[dict]:
    """Agrandit les segments VAD pour ne pas couper l'attaque de phrase,
    puis fusionne les segments trop proches."""
    if not segments:
        return []

    expanded: list[dict] = []
    lead_pad_s = 0.22
    tail_pad_s = 0.30
    merge_gap_s = 0.42

    for idx, seg in enumerate(segments):
        start = max(0.0, float(seg["start"]) - lead_pad_s)
        end = min(audio_len_s, float(seg["end"]) + tail_pad_s)

        if idx == 0 and float(seg["start"]) < 0.9:
            start = 0.0

        if expanded and start <= expanded[-1]["end"] + merge_gap_s:
            expanded[-1]["end"] = max(expanded[-1]["end"], end)
        else:
            expanded.append({"start": start, "end": end})

    return expanded


# ══════════════════════════════════════════════════════════════════════════════
#  STT ENGINE v5
# ══════════════════════════════════════════════════════════════════════════════

class STTEngine:
    """
    Moteur de transcription Whisper avec :
      - VAD Silero en pré-segmentation
      - Prompts contextuels courts (sans mots-clés qui déclenchent des hallucinations)
      - Anti-hallucination renforcé
      - Analyse complète des silences
    """

    MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

    SUPPORTED_LANGUAGES = {"en", "fr", "ar"}
    ARABIC_VARIANTS = {"ar", "ara", "arb", "ars"}

    # ── Détection dynamique : pas de liste statique d'hallucinations ───────
    # ── Prompts courts : amorce de contexte UNIQUEMENT, sans listes de mots ──
    # Règle : une seule phrase courte, aucun deux-points ni énumération.
    # Les énumérations poussent Whisper à générer exactement ces mots (hallucination écho).
    _BASE_PROMPTS = {
        "fr": "Entretien d'embauche en français.",
        "en": "Professional job interview in English.",
        "ar": "مقابلة عمل احترافية باللغة العربية.",
    }

    def __init__(self):
        self.model      = None
        self.device     = "cpu"
        self.compute_type = "int8"

        log_device_config("STT")
        self._load_model()

        self.vad = SileroVAD(device=self.device if self.device == "cuda" else "cpu")

        self.vision_engine: object | None = None
        self.candidate_name: str | None = None

    # ── Chargement du modèle ─────────────────────────────────────────────────

    def _load_model(self):
        device       = DEVICE_CONFIG["whisper_device"]
        compute_type = DEVICE_CONFIG["whisper_compute_type"]
        vram         = DEVICE_CONFIG.get("vram_gb", 0)

        if device == "cuda":
            print(f"🚀 STT → CUDA float16  ({DEVICE_CONFIG.get('gpu_name','GPU')}, {vram} Go VRAM)")
            try:
                self.model = WhisperModel(
                    self.MODEL_SIZE,
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
                if any(k in err.lower() for k in ("cublas", "cudnn", "dll", "cuda", "oom")):
                    print(f"⚠️  CUDA indisponible ({err.split(chr(10))[0]}) → fallback CPU")
                else:
                    print(f"⚠️  Erreur CUDA inattendue : {e} → fallback CPU")

        print("⚙️  STT → CPU int8")
        try:
            self.model = WhisperModel(
                self.MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                num_workers=2,
            )
            self.device       = "cpu"
            self.compute_type = "int8"
            print("✅ STTEngine chargé sur CPU (int8)")
        except Exception as e:
            raise RuntimeError(f"❌ Chargement Whisper impossible : {e}")

    def _force_cpu_reload(self):
        print("⚙️  Rechargement Whisper sur CPU…")
        try:
            self.model = WhisperModel(
                self.MODEL_SIZE, device="cpu", compute_type="int8", num_workers=2
            )
            self.device       = "cpu"
            self.compute_type = "int8"
            print("✅ STT rebasculé CPU (int8)")
        except Exception as e:
            raise RuntimeError(f"❌ Rechargement CPU impossible : {e}")

    # ── Normalisation audio ───────────────────────────────────────────────────

    def _normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        if audio_array is None or len(audio_array) == 0:
            raise ValueError("❌ Audio vide ou None reçu.")
        audio = np.array(audio_array, dtype=np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0
        return np.clip(audio, -1.0, 1.0)

    # ── Construction du prompt Whisper ───────────────────────────────────────

    def _build_prompt(self, language: str | None) -> str:
        """
        Prompt court et neutre par langue + nom optionnel du candidat.
        Aucun mot-clé ni énumération : évite les hallucinations-écho de Whisper.
        """
        name_hint_fr = f" Le candidat s'appelle {self.candidate_name}." if self.candidate_name else ""
        name_hint_en = f" The candidate's name is {self.candidate_name}." if self.candidate_name else ""
        name_hint_ar = f" اسم المرشح {self.candidate_name}." if self.candidate_name else ""

        prompts = {
            "fr": self._BASE_PROMPTS["fr"] + name_hint_fr,
            "en": self._BASE_PROMPTS["en"] + name_hint_en,
            "ar": self._BASE_PROMPTS["ar"] + name_hint_ar,
        }

        if language in prompts:
            return prompts[language]

        # Multilingue : concaténation des 3 pour aider la détection automatique
        return (
            self._BASE_PROMPTS["fr"] + name_hint_fr + " | " +
            self._BASE_PROMPTS["en"] + name_hint_en + " | " +
            self._BASE_PROMPTS["ar"] + name_hint_ar
        )

    # ── Normalisation de la langue détectée ──────────────────────────────────

    def _normalize_language(self, detected: str) -> str:
        low = (detected or "").lower()
        if low in self.ARABIC_VARIANTS:
            return "ar"
        return low

    # ── Anti-hallucination ───────────────────────────────────────────────────

    def _is_hallucination(
        self,
        text: str,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        avg_logprob: float | None = None,
        no_speech_prob: float | None = None,
    ) -> bool:
        stripped = (text or "").strip()
        dur_s = len(audio) / sr if sr > 0 else 0.0

        if not stripped:
            print("[STT] 🚫 Hallucination: texte vide")
            return True

        if len(audio) == 0:
            print("[STT] 🚫 Hallucination: audio vide")
            return True

        audio_f = audio.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio_f ** 2))) if len(audio_f) else 0.0
        peak = float(np.max(np.abs(audio_f))) if len(audio_f) else 0.0

        if rms < 0.0022 and peak < 0.010:
            print(f"[STT] 🚫 Silence/quasi-silence (rms={rms:.4f}, peak={peak:.4f})")
            return True

        if no_speech_prob is not None and no_speech_prob >= 0.85 and rms < 0.006:
            print(f"[STT] 🚫 no_speech_prob élevé ({no_speech_prob:.2f})")
            return True

        if avg_logprob is not None and avg_logprob < -1.10 and rms < 0.008:
            print(f"[STT] 🚫 avg_logprob faible ({avg_logprob:.2f})")
            return True

        if is_probable_hallucination_text_only(stripped, dur_s):
            print(f"[STT] 🚫 Hallucination dynamique: '{stripped[:80]}'")
            return True

        return False

    # ── Transcription d'un segment audio isolé ───────────────────────────────

    def _transcribe_segment(
        self,
        audio_chunk: np.ndarray,
        language: str | None,
        time_offset: float = 0.0,
        use_whisper_vad: bool = False,
    ) -> list:
        prompt    = self._build_prompt(language)
        beam_size = DEVICE_CONFIG.get("whisper_beam_size", BEAM_SIZE_DEFAULT)

        segments_gen, info = self.model.transcribe(
            audio_chunk,
            language=language,
            initial_prompt=prompt,
            beam_size=beam_size,
            best_of=1,
            vad_filter=use_whisper_vad,
            temperature=0.0,
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            compression_ratio_threshold=COMPRESSION_THRESHOLD,
            log_prob_threshold=LOG_PROB_THRESHOLD,
        )

        raw_segments = list(segments_gen)

        class _OffsetSegment:
            __slots__ = ("start", "end", "text", "words", "language", "avg_logprob", "no_speech_prob", "compression_ratio")

            def __init__(self, seg, offset: float, lang: str):
                self.start    = seg.start + offset
                self.end      = seg.end   + offset
                self.text     = seg.text
                self.language = lang
                self.avg_logprob = getattr(seg, "avg_logprob", None)
                self.no_speech_prob = getattr(seg, "no_speech_prob", None)
                self.compression_ratio = getattr(seg, "compression_ratio", None)

                class _W:
                    __slots__ = ("word", "start", "end")
                    def __init__(self, w, o):
                        self.word  = w.word
                        self.start = w.start + o
                        self.end   = w.end   + o

                self.words = [_W(w, offset) for w in (seg.words or [])]

        detected_lang = self._normalize_language(info.language)
        return [_OffsetSegment(s, time_offset, detected_lang) for s in raw_segments]

    # ── Pipeline principal : VAD → Whisper → fusion ──────────────────────────

    def transcribe_stream(self, audio_array: np.ndarray, language: str | None = None) -> tuple[list, str]:
        audio = self._normalize_audio(audio_array)

        if language and language not in self.SUPPORTED_LANGUAGES:
            print(f"⚠️ Langue '{language}' non supportée → auto-détection")
            language = None

        full_rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if len(audio) else 0.0
        full_peak = float(np.max(np.abs(audio.astype(np.float32)))) if len(audio) else 0.0
        if full_rms < 0.0020 and full_peak < 0.010:
            print(f"[STT] 🔇 Buffer silencieux (rms={full_rms:.4f}, peak={full_peak:.4f})")
            return [], language or "fr"

        speech_segments = self.vad.get_speech_segments(audio, sr=SAMPLE_RATE)
        use_whisper_vad = speech_segments is None
        if use_whisper_vad:
            speech_segments = [{"start": 0.0, "end": len(audio) / SAMPLE_RATE}]
        elif not speech_segments:
            print("[STT] 🔇 Aucun segment de parole détecté par le VAD")
            return [], language or "fr"
        else:
            speech_segments = _expand_and_merge_segments(speech_segments, len(audio) / SAMPLE_RATE)

        print(f"[STT] 🎙️  VAD → {len(speech_segments)} segment(s) de parole"
              + (" (fallback Whisper vad_filter)" if use_whisper_vad else ""))

        all_segments = []
        lang_votes: dict[str, int] = {}

        for idx, seg in enumerate(speech_segments):
            seg_start   = seg["start"]
            seg_end     = seg["end"]
            i0          = int(seg_start * SAMPLE_RATE)
            i1          = min(len(audio), int(seg_end * SAMPLE_RATE))
            audio_chunk = audio[i0:i1]

            dur_s = (i1 - i0) / SAMPLE_RATE
            if dur_s < 0.15:
                continue

            try:
                segs = self._transcribe_segment(
                    audio_chunk, language,
                    time_offset=seg_start,
                    use_whisper_vad=use_whisper_vad,
                )
            except RuntimeError as e:
                err = str(e)
                if self.device == "cuda" and any(k in err.lower() for k in ("cublas", "oom", "out of memory")):
                    print("⚠️  Erreur CUDA → basculement CPU…")
                    self._force_cpu_reload()
                    segs = self._transcribe_segment(
                        audio_chunk, language,
                        time_offset=seg_start,
                        use_whisper_vad=use_whisper_vad,
                    )
                else:
                    print(f"⚠️  Segment {idx} transcription échouée : {e}")
                    continue

            for s in segs:
                seg_audio = audio[int(s.start * SAMPLE_RATE): int(s.end * SAMPLE_RATE)]

                cleaned_text = clean_transcript(s.text)
                if not cleaned_text:
                    continue

                s.text = cleaned_text

                if seg_audio.size > 0 and self._is_hallucination(
                    s.text,
                    seg_audio,
                    avg_logprob=getattr(s, "avg_logprob", None),
                    no_speech_prob=getattr(s, "no_speech_prob", None),
                ):
                    continue

                seg_words = len(s.text.split())
                seg_dur = max(0.01, s.end - s.start)

                if seg_words >= 12 and seg_words / seg_dur > MAX_SEGMENT_WORDS:
                    print(f"[STT] ⚠️ Densité anormale ({seg_words/seg_dur:.1f} mots/s) → ignoré")
                    continue

                all_segments.append(s)
                lang_votes[s.language] = lang_votes.get(s.language, 0) + 1

        if lang_votes:
            detected_lang = max(lang_votes, key=lambda k: lang_votes[k])
        else:
            detected_lang = language or "fr"

        detected_lang = self._normalize_language(detected_lang)

        if detected_lang not in self.SUPPORTED_LANGUAGES:
            print(f"⚠️ Langue finale '{detected_lang}' hors périmètre (votes: {lang_votes})")
        else:
            print(f"🌐 Langue : {detected_lang} (votes: {lang_votes})  |  device: {self.device}")

        all_segments.sort(key=lambda s: s.start)
        return all_segments, detected_lang

    # ── API principale enrichie ───────────────────────────────────────────────

    def get_full_text(self, audio_array: np.ndarray, language: str | None = None) -> dict:
        audio    = self._normalize_audio(audio_array)
        segments, detected_lang = self.transcribe_stream(audio_array, language)

        vision_stress = False
        if self.vision_engine is not None:
            vision_stress = bool(getattr(self.vision_engine, "vision_stress_flag", False))

        silence_events = _detect_silences(audio, segments, sr=SAMPLE_RATE, vision_stress=vision_stress)
        speech_stats   = _compute_speech_stats(segments)

        summary: dict[str, int] = {
            SilenceEvent.END_OF_TURN: 0,
            SilenceEvent.REFLECTION:  0,
            SilenceEvent.STRESSED:    0,
        }
        for ev in silence_events:
            summary[ev.kind] = summary.get(ev.kind, 0) + 1

        has_reflection = summary[SilenceEvent.REFLECTION] > 0 or summary[SilenceEvent.STRESSED] > 0
        has_stress     = summary[SilenceEvent.STRESSED] > 0

        if silence_events:
            icons = {"end_of_turn": "✅", "reflection": "🤔", "stressed_reflection": "😰"}
            parts = [f"{icons.get(ev.kind,'?')}{ev.duration_ms}ms" for ev in silence_events]
            slow  = " | 🐢 débit lent" if speech_stats["slow_speech"] else ""
            print(f"🔇 Silences : {' '.join(parts)}{slow}")

        full_text = join_cleaned_segments([seg.text.strip() for seg in segments])

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
            "silences":        [ev.to_dict() for ev in silence_events],
            "silence_summary": summary,
            "speech_stats":    speech_stats,
            "has_reflection":  has_reflection,
            "has_stress":      has_stress,
        }