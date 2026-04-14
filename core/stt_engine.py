# stt_engine.py — VERSION SANS NOISE REDUCTION (low latency)

import torch
import numpy as np
from faster_whisper import WhisperModel

from device_config import DEVICE_CONFIG, log_device_config


class STTEngine:
    def __init__(self):

        # ⚡ modèle (garde ton choix)
        self.model_size = "large-v3"  # ou "medium" si tu veux plus rapide

        self.model = None
        log_device_config("STT")
        self._load_model()

        self.supported_languages = {"en", "fr", "ar"}

        self.previous_text = ""
        self.candidate_name: str | None = None

    # ─────────────────────────────────────────────
    # 🚀 modèle
    # ─────────────────────────────────────────────
    def _load_model(self):
        device = DEVICE_CONFIG["whisper_device"]

        if device == "cuda":
            self.model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="float16",
                device_index=0,
            )
            self.device = "cuda"
            print("✅ STT GPU ready")
            return

        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8"
        )
        self.device = "cpu"
        print("⚙️ STT CPU ready")

    # ─────────────────────────────────────────────
    # 🎧 NORMALISATION SEULEMENT (NOISE REMOVED)
    # ─────────────────────────────────────────────
    def _normalize_audio(self, audio_array):
        audio = np.array(audio_array, dtype=np.float32)

        # conversion si int16
        if audio.dtype == np.int16 or audio.max() > 1.0:
            audio = audio / 32768.0

        # juste clamp (IMPORTANT)
        return np.clip(audio, -1.0, 1.0)

    # ─────────────────────────────────────────────
    # 🧠 prompt
    # ─────────────────────────────────────────────
    def _build_prompt(self):
        name_hint = (
            f"The candidate's name is {self.candidate_name}. "
            if self.candidate_name else ""
        )

        return (
            "Professional job interview in English, French, and Saudi Arabic (Khaliji dialect). "
            "Transcribe exactly what is spoken without translation. "
            "Keep Arabic dialect as spoken."
            + name_hint
        )

    # ─────────────────────────────────────────────
    # 🧹 nettoyage léger
    # ─────────────────────────────────────────────
    def _clean_text(self, text: str):
        for w in [" euh ", " um ", " uh "]:
            text = text.replace(w, " ")
        return text.strip()

    # ─────────────────────────────────────────────
    # 🚀 transcription
    # ─────────────────────────────────────────────
    def transcribe_stream(self, audio_array, language=None):

        audio = self._normalize_audio(audio_array)

        prompt = self._build_prompt()

        if self.previous_text:
            prompt = self.previous_text[-200:] + " " + prompt

        try:
            segments_gen, info = self.model.transcribe(
                audio,

                language=None,
                task="transcribe",

                # ⚡ balanced quality / speed
                beam_size=5,        # ↓ réduit latence vs 7
                best_of=1,          # ↓ énorme gain perf
                temperature=0.0,

                condition_on_previous_text=True,

                word_timestamps=True,

                vad_filter=True,

                no_speech_threshold=0.7,
                log_prob_threshold=-0.8,
            )

            segments = list(segments_gen)

            full_text = " ".join(s.text.strip() for s in segments)
            full_text = self._clean_text(full_text)

            if full_text:
                self.previous_text = full_text

            return {
                "text": full_text,
                "language": info.language,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text.strip(),
                    }
                    for s in segments
                ]
            }

        except Exception as e:
            raise RuntimeError(f"❌ STT error: {e}")