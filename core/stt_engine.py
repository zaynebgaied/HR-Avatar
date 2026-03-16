import torch
import numpy as np
from faster_whisper import WhisperModel


class STTEngine:
    def __init__(self):
        self.model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"
        self.model = None
        self._load_model()

        self.multilingual_prompt = (
            "This is a job interview conversation in English, French, "
            "and Arabic (Saudi dialect - اللهجة السعودية). "
            "The speaker discusses professional experience and skills."
        )

        self.supported_languages = {"en", "fr", "ar"}

    def _load_model(self):
        """
        Tente de charger Whisper sur CUDA d'abord.
        Si cublas64_12.dll ou toute lib CUDA est absente → repli automatique sur CPU (int8).
        """
        cuda_available = torch.cuda.is_available()

        # ── Tentative CUDA ────────────────────────────────────────────────
        if cuda_available:
            print("🚀 GPU détecté ! Tentative de chargement Whisper sur CUDA…")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16",
                    device_index=0,
                )
                self.device       = "cuda"
                self.compute_type = "float16"
                print("✅ STTEngine chargé sur CUDA (float16)")
                return
            except Exception as e:
                err = str(e)
                if "cublas" in err.lower() or "cudnn" in err.lower() or "dll" in err.lower():
                    print(f"⚠️  CUDA indisponible (lib manquante : {err.split(chr(10))[0]})")
                    print("   → Repli automatique sur CPU (int8).")
                else:
                    # Autre erreur CUDA inattendue — on tente quand même le CPU
                    print(f"⚠️  Erreur CUDA inattendue : {e}\n   → Repli sur CPU.")

        # ── Fallback CPU ──────────────────────────────────────────────────
        print("⚙️  Chargement Whisper sur CPU (int8)…")
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )
            self.device       = "cpu"
            self.compute_type = "int8"
            print("✅ STTEngine chargé sur CPU (int8) — transcription plus lente mais fonctionnelle.")
        except Exception as e:
            raise RuntimeError(f"❌ Échec du chargement du modèle Whisper (CPU) : {e}")

    # ── Normalisation audio ───────────────────────────────────────────────
    def _normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalise le tableau audio en float32 entre -1 et 1."""
        if audio_array is None or len(audio_array) == 0:
            raise ValueError("❌ Audio vide ou None reçu.")

        audio = np.array(audio_array, dtype=np.float32)

        if audio.dtype == np.int16 or audio.max() > 1.0:
            audio = audio.astype(np.float32) / 32768.0

        audio = np.clip(audio, -1.0, 1.0)
        return audio

    # ── Transcription principale ──────────────────────────────────────────
    def transcribe_stream(self, audio_array, language: str = None) -> tuple[list, str]:
        """
        Transcrit un segment audio.

        Args:
            audio_array : numpy array audio (float32 ou int16)
            language    : code langue forcé ('fr', 'en', 'ar') ou None pour auto-détection

        Returns:
            (segments, detected_language)
        """
        audio = self._normalize_audio(audio_array)

        if language and language not in self.supported_languages:
            print(f"⚠️ Langue '{language}' non supportée, passage en auto-détection.")
            language = None

        try:
            segments_generator, info = self.model.transcribe(
                audio,
                language=language,
                initial_prompt=self.multilingual_prompt,
                # beam_size=1 pour le temps reel (greedy, -40% latence vs beam_size=2)
                # beam_size=2 si qualite prioritaire sur latence
                beam_size=1,
                best_of=1,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.3,
                    min_speech_duration_ms=100,
                    # 500ms au lieu de 800ms : detecte fin de parole plus vite
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
                temperature=0.0,
                word_timestamps=False,   # desactive pour gagner ~50ms
                condition_on_previous_text=False,  # evite hallucinations inter-chunks
            )

            segments      = list(segments_generator)
            detected_lang = info.language
            confidence    = round(info.language_probability, 3)

            if detected_lang not in self.supported_languages:
                print(f"⚠️ Langue détectée '{detected_lang}' hors périmètre projet (confiance: {confidence})")
            else:
                print(f"🌐 Langue détectée : {detected_lang} (confiance: {confidence})")

            return segments, detected_lang

        except RuntimeError as e:
            err = str(e)
            # Si on était en CUDA et qu'une lib manque en runtime → retry CPU
            if self.device == "cuda" and ("cublas" in err.lower() or "dll" in err.lower()):
                print("⚠️  Erreur CUDA à l'exécution — basculement sur CPU pour cette transcription…")
                self._force_cpu_reload()
                return self.transcribe_stream(audio_array, language)
            raise RuntimeError(f"❌ Erreur lors de la transcription : {e}")

        except Exception as e:
            raise RuntimeError(f"❌ Erreur lors de la transcription : {e}")

    def _force_cpu_reload(self):
        """Recharge le modèle sur CPU si CUDA échoue à l'exécution."""
        try:
            print("⚙️  Rechargement forcé sur CPU…")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )
            self.device       = "cpu"
            self.compute_type = "int8"
            print("✅ STTEngine rebasculé sur CPU (int8).")
        except Exception as e:
            raise RuntimeError(f"❌ Impossible de recharger sur CPU : {e}")

    # ── Raccourci texte complet ───────────────────────────────────────────
    def get_full_text(self, audio_array, language: str = None) -> dict:
        """
        Raccourci pour obtenir le texte complet + métadonnées en un appel.

        Returns:
            dict avec 'text', 'language', 'segments'
        """
        segments, detected_lang = self.transcribe_stream(audio_array, language)
        full_text = " ".join(seg.text.strip() for seg in segments)

        return {
            "text": full_text,
            "language": detected_lang,
            "segments": [
                {
                    "start": seg.start,
                    "end":   seg.end,
                    "text":  seg.text.strip(),
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in (seg.words or [])
                    ]
                }
                for seg in segments
            ]
        }