"""
vision_engine.py  —  Analyse d'émotion en temps réel (RTX 2050 optimisé)

v4 — corrections :
  - Suppression de DeepFace.build_model("Emotion") qui appelle le provider
    facial_recognition (non installé) → remplacé par un warm-up via analyze()
  - Lissage temporel sur fenêtre glissante N=5 frames (élimine faux-positifs)
  - backend sélectionné automatiquement : ssd > opencv > skip
  - angry/disgust nécessitent confirmation sur 2 frames consécutives
  - vision_stress_flag ne bascule que si score lissé > STRESS_ON_THRESHOLD
    ET la même émotion persiste sur au moins STRESS_MIN_FRAMES frames
  - Exposition de stress_score (0.0-1.0) en plus du flag booléen
"""

import base64
import time
import threading
import numpy as np
from collections import deque
from device_config import DEVICE_CONFIG

# ── Labels ────────────────────────────────────────────────────────────────────
_EMOTION_LABELS = {
    "angry":    "tendu",
    "disgust":  "tendu",
    "fear":     "anxieux",
    "sad":      "découragé",
    "surprise": "surpris",
    "happy":    "détendu",
    "neutral":  "neutre",
}

_STRESS_EMOTIONS  = {"fear", "sad"}
_STRESS_AMBIGUOUS = {"angry", "disgust"}

# ── Seuils ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD       = 52.0
FACE_CONF_THRESHOLD        = 0.45
STRESS_ON_THRESHOLD        = 60.0
STRESS_OFF_THRESHOLD       = 55.0
STRESS_AMBIGUOUS_THRESHOLD = 72.0
STRESS_MIN_FRAMES          = 2
SMOOTHING_WINDOW           = 5
MAX_TIMELINE_ENTRIES       = 500


class VisionEngine:

    def __init__(self):
        self._deepface_available  = False
        self._deepface            = None
        self._backend             = "opencv"

        self._emotion_window: deque = deque(maxlen=SMOOTHING_WINDOW)
        self._consecutive_stress    = 0

        self._total_frames    = 0
        self._detected_frames = 0

        self.vision_stress_flag   = False
        self.vision_emotion_label = "neutre"
        self.stress_score         = 0.0
        self._latest_public_result = {
            "valid": False,
            "emotion": "neutre",
            "raw": "neutral",
            "confidence": 0.0,
            "stress_score": 0.0,
            "stress_flag": False,
            "timestamp": time.time(),
            "reason": "Aucune analyse encore disponible",
        }

        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self):
        try:
            from deepface import DeepFace

            # Config GPU TensorFlow
            if DEVICE_CONFIG["use_gpu"]:
                try:
                    import tensorflow as tf
                    gpus = tf.config.list_physical_devices("GPU")
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"✅ VisionEngine : TF GPU memory_growth activé ({len(gpus)} GPU)")
                except Exception as e:
                    print(f"⚠️  VisionEngine : config TF GPU ignorée ({e})")
            else:
                import os
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

            # ── Sélection backend AVANT warm-up ──────────────────────────────
            # IMPORTANT : ne pas appeler DeepFace.build_model("Emotion") directement
            # car il tente d'importer le provider facial_recognition (pip séparé).
            # On fait un warm-up via analyze() avec enforce_detection=False,
            # ce qui charge le modèle Emotion sans déclencher le detector manquant.
            print("⏳ VisionEngine : sélection backend et warm-up...")
            self._backend = self._select_best_backend(DeepFace)

            if self._backend is None:
                print("⚠️  VisionEngine : aucun backend disponible — vision désactivée.")
                return

            # Warm-up : charge le modèle Emotion en mémoire via une image noire
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                DeepFace.analyze(
                    img_path          = dummy,
                    actions           = ["emotion"],
                    enforce_detection = False,
                    detector_backend  = self._backend,
                    silent            = True,
                )
                print(f"✅ VisionEngine : modèle Emotion chargé en mémoire (backend={self._backend})")
            except Exception as e:
                # Le warm-up peut échouer sur image noire (pas de visage) sans être bloquant
                print(f"⚠️  VisionEngine : warm-up non bloquant ({e}) — on continue")

            self._deepface           = DeepFace
            self._deepface_available = True
            _mode = "GPU" if DEVICE_CONFIG["use_gpu"] else "CPU"
            print(f"✅ VisionEngine prêt (DeepFace — {_mode} | backend: {self._backend})")

        except ImportError:
            print("⚠️  VisionEngine : DeepFace non installé — vision désactivée.")
            print("   → pip install deepface opencv-python tensorflow")
        except Exception as e:
            print(f"⚠️  VisionEngine : erreur chargement ({e})")

    def _select_best_backend(self, DeepFace) -> str | None:
        """
        Teste les backends dans l'ordre de préférence.
        Retourne le premier qui fonctionne, ou None si aucun ne marche.

        Ordre :
          1. ssd      — MobileNet, bon équilibre vitesse/précision
          2. opencv   — Haar Cascade, fiable et universel
          3. fastmtcnn — rapide, bon en production
          4. yunet    — très rapide, OpenCV 4.8+
        Exclus intentionnellement :
          - facial_recognition : nécessite dlib + cmake, souvent absent
          - retinaface          : lent, inadapté au temps réel sur RTX 2050
          - mediapipe           : instable sur certaines versions TF
        """
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        candidates = ["ssd", "opencv", "fastmtcnn", "yunet"]

        for backend in candidates:
            try:
                DeepFace.analyze(
                    img_path          = dummy,
                    actions           = ["emotion"],
                    enforce_detection = False,
                    detector_backend  = backend,
                    silent            = True,
                )
                print(f"✅ VisionEngine backend sélectionné : {backend}")
                return backend
            except Exception as e:
                print(f"   · backend '{backend}' indisponible : {e}")
                continue

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSE D'UNE FRAME
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_frame(self, jpeg_b64: str) -> dict:
        if not self._deepface_available or self._deepface is None:
            return self._empty_result("DeepFace non disponible")

        try:
            raw_bytes = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)

            import cv2
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return self._empty_result("Image non décodable")

            result = self._deepface.analyze(
                img_path          = img,
                actions           = ["emotion"],
                enforce_detection = False,
                detector_backend  = self._backend,
                silent            = True,
            )

            if isinstance(result, list):
                result = result[0]

            face_conf = float(result.get("face_confidence", 1.0))
            if face_conf < FACE_CONF_THRESHOLD:
                return self._empty_result(
                    f"Visage non détecté (face_confidence={face_conf:.2f})"
                )

            dominant   = result.get("dominant_emotion", "neutral")
            emotions   = result.get("emotion", {})
            confidence = float(emotions.get(dominant, 0.0))

            if confidence < CONFIDENCE_THRESHOLD:
                return self._empty_result(
                    f"Confiance trop basse : {confidence:.1f}% < {CONFIDENCE_THRESHOLD}%"
                )

            label = _EMOTION_LABELS.get(dominant, "neutre")
            return {
                "emotion":    label,
                "raw":        dominant,
                "confidence": round(confidence, 1),
                "all_scores": {k: round(v, 1) for k, v in emotions.items()},
                "timestamp":  time.time(),
                "valid":      True,
            }

        except Exception as e:
            return self._empty_result(f"Erreur analyse : {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # LISSAGE TEMPOREL
    # ─────────────────────────────────────────────────────────────────────────

    def _smooth_and_update_stress(self, result: dict):
        if not result["valid"]:
            if self._emotion_window:
                self._emotion_window.append({"raw": "neutral", "confidence": 30.0})
            return

        self._emotion_window.append({
            "raw":        result["raw"],
            "confidence": result["confidence"],
        })

        if len(self._emotion_window) < 2:
            return

        window = list(self._emotion_window)
        n = len(window)
        weighted_stress = 0.0
        total_weight    = 0.0

        for i, frame in enumerate(window):
            weight = (i + 1) / n
            raw    = frame["raw"]
            conf   = frame["confidence"]

            if raw in _STRESS_EMOTIONS:
                stress_contrib = conf
            elif raw in _STRESS_AMBIGUOUS:
                stress_contrib = conf * 0.60
            else:
                stress_contrib = 0.0

            weighted_stress += weight * stress_contrib
            total_weight    += weight

        smoothed = (weighted_stress / total_weight) if total_weight > 0 else 0.0
        self.stress_score = round(smoothed / 100.0, 3)

        last_raw = result["raw"]

        if last_raw in _STRESS_EMOTIONS and result["confidence"] >= STRESS_ON_THRESHOLD:
            self._consecutive_stress += 1
        elif last_raw in _STRESS_AMBIGUOUS and result["confidence"] >= STRESS_AMBIGUOUS_THRESHOLD:
            self._consecutive_stress += 1
        else:
            self._consecutive_stress = 0

        if self._consecutive_stress >= STRESS_MIN_FRAMES and smoothed >= STRESS_ON_THRESHOLD:
            self.vision_stress_flag   = True
            self.vision_emotion_label = result["emotion"]
        elif smoothed < STRESS_OFF_THRESHOLD and last_raw in ("happy", "neutral"):
            self.vision_stress_flag   = False
            self.vision_emotion_label = result["emotion"]
            self._consecutive_stress  = 0

    # ─────────────────────────────────────────────────────────────────────────
    # THREAD DE FOND
    # ─────────────────────────────────────────────────────────────────────────

    def start_background_analysis(self, brain, interval_s: float = 1.5):
        self._brain            = brain
        self._interval_s       = interval_s
        self._latest_frame     = None
        self._frame_lock       = threading.Lock()
        self._running          = False
        self._emotion_timeline = []

        self._running = True
        t = threading.Thread(target=self._analysis_loop, daemon=True)
        t.start()
        _mode = "GPU" if DEVICE_CONFIG["use_gpu"] else "CPU"
        print(f"✅ VisionEngine thread démarré (intervalle: {interval_s}s | {_mode} | lissage: {SMOOTHING_WINDOW} frames)")

    def stop_background_analysis(self):
        self._running = False

    def push_frame(self, jpeg_b64: str):
        if not hasattr(self, "_frame_lock"):
            return
        with self._frame_lock:
            self._latest_frame = jpeg_b64

    def get_emotion_timeline(self) -> list:
        return list(getattr(self, "_emotion_timeline", []))

    def get_detection_rate(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return round(self._detected_frames / self._total_frames, 3)

    def get_live_snapshot(self) -> dict:
        return dict(getattr(self, "_latest_public_result", {
            "valid": False,
            "emotion": self.vision_emotion_label or "neutre",
            "raw": "neutral",
            "confidence": 0.0,
            "stress_score": self.stress_score,
            "stress_flag": self.vision_stress_flag,
            "timestamp": time.time(),
            "reason": "Aucune analyse encore disponible",
        }))

    def process_frame_now(self, jpeg_b64: str) -> dict:
        self._total_frames += 1
        result = self.analyze_frame(jpeg_b64)
        if result.get("valid"):
            self._detected_frames += 1
        self._commit_result(result)
        return self.get_live_snapshot()

    def _commit_result(self, result: dict):
        self._smooth_and_update_stress(result)

        if hasattr(self, "_brain") and self._brain is not None:
            self._brain.vision_stress_flag   = self.vision_stress_flag
            self._brain.vision_emotion_label = self.vision_emotion_label
            if hasattr(self._brain, "vision_stress_score"):
                self._brain.vision_stress_score = self.stress_score

        snapshot = {
            "valid": bool(result.get("valid")),
            "emotion": result.get("emotion", self.vision_emotion_label or "neutre") if result.get("valid") else (self.vision_emotion_label or "neutre"),
            "raw": result.get("raw", "neutral"),
            "confidence": float(result.get("confidence", 0.0)) if result.get("valid") else 0.0,
            "stress_score": self.stress_score,
            "stress_flag": self.vision_stress_flag,
            "timestamp": result.get("timestamp", time.time()),
            "reason": result.get("reason", ""),
        }
        self._latest_public_result = snapshot

        if not result.get("valid"):
            return

        try:
            phase = self._brain.steps[self._brain.current_step_index]
        except Exception:
            phase = "UNKNOWN"

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "emotion": result["emotion"],
            "raw": result["raw"],
            "confidence": result["confidence"],
            "stress_score": self.stress_score,
            "stress_flag": self.vision_stress_flag,
        }
        self._emotion_timeline.append(entry)
        if len(self._emotion_timeline) > MAX_TIMELINE_ENTRIES:
            self._emotion_timeline = self._emotion_timeline[-MAX_TIMELINE_ENTRIES:]

    def _analysis_loop(self):
        import time as _time
        while getattr(self, "_running", False):
            _time.sleep(self._interval_s)

            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            result = self.analyze_frame(frame)
            if result["valid"]:
                self._detected_frames += 1
            self._commit_result(result)

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(reason: str = "") -> dict:
        return {
            "emotion":    "neutre",
            "raw":        "neutral",
            "confidence": 0.0,
            "all_scores": {},
            "timestamp":  time.time(),
            "valid":      False,
            "reason":     reason,
        }