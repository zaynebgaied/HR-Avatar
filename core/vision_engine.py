"""
vision_engine.py  —  Analyse d'émotion en temps réel (RTX 2050 optimisé)

v3 — corrections de fiabilité :
  - Lissage temporel sur fenêtre glissante N=5 frames (élimine faux-positifs)
  - backend="ssd" : meilleur compromis vitesse/précision vs opencv Haar Cascade
    (fallback automatique vers opencv si ssd non dispo)
  - angry/disgust nécessitent confirmation sur 2 frames consécutives
    car ce sont les émotions les plus souvent faux-positifs sur FER2013 en entretien
  - vision_stress_flag ne bascule que si score lissé > STRESS_ON_THRESHOLD
    ET la même émotion persiste sur au moins STRESS_MIN_FRAMES frames
  - Exposition de stress_score (0.0-1.0) en plus du flag booléen
  - Toutes les frames (valides ET invalides) sont comptabilisées
    pour calculer un taux de détection réel
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

# Émotions considérées comme "stress" pour le flag
_STRESS_EMOTIONS = {"fear", "sad"}          # retiré angry/disgust : trop de faux-positifs
_STRESS_AMBIGUOUS = {"angry", "disgust"}    # acceptés uniquement si score > seuil renforcé

# ── Seuils ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD      = 52.0   # % — seuil minimal par frame (légèrement abaissé car lissage compense)
FACE_CONF_THRESHOLD       = 0.45   # détection visage (ssd est plus précis qu'opencv)
STRESS_ON_THRESHOLD       = 60.0   # % — score lissé pour activer stress
STRESS_OFF_THRESHOLD      = 55.0   # % — score lissé pour désactiver stress (asymétrique intentionnel)
STRESS_AMBIGUOUS_THRESHOLD = 72.0  # % — seuil renforcé pour angry/disgust
STRESS_MIN_FRAMES         = 2      # nb de frames consécutives stress avant de basculer le flag
SMOOTHING_WINDOW          = 5      # fenêtre glissante (frames) pour le lissage
MAX_TIMELINE_ENTRIES      = 500


class VisionEngine:

    def __init__(self):
        self._deepface_available = False
        self._deepface           = None
        self._backend            = "ssd"    # sera mis à jour si fallback nécessaire

        # État interne lissage
        self._emotion_window: deque = deque(maxlen=SMOOTHING_WINDOW)
        self._consecutive_stress    = 0

        # Métriques de détection
        self._total_frames    = 0
        self._detected_frames = 0

        # Flag stress exposé publiquement (lu par STTEngine et brain)
        self.vision_stress_flag   = False
        self.vision_emotion_label = "neutre"
        self.stress_score         = 0.0     # 0.0 → 1.0, score lissé

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

            # Préchargement modèle Emotion
            print("⏳ VisionEngine : préchargement modèle Emotion...")
            DeepFace.build_model("Emotion")
            print("✅ VisionEngine : modèle Emotion préchargé")

            # Test du backend ssd (plus précis qu'opencv pour visages en entretien)
            # ssd utilise MobileNet — détecte mieux les visages légèrement de profil
            # et avec accessoires (lunettes, barbe) contrairement à Haar Cascade.
            self._backend = self._select_best_backend(DeepFace)

            self._deepface           = DeepFace
            self._deepface_available = True
            _mode = "GPU" if DEVICE_CONFIG["use_gpu"] else "CPU"
            print(f"✅ VisionEngine prêt (DeepFace — {_mode} | backend: {self._backend})")

        except ImportError:
            print("⚠️  VisionEngine : DeepFace non installé — vision désactivée.")
            print("   → pip install deepface opencv-python tensorflow")
        except Exception as e:
            print(f"⚠️  VisionEngine : erreur chargement modèle ({e})")

    def _select_best_backend(self, DeepFace) -> str:
        """
        Teste ssd en premier (MobileNet, bon équilibre vitesse/précision).
        Fallback vers opencv si non disponible.
        """
        import numpy as np
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        for backend in ("ssd", "opencv"):
            try:
                DeepFace.analyze(
                    img_path=dummy,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend=backend,
                    silent=True,
                )
                print(f"✅ VisionEngine backend sélectionné : {backend}")
                return backend
            except Exception:
                continue
        print("⚠️  VisionEngine : aucun backend fiable → opencv par défaut")
        return "opencv"

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSE D'UNE FRAME
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_frame(self, jpeg_b64: str) -> dict:
        """
        Analyse une frame JPEG (base64).
        Retourne le résultat brut (non lissé) — le lissage est fait dans _analysis_loop.
        """
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
                "all_scores": {k: round(v, 1) for k, v in emotions.items()},  # nouveau : scores complets
                "timestamp":  time.time(),
                "valid":      True,
            }

        except Exception as e:
            return self._empty_result(f"Erreur analyse : {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # LISSAGE TEMPOREL
    # ─────────────────────────────────────────────────────────────────────────

    def _smooth_and_update_stress(self, result: dict):
        """
        Ajoute le résultat brut à la fenêtre glissante et recalcule
        le stress_score lissé. Met à jour vision_stress_flag uniquement
        si le signal est stable sur STRESS_MIN_FRAMES frames consécutives.
        """
        if not result["valid"]:
            # Frame invalide : on dégrade légèrement le score lissé sans reset brutal
            if self._emotion_window:
                # On injecte un "neutre" faible pour amortir
                self._emotion_window.append({"raw": "neutral", "confidence": 30.0})
            return

        self._emotion_window.append({
            "raw":        result["raw"],
            "confidence": result["confidence"],
        })

        if len(self._emotion_window) < 2:
            return  # pas assez de données pour lissage fiable

        # ── Calcul score stress lissé ────────────────────────────────────────
        # On pondère chaque frame par sa confiance et sa nature stress/non-stress.
        # Les frames récentes comptent davantage (poids linéaire).
        window = list(self._emotion_window)
        n = len(window)
        weighted_stress = 0.0
        total_weight    = 0.0

        for i, frame in enumerate(window):
            weight = (i + 1) / n   # frames récentes = poids plus élevé
            raw    = frame["raw"]
            conf   = frame["confidence"]

            if raw in _STRESS_EMOTIONS:
                stress_contrib = conf
            elif raw in _STRESS_AMBIGUOUS:
                # angry/disgust : contribution réduite de 40% pour limiter faux-positifs
                stress_contrib = conf * 0.60
            else:
                stress_contrib = 0.0

            weighted_stress += weight * stress_contrib
            total_weight    += weight

        smoothed = (weighted_stress / total_weight) if total_weight > 0 else 0.0
        self.stress_score = round(smoothed / 100.0, 3)   # normalisation 0-1

        # ── Mise à jour flag avec hystérésis ────────────────────────────────
        last_raw = result["raw"]

        if last_raw in _STRESS_EMOTIONS and result["confidence"] >= STRESS_ON_THRESHOLD:
            self._consecutive_stress += 1
        elif last_raw in _STRESS_AMBIGUOUS and result["confidence"] >= STRESS_AMBIGUOUS_THRESHOLD:
            self._consecutive_stress += 1
        else:
            self._consecutive_stress = 0   # reset dès qu'une frame non-stress arrive

        # Activation : signal stable ET score lissé élevé
        if self._consecutive_stress >= STRESS_MIN_FRAMES and smoothed >= STRESS_ON_THRESHOLD:
            self.vision_stress_flag   = True
            self.vision_emotion_label = result["emotion"]

        # Désactivation : score lissé retombe (seuil légèrement plus bas = moins de ping-pong)
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
        """Retourne le taux de frames valides (0.0-1.0). Utile pour diagnostic."""
        if self._total_frames == 0:
            return 0.0
        return round(self._detected_frames / self._total_frames, 3)

    def _analysis_loop(self):
        import time as _time
        while getattr(self, "_running", False):
            _time.sleep(self._interval_s)

            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            self._total_frames += 1
            result = self.analyze_frame(frame)

            if result["valid"]:
                self._detected_frames += 1

            # Lissage et mise à jour flag stress (même sur frames invalides)
            self._smooth_and_update_stress(result)

            # Mise à jour brain (stress_score en plus du flag)
            self._brain.vision_stress_flag   = self.vision_stress_flag
            self._brain.vision_emotion_label = self.vision_emotion_label
            if hasattr(self._brain, "vision_stress_score"):
                self._brain.vision_stress_score = self.stress_score

            if not result["valid"]:
                continue

            # Phase courante
            try:
                phase = self._brain.steps[self._brain.current_step_index]
            except Exception:
                phase = "UNKNOWN"

            entry = {
                "timestamp":    _time.strftime("%Y-%m-%d %H:%M:%S"),
                "phase":        phase,
                "emotion":      result["emotion"],
                "raw":          result["raw"],
                "confidence":   result["confidence"],
                "stress_score": self.stress_score,          # nouveau
                "stress_flag":  self.vision_stress_flag,    # nouveau
            }
            self._emotion_timeline.append(entry)
            if len(self._emotion_timeline) > MAX_TIMELINE_ENTRIES:
                self._emotion_timeline = self._emotion_timeline[-MAX_TIMELINE_ENTRIES:]

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