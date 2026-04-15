# device_config.py  —  RTX 2050 optimisé (4 Go VRAM)
# Ce fichier est importé par : llm_chain.py, stt_engine.py, vision_engine.py, main.py
#
# VRAM budget estimé RTX 2050 (4 Go) :
#   SentenceTransformer  float16  ≈ 0.3 Go
#   Whisper large-v3     float16  ≈ 1.5 Go
#   CrossEncoder                  ≈ 0.0 Go  (désactivé par défaut)
#   Ollama qwen2.5:7b    q4       ≈ 1.5 Go  (géré par Ollama directement)
#   DeepFace Emotion (TF)         ≈ 0.2 Go
#   ─────────────────────────────────────────
#   Total estimé                  ≈ 3.5 Go  ✅ dans les limites
#
import os
import torch


def _detect() -> dict:
    """
    Détecte automatiquement le GPU disponible.
    Override possible via variable d'environnement FORCE_DEVICE=cpu|cuda
    """
    forced = os.getenv("FORCE_DEVICE", "").strip().lower()

    if forced == "cpu":
        print("[DEVICE] ⚠️  FORCE_DEVICE=cpu → mode CPU forcé")
        return _cpu_profile()

    if forced == "cuda":
        print("[DEVICE] ✅ FORCE_DEVICE=cuda → GPU forcé")
        return _gpu_profile(forced=True)

    # ── Détection automatique ──────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("[DEVICE] ⚠️  torch.cuda.is_available() = False → CPU mode")
        print("[DEVICE]     Si tu as un GPU, réinstalle PyTorch CUDA :")
        print("[DEVICE]     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return _cpu_profile()

    return _gpu_profile()


def _gpu_profile(forced: bool = False) -> dict:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    tag      = " (forcé)" if forced else ""
    print(f"[DEVICE] ✅ GPU détecté{tag} : {gpu_name} ({vram_gb} Go VRAM)")

    # RTX 2050 = 4 Go → lite_mode activé pour protéger la VRAM
    lite_mode = vram_gb < 5.0
    if lite_mode:
        print(f"[DEVICE]    → Mode LITE activé (VRAM < 5 Go) : CrossEncoder désactivé pour réduire la latence")

    return {
        # ── Identité GPU ───────────────────────────────────────────────────
        "use_gpu":    True,
        "gpu_name":   gpu_name,
        "vram_gb":    vram_gb,

        # ── SentenceTransformer (RAG embeddings) ──────────────────────────
        "st_model_device": "cuda",
        "torch_dtype":     "float16",   # moitié moins de VRAM
        "st_batch_size":   64,

        # ── CrossEncoder (RAG re-ranking) ─────────────────────────────────
        # Désactivé par défaut sur RTX 2050 pour réduire la latence
        "cross_encoder_device":  "cpu",
        "disable_cross_encoder": True,

        # ── Whisper STT ───────────────────────────────────────────────────
        "whisper_device":       "cuda",
        "whisper_compute_type": "float16",
        "whisper_beam_size": 5,      # beam=1 → latence minimale sur RTX 2050

        # ── Ollama LLM ────────────────────────────────────────────────────
        "ollama_num_gpu": 20,           # charge TOUTES les layers sur GPU
        "ollama_num_ctx": 1536,

        # ── DeepFace / TensorFlow (VisionEngine) ──────────────────────────
        # use_gpu=True → vision_engine.py active tf.config memory_growth
        # (empêche TF de réserver toute la VRAM d'un coup)
    }


def _cpu_profile() -> dict:
    print("[DEVICE] ⚙️  Profil CPU activé (inférence plus lente)")
    return {
        "use_gpu":    False,
        "gpu_name":   "CPU",
        "vram_gb":    0.0,

        "st_model_device": "cpu",
        "torch_dtype":     "float32",
        "st_batch_size":   32,

        "cross_encoder_device":  "cpu",
        "disable_cross_encoder": True,

        "whisper_device":       "cpu",
        "whisper_compute_type": "int8",
        "whisper_beam_size":    5,

        "ollama_num_gpu": 0,
        "ollama_num_ctx": 1536,
    }


# ── Singleton chargé une seule fois à l'import ────────────────────────────────
DEVICE_CONFIG = _detect()


def log_device_config(caller: str = "") -> None:
    """Affiche un résumé lisible de la configuration détectée."""
    tag = f"[{caller}] " if caller else ""
    cfg = DEVICE_CONFIG
    gpu_label = f"{cfg['gpu_name']} {cfg['vram_gb']} Go" if cfg["use_gpu"] else "CPU"
    print(
        f"{tag}Device  : {gpu_label}\n"
        f"{tag}STT     : {cfg['whisper_device']} / {cfg['whisper_compute_type']}\n"
        f"{tag}Embeds  : {cfg['st_model_device']} / {cfg['torch_dtype']}\n"
        f"{tag}Ollama  : num_gpu={cfg['ollama_num_gpu']}  ctx={cfg['ollama_num_ctx']}"
    )