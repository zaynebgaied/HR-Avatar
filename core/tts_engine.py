import asyncio
import os
import sys
import uuid
import glob
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR      = "temp_audio"
MAX_CACHE_FILES = 20
MAX_CHARS       = 500

# Voix edge-tts par langue.
# Liste complète : `edge-tts --list-voices`
EDGE_VOICES = {
    "Français": "fr-FR-DeniseNeural",   # voix féminine naturelle
    "Anglais":  "en-US-JennyNeural",
    "Arabe":    "ar-SA-ZariyahNeural",
}

# Débit moyen par langue pour estimer la durée (lip-sync)
SPEECH_RATE = {
    "Français": 14.0,
    "Anglais":  13.5,
    "Arabe":    12.0,
}


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================
class TTSEngine:

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Vérification qu'edge-tts est installé
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "edge-tts n'est pas installé. Lance : pip install edge-tts"
            )

        print(f"TTSEngine initialisé — moteur : edge-tts (cloud) | sortie : {self.output_dir}")

    # -------------------------------------------------------------------------
    # STREAMING DIRECT — yield chunks MP3 bruts, aucun fichier écrit
    # -------------------------------------------------------------------------
    async def stream_speech(self, text: str, lang: str):
        """
        Générateur async : yield des bytes MP3 dès qu'edge-tts les produit.
        Le serveur les pipe directement vers le client WebSocket en binaire.
        Latence premier chunk : ~80-150 ms au lieu de ~500 ms pour save().
        """
        import edge_tts
        if not text or not text.strip():
            return
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS].rsplit(" ", 1)[0] + "..."
        voice = EDGE_VOICES.get(lang, EDGE_VOICES["Français"])
        try:
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                # chunk = {"type": "audio", "data": bytes}
                #       | {"type": "WordBoundary", ...}  ← ignoré
                if chunk.get("type") == "audio" and chunk.get("data"):
                    yield chunk["data"]
        except Exception as e:
            print(f"[TTS stream] erreur : {e}")

    # -------------------------------------------------------------------------
    # SYNTHÈSE FICHIER (conservé pour le greeting HTTP au démarrage)
    # -------------------------------------------------------------------------
    async def _synthesize(self, text: str, lang: str, output_path: str) -> bool:
        """
        Génère un fichier MP3 via edge-tts.
        Retourne True si le fichier est non-vide.
        """
        import edge_tts

        voice = EDGE_VOICES.get(lang, EDGE_VOICES["Français"])

        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

            size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            print(f"MP3 généré : {os.path.basename(output_path)} ({size} bytes)")
            return size > 0

        except Exception as e:
            print(f"Erreur synthèse edge-tts : {e}")
            import traceback
            traceback.print_exc()
            return False

    # -------------------------------------------------------------------------
    # GÉNÉRATION AUDIO — API publique (async)
    # -------------------------------------------------------------------------
    async def generate_speech(
        self,
        text:     str,
        lang:     str,
        filename: Optional[str] = None,
    ) -> dict:
        """
        Génère un fichier MP3 via edge-tts.

        Returns:
            dict { path, filename, lang, engine,
                   text_length, estimated_duration_s, success }
        """
        if not text or not text.strip():
            return self._error_result("Texte vide fourni.")

        if len(text) > MAX_CHARS:
            print(f"Texte tronqué ({len(text)} -> {MAX_CHARS} chars)")
            text = text[:MAX_CHARS].rsplit(" ", 1)[0] + "..."

        # edge-tts produit du MP3 nativement — on force l'extension .mp3
        if filename is None:
            filename = f"tts_{lang}_{uuid.uuid4().hex[:8]}.mp3"
        else:
            # Normaliser l'extension vers .mp3
            base = os.path.splitext(filename)[0]
            filename = base + ".mp3"

        output_path = os.path.join(self.output_dir, filename)

        try:
            success = await self._synthesize(text, lang, output_path)

            if not success:
                return self._error_result("edge-tts : synthèse échouée.")

            rate               = SPEECH_RATE.get(lang, 13.0)
            estimated_duration = round(len(text) / rate, 2)

            print(f"[edge-tts/{lang}] {filename} (~{estimated_duration}s | {len(text)} chars)")
            self._cleanup_old_files()

            return {
                "path":                 os.path.abspath(output_path),
                "filename":             filename,
                "lang":                 lang,
                "engine":               "edge-tts",
                "text_length":          len(text),
                "estimated_duration_s": estimated_duration,
                "success":              True,
            }

        except Exception as e:
            return self._error_result(f"generate_speech erreur : {e}")

    # -------------------------------------------------------------------------
    # WRAPPER SYNCHRONE
    # -------------------------------------------------------------------------
    def generate_speech_sync(
        self,
        text:     str,
        lang:     str,
        filename: Optional[str] = None,
    ) -> dict:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.generate_speech(text, lang, filename))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.generate_speech(text, lang, filename))

    # -------------------------------------------------------------------------
    # GÉNÉRATION BATCH
    # -------------------------------------------------------------------------
    async def generate_batch(self, requests: list[dict]) -> list[dict]:
        tasks = [
            self.generate_speech(
                r.get("text", ""),
                r.get("lang", "Français"),
                r.get("filename"),
            )
            for r in requests
        ]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    # -------------------------------------------------------------------------
    # NETTOYAGE CACHE
    # -------------------------------------------------------------------------
    def _cleanup_old_files(self):
        files = sorted(
            glob.glob(os.path.join(self.output_dir, "tts_*.mp3")) +
            glob.glob(os.path.join(self.output_dir, "tts_*.wav")),
            key=os.path.getmtime,
        )
        while len(files) > MAX_CACHE_FILES:
            oldest = files.pop(0)
            try:
                os.remove(oldest)
            except OSError:
                pass

    def clear_all_audio(self):
        for f in (glob.glob(os.path.join(self.output_dir, "*.mp3")) +
                  glob.glob(os.path.join(self.output_dir, "*.wav"))):
            try:
                os.remove(f)
            except OSError:
                pass
        print(f"Tous les fichiers audio supprimés de : {self.output_dir}")

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def _error_result(message: str) -> dict:
        print(f"TTSEngine erreur : {message}")
        return {
            "path":                 None,
            "filename":             None,
            "lang":                 None,
            "engine":               "edge-tts",
            "text_length":          0,
            "estimated_duration_s": 0.0,
            "success":              False,
            "error":                message,
        }

    @staticmethod
    def list_voices() -> dict:
        return {lang: voice for lang, voice in EDGE_VOICES.items()}


# =============================================================================
# TESTS
# =============================================================================
async def main():
    import time

    tts = TTSEngine()

    print("\n" + "=" * 55)
    print("         TEST TTSEngine — edge-tts cloud")
    print("=" * 55)

    test_cases = [
        {"lang": "Français", "text": "Bonjour, pouvez-vous me parler de votre expérience avec FastAPI ?"},
        {"lang": "Anglais",  "text": "Your experience with machine learning is very impressive."},
        {"lang": "Arabe",    "text": "يسعدني التعرف على مشاريعك في مجال الذكاء الاصطناعي."},
    ]

    print("\n--- Test séquentiel ---")
    for case in test_cases:
        t0      = time.time()
        result  = await tts.generate_speech(case["text"], case["lang"])
        elapsed = round((time.time() - t0) * 1000)
        status  = "OK" if result["success"] else "FAIL"
        print(f"  [{status}] {case['lang']:<10} | {elapsed}ms | {result.get('filename', 'N/A')}")

    print("\n--- Test batch (parallèle) ---")
    t0      = time.time()
    results = await tts.generate_batch(test_cases)
    elapsed = round((time.time() - t0) * 1000)
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  [{status}] {r.get('lang','?'):<10} | {r.get('filename','N/A')}")
    print(f"  Batch total : {elapsed}ms")

    print("\nTests terminés.")


if __name__ == "__main__":
    asyncio.run(main())