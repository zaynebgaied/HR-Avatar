import os
import sys
import json
import asyncio
import uuid
import shutil
import time
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from stt_engine          import STTEngine
from llm_chain           import HRInteractiveBrain
from tts_engine          import TTSEngine
from interview_evaluator import InterviewEvaluator
from vision_engine       import VisionEngine
from device_config       import DEVICE_CONFIG, log_device_config

from contextlib import asynccontextmanager
import contextlib
import re

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_device_config("APP")
    # Warm-up automatique de qwen2.5 au démarrage
    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": "qwen2.5:7b", "prompt": "hi", "stream": False}
            )
        print("✅ Ollama warm-up OK (qwen2.5:7b en mémoire GPU)")
    except Exception as e:
        print(f"⚠️  Ollama warm-up échoué : {e} — relance ollama serve")
    print("🚀 App démarrée")
    yield

# =============================================================================
# APP & DOSSIERS
# =============================================================================
app = FastAPI(title="Avatar RH Interactif", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

class MediaSecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["Feature-Policy"]     = "camera *; microphone *"
        response.headers["Permissions-Policy"] = "camera=*, microphone=*, autoplay=*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        return response

app.add_middleware(MediaSecurityHeadersMiddleware)

BASE_DIR         = Path(__file__).parent
STATIC_DIR       = BASE_DIR / "static"
DATA_DIR         = BASE_DIR / "data"
TEMP_DIR         = BASE_DIR / "temp_audio"
COMPANY_INFO_DIR = DATA_DIR / "company_info"
REPORTS_DIR      = DATA_DIR / "reports"
CV_OFFRES_DIR    = DATA_DIR / "cv_offres"

for d in [STATIC_DIR, DATA_DIR, TEMP_DIR, COMPANY_INFO_DIR, REPORTS_DIR, CV_OFFRES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CV_OFFRE_EXTS = {".pdf", ".docx"}
COMPANY_EXTS  = {".pdf", ".docx", ".txt"}

app.mount("/static",     StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/temp_audio", StaticFiles(directory=str(TEMP_DIR)),   name="temp_audio")
app.mount("/data",       StaticFiles(directory=str(DATA_DIR)),   name="data")

# =============================================================================
# INSTANCES GLOBALES
# =============================================================================
stt_engine           = STTEngine()
tts_engine           = TTSEngine(output_dir=str(TEMP_DIR))
vision_engine_global = VisionEngine()
stt_engine.vision_engine = vision_engine_global   # Injection


sessions:        dict[str, HRInteractiveBrain] = {}
sessions_vision: dict[str, VisionEngine]       = {}

WS_EVENT_POLL_S = 0.5
WS_PROGRESS_INTERVAL_S = 1.5
WS_FIRST_TOKEN_SOFT_TIMEOUT_S = 6.0
WS_LLM_HARD_TIMEOUT_S = 75.0


async def _warmup_brain_runtime(brain: HRInteractiveBrain) -> None:
    loop = asyncio.get_running_loop()

    def _do() -> None:
        try:
            brain.ensure_embeddings_ready()
        except Exception as exc:
            print(f"⚠️ Warmup embeddings failed: {exc}")
        try:
            brain._ensure_llm_analysis()
        except Exception as exc:
            print(f"⚠️ Warmup brain analysis failed: {exc}")

    await loop.run_in_executor(None, _do)



@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # SVG favicon inline — no file needed
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="8" fill="#07080d"/>'
        '<text x="16" y="23" text-anchor="middle" font-size="20">🤖</text>'
        '</svg>'
    )
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")

# =============================================================================
# PAGE PRINCIPALE
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    p = STATIC_DIR / "index.html"
    return HTMLResponse(
        content=p.read_text(encoding="utf-8") if p.exists()
        else "<h1>index.html introuvable dans static/</h1>",
        status_code=200,
    )

# =============================================================================
# LIEN CANDIDAT : /interview/{token}
# =============================================================================
USERS_FILE         = DATA_DIR / "users.json"
SESSION_USERS_FILE = DATA_DIR / "session_users.json"

def _load_users_dict() -> dict:
    if not USERS_FILE.exists():
        print(f"[WARN] users.json INTROUVABLE : {USERS_FILE}")
        return {}
    raw = USERS_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        print(f"[WARN] users.json est VIDE")
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[ERROR] users.json corrompu : {e}")
        return {}

def _find_user_by_token(token: str) -> tuple[str, dict] | tuple[None, None]:
    users = _load_users_dict()
    for uname, u in users.items():
        stored = u.get("access_token", "")
        if stored == token:
            return uname, u
    return None, None

@app.get("/interview/{token}", response_class=HTMLResponse)
async def candidate_interview_link(token: str):
    import datetime as _dt

    uname, u = _find_user_by_token(token)
    if not u:
        return HTMLResponse(
            _error_page("❌ Lien invalide ou expiré.", "Contactez votre recruteur."),
            status_code=200,
        )

    scheduled_at = u.get("scheduled_at")
    if scheduled_at:
        try:
            dt_start = _dt.datetime.fromisoformat(scheduled_at).replace(tzinfo=None)
            window   = int(u.get("scheduled_window_min", 30))
            dt_end   = dt_start + _dt.timedelta(minutes=window)
            now      = _dt.datetime.now().replace(tzinfo=None)
            if now < dt_start:
                diff = int((dt_start - now).total_seconds() / 60)
                return HTMLResponse(
                    _error_page(
                        f"⏳ Entretien planifié le {dt_start.strftime('%d/%m/%Y à %H:%M')}",
                        f"Il vous reste {diff} minute(s) avant l'ouverture."
                    ), status_code=200)
            if now > dt_end:
                return HTMLResponse(
                    _error_page(
                        "⛔ Fenêtre d'accès expirée",
                        f"L'entretien était prévu le {dt_start.strftime('%d/%m/%Y à %H:%M')}."
                    ), status_code=200)
        except Exception as e:
            print(f"[INTERVIEW] Erreur parsing scheduled_at : {e} — accès autorisé")

    if u.get("session_status") == "termine":
        return HTMLResponse(
            _error_page("✅ Entretien déjà réalisé", "Merci pour votre participation."),
            status_code=200,
        )

    effective_lang     = u.get("langue", "Français")
    effective_duration = int(u.get("duree", 30))

    brain = HRInteractiveBrain(
        target_lang=effective_lang,
        duration_minutes=effective_duration,
    )
    brain.candidate_username = uname
    brain.candidate_name     = u.get("name", uname)

    # Charger CV
    cv_file = u.get("cv_file")
    if cv_file and (CV_OFFRES_DIR / cv_file).exists():
        cv_ext  = Path(cv_file).suffix
        dest_cv = DATA_DIR / f"cv{cv_ext}"
        for old_ext in CV_OFFRE_EXTS:
            old_p = DATA_DIR / f"cv{old_ext}"
            if old_p.exists() and old_p != dest_cv:
                old_p.unlink()
        shutil.copy2(str(CV_OFFRES_DIR / cv_file), str(dest_cv))
        brain.ingest_document(str(dest_cv), "cv", build_embeddings=False)
    else:
        print(f"[INTERVIEW] ⚠️ CV manquant : {cv_file}")

    # Charger Offre
    offre_file = u.get("offre_file")
    if offre_file and (CV_OFFRES_DIR / offre_file).exists():
        offre_ext  = Path(offre_file).suffix
        dest_offre = DATA_DIR / f"offre{offre_ext}"
        for old_ext in CV_OFFRE_EXTS:
            old_p = DATA_DIR / f"offre{old_ext}"
            if old_p.exists() and old_p != dest_offre:
                old_p.unlink()
        shutil.copy2(str(CV_OFFRES_DIR / offre_file), str(dest_offre))
        brain.ingest_document(str(dest_offre), "job_offer", build_embeddings=False)
    else:
        print(f"[INTERVIEW] ⚠️ Offre manquante : {offre_file}")

    # KB entreprise
    for doc in sorted(COMPANY_INFO_DIR.iterdir()):
        if doc.suffix.lower() in COMPANY_EXTS:
            brain.ingest_document(str(doc), "company_info", build_embeddings=False)

    asyncio.create_task(_warmup_brain_runtime(brain))

    session_id = uuid.uuid4().hex
    sessions[session_id]        = brain
    ve = VisionEngine()
    ve.start_background_analysis(brain, interval_s=1.5)
    sessions_vision[session_id] = ve

    # Mise à jour users.json
    users_data = _load_users_dict()
    if uname in users_data:
        users_data[uname]["session_id"]     = session_id
        users_data[uname]["session_status"] = "en_cours"
        USERS_FILE.write_text(
            json.dumps(users_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    mapping_file = DATA_DIR / "session_users.json"
    mapping = {}
    if mapping_file.exists():
        try:
            mapping = json.loads(mapping_file.read_text())
        except Exception:
            pass
    mapping[session_id] = uname
    mapping_file.write_text(json.dumps(mapping, indent=2))

    greeting_text = brain.get_initial_greeting()
    tts_result = {"success": False, "filename": None, "estimated_duration_s": 0}
    try:
        tts_result = await asyncio.wait_for(
            tts_engine.generate_speech(
                greeting_text, effective_lang,
                filename=f"greeting_{session_id}.wav",
            ),
            timeout=12.0,
        )
    except Exception as e:
        print(f"[INTERVIEW] TTS erreur : {e}")

    possible_paths = [
        STATIC_DIR / "index.html",
        BASE_DIR / "index.html",
        BASE_DIR.parent / "static" / "index.html",
    ]
    html = None
    for p in possible_paths:
        if p.exists():
            html = p.read_text(encoding="utf-8")
            break

    if not html:
        return HTMLResponse(
            _error_page("❌ Erreur serveur", "index.html introuvable."),
            status_code=200,
        )

    poste_safe    = (u.get("poste") or "").replace('"', '\\"')
    greeting_safe = greeting_text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", "")
    audio_url     = f"/temp_audio/{tts_result['filename']}" if tts_result.get("success") else ""
    rag_sources   = {
        "cv":           brain.ingested_docs.get("cv", []),
        "job_offer":    brain.ingested_docs.get("job_offer", []),
        "company_docs": brain.ingested_docs.get("company_info", []),
    }

    autostart_script = (
    "<script>\n"
    "  window.__AUTOSTART__ = {\n"
    f'    enabled:    true,\n'
    f'    token:      "{token}",\n'
    f'    session_id: "{session_id}",\n'
    f'    lang:       "{effective_lang}",\n'
    f'    duration:   {effective_duration},\n'
    f'    poste:      "{poste_safe}",\n'
    f'    username:   "{uname}",\n'
    f'    greeting:   "{greeting_safe}",\n'
    f'    audio_url:  "{audio_url}",\n'
    f'    phase:      "{brain.steps[brain.current_step_index]}",\n'
    f'    time_left:  {brain.get_time_remaining()},\n'
    f'    rag_sources: {json.dumps(rag_sources)}\n'
    "  };\n"
    "</script>"
)
    html = html.replace("<head>", "<head>\n" + autostart_script, 1)
    return HTMLResponse(content=html, status_code=200)

def _error_page(title: str, detail: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Avatar RH</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#07080d;color:#e8eaf0;font-family:'DM Sans',sans-serif;
       min-height:100vh;display:flex;align-items:center;justify-content:center;
       text-align:center;padding:2rem;}}
  h1{{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
      background:linear-gradient(135deg,#fff 30%,#4f8eff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      margin-bottom:.8rem;}}
  p{{color:#6b7280;font-size:.95rem;max-width:380px;line-height:1.6;}}
</style></head>
<body>
  <div><div style="font-size:3rem;margin-bottom:1rem;">{title.split()[0]}</div>
  <h1>{' '.join(title.split()[1:])}</h1>
  <p>{detail}</p></div>
</body></html>"""

# =============================================================================
# API STATUT
# =============================================================================
@app.get("/api/candidates/status")
async def get_candidates_status():
    users  = _load_users_dict()
    result = []
    for uname, u in users.items():
        if u.get("role") == "user":
            result.append({
                "username":       uname,
                "name":           u.get("name", uname),
                "session_status": u.get("session_status", "en_attente"),
                "scheduled_at":   u.get("scheduled_at"),
                "poste":          u.get("poste", ""),
                "report_file":    u.get("report_file"),
            })
    return {"candidates": result}

@app.post("/api/session/{session_id}/status")
async def update_session_status(session_id: str, status: str = Form(...)):
    users_file = DATA_DIR / "users.json"
    if not users_file.exists():
        return JSONResponse({"error": "users.json introuvable"}, status_code=404)
    users_data   = json.loads(users_file.read_text(encoding="utf-8"))
    mapping_file = SESSION_USERS_FILE
    if mapping_file.exists():
        mapping = json.loads(mapping_file.read_text())
        uname   = mapping.get(session_id)
        if uname and uname in users_data:
            users_data[uname]["session_status"] = status
            users_file.write_text(
                json.dumps(users_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return {"ok": True, "username": uname, "status": status}
    return JSONResponse({"error": "session non trouvée"}, status_code=404)

# =============================================================================
# UPLOAD CV / OFFRE
# =============================================================================
@app.post("/upload/cv")
async def upload_cv(file: UploadFile = File(...)):
    return await _save_cv_offre(file, "cv")

@app.post("/upload/offre")
async def upload_offre(file: UploadFile = File(...)):
    return await _save_cv_offre(file, "offre")

async def _save_cv_offre(file: UploadFile, doc_type: str) -> JSONResponse:
    ext = Path(file.filename).suffix.lower()
    if ext not in CV_OFFRE_EXTS:
        return JSONResponse(
            {"error": f"Format '{ext}' non supporté. PDF ou DOCX uniquement."},
            status_code=400,
        )
    content = await file.read()
    dest    = DATA_DIR / f"{doc_type}{ext}"
    dest.write_bytes(content)
    return JSONResponse({
        "status":   "ok",
        "filename": file.filename,
        "type":     doc_type,
        "size_kb":  round(len(content) / 1024, 1),
    })

# =============================================================================
# UPLOAD BASE DE CONNAISSANCE ENTREPRISE
# =============================================================================
@app.post("/upload/company")
async def upload_company(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in COMPANY_EXTS:
            results.append({
                "filename": file.filename,
                "status":   "error",
                "error":    f"Format '{ext}' non supporté. Acceptés : PDF, DOCX, TXT.",
            })
            continue
        content = await file.read()
        dest    = COMPANY_INFO_DIR / file.filename
        dest.write_bytes(content)
        results.append({
            "filename": file.filename,
            "status":   "ok",
            "size_kb":  round(len(content) / 1024, 1),
            "ext":      ext,
            "type":     "company_info",
        })
    return JSONResponse({"uploaded": results, "total": len(results)})

@app.get("/upload/company/list")
async def list_company_docs():
    docs = []
    for f in sorted(COMPANY_INFO_DIR.iterdir()):
        if f.suffix.lower() in COMPANY_EXTS:
            docs.append({
                "filename": f.name,
                "size_kb":  round(f.stat().st_size / 1024, 1),
                "ext":      f.suffix.lower(),
            })
    return {"documents": docs, "count": len(docs)}

@app.delete("/upload/company/{filename}")
async def delete_company_doc(filename: str):
    target = COMPANY_INFO_DIR / filename
    if not target.exists():
        return JSONResponse({"error": "Fichier introuvable."}, status_code=404)
    if not target.resolve().is_relative_to(COMPANY_INFO_DIR.resolve()):
        return JSONResponse({"error": "Accès refusé."}, status_code=403)
    target.unlink()
    return {"status": "ok", "deleted": filename}

# =============================================================================
# SESSION : DÉMARRAGE
# =============================================================================
@app.post("/session/start")
async def start_session(
    lang:     str = Form(...),
    duration: int = Form(20),
    username: str = Form(default=""),
    token:    str = Form(default=""),
):
    session_id = uuid.uuid4().hex

    effective_lang     = lang
    effective_duration = duration

    brain = HRInteractiveBrain(target_lang=lang, duration_minutes=duration)
    brain.candidate_username = username.strip() if username else None

    if token:
        users_file = DATA_DIR / "users.json"
        if users_file.exists():
            import json as _json
            users_data = _json.loads(users_file.read_text(encoding="utf-8"))
            for uname, u in users_data.items():
                if u.get("access_token") == token and u.get("role") == "user":
                    brain.candidate_username = uname
                    brain.candidate_name     = u.get("name", uname)
                    effective_lang     = u.get("langue", lang)
                    effective_duration = int(u.get("duree", duration))
                    brain.target_lang      = effective_lang
                    brain.duration_minutes = effective_duration

                    cv_file = u.get("cv_file")
                    if cv_file and (CV_OFFRES_DIR / cv_file).exists():
                        cv_ext  = Path(cv_file).suffix
                        dest_cv = DATA_DIR / f"cv{cv_ext}"
                        for old_ext in CV_OFFRE_EXTS:
                            old = DATA_DIR / f"cv{old_ext}"
                            if old.exists() and old != dest_cv:
                                old.unlink()
                        shutil.copy2(str(CV_OFFRES_DIR / cv_file), str(dest_cv))

                    offre_file = u.get("offre_file")
                    if offre_file and (CV_OFFRES_DIR / offre_file).exists():
                        offre_ext  = Path(offre_file).suffix
                        dest_offre = DATA_DIR / f"offre{offre_ext}"
                        for old_ext in CV_OFFRE_EXTS:
                            old = DATA_DIR / f"offre{old_ext}"
                            if old.exists() and old != dest_offre:
                                old.unlink()
                        shutil.copy2(str(CV_OFFRES_DIR / offre_file), str(dest_offre))

                    u["session_status"] = "en_cours"
                    users_data[uname]   = u
                    users_file.write_text(
                        _json.dumps(users_data, indent=2, ensure_ascii=False)
                    )
                    break

    for ext in CV_OFFRE_EXTS:
        p = DATA_DIR / f"cv{ext}"
        if p.exists():
            brain.ingest_document(str(p), "cv", build_embeddings=False)
            break

    for ext in CV_OFFRE_EXTS:
        p = DATA_DIR / f"offre{ext}"
        if p.exists():
            brain.ingest_document(str(p), "job_offer", build_embeddings=False)
            break

    for doc in sorted(COMPANY_INFO_DIR.iterdir()):
        if doc.suffix.lower() in COMPANY_EXTS:
            brain.ingest_document(str(doc), "company_info", build_embeddings=False)

    asyncio.create_task(_warmup_brain_runtime(brain))

    sessions[session_id] = brain

    mapping_file = DATA_DIR / "session_users.json"
    if mapping_file.exists():
        try:
            mapping = json.loads(mapping_file.read_text())
            uname   = mapping.get(session_id, "")
            if uname:
                brain.candidate_username = uname
                users_file = DATA_DIR / "users.json"
                if users_file.exists():
                    users_data = json.loads(users_file.read_text())
                    u          = users_data.get(uname, {})
                    brain.candidate_name = u.get("name", uname)
                    if users_data[uname].get("session_status") != "en_cours":
                        users_data[uname]["session_status"] = "en_cours"
                        users_file.write_text(
                            json.dumps(users_data, indent=2, ensure_ascii=False)
                        )
        except Exception as e:
            print(f"⚠️  Liaison session→user : {e}")

    ve = VisionEngine()
    ve.start_background_analysis(brain, interval_s=1.5)
    sessions_vision[session_id] = ve

    greeting_text = brain.get_initial_greeting()
    tts_result = {"success": False, "filename": None, "estimated_duration_s": 0}
    try:
        tts_result = await asyncio.wait_for(
            tts_engine.generate_speech(
                greeting_text,
                brain.target_lang,
                filename=f"greeting_{session_id}.wav",
            ),
            timeout=12.0,
        )
    except Exception as e:
        print(f"⚠️  TTS greeting erreur ({brain.target_lang}): {e}")

    return {
        "session_id":  session_id,
        "greeting":    greeting_text,
        "audio_url":   f"/temp_audio/{tts_result['filename']}" if tts_result.get("success") else None,
        "duration_s":  tts_result.get("estimated_duration_s", 0),
        "phase":       brain.steps[brain.current_step_index],
        "time_left":   brain.get_time_remaining(),
        "langue":      brain.target_lang,
        "duree_minutes": brain.duration_minutes,
        "rag_sources": {
            "cv":           brain.ingested_docs.get("cv",           []),
            "job_offer":    brain.ingested_docs.get("job_offer",    []),
            "company_docs": brain.ingested_docs.get("company_info", []),
        },
    }

# =============================================================================
# HELPER : RAPPORT FINAL
# =============================================================================
def _build_live_candidate_assessment(session_id: str, brain: HRInteractiveBrain) -> dict:
    ve = sessions_vision.get(session_id)
    live_vision = ve.get_live_snapshot() if ve else {
        "valid": False,
        "emotion": getattr(brain, "vision_emotion_label", "neutre") or "neutre",
        "confidence": 0.0,
        "stress_score": float(getattr(brain, "vision_stress_score", 0.0) or 0.0),
        "stress_flag": bool(getattr(brain, "vision_stress_flag", False)),
    }

    last_candidate_turn = None
    for turn in reversed(getattr(brain, "turns", [])):
        speaker = getattr(turn, "speaker", "") or ""
        if "candidate" in speaker.lower() or "candidat" in speaker.lower():
            last_candidate_turn = turn
            break

    return {
        "emotion": live_vision.get("emotion", "neutre"),
        "emotion_confidence": live_vision.get("confidence", 0.0),
        "stress_score": live_vision.get("stress_score", 0.0),
        "stress_flag": live_vision.get("stress_flag", False),
        "answer_quality": getattr(last_candidate_turn, "answer_quality", "N/A") if last_candidate_turn else "N/A",
        "behavioral_story_detected": bool(getattr(last_candidate_turn, "behavioral_story_detected", False)) if last_candidate_turn else False,
        "decision_reasoning_detected": bool(getattr(last_candidate_turn, "decision_reasoning_detected", False)) if last_candidate_turn else False,
        "weak_signals": list(getattr(last_candidate_turn, "weak_signals_detected", []) or []) if last_candidate_turn else [],
        "phase": getattr(last_candidate_turn, "phase", brain.steps[brain.current_step_index]) if last_candidate_turn else brain.steps[brain.current_step_index],
        "timestamp": getattr(last_candidate_turn, "timestamp", None),
        "text_preview": (getattr(last_candidate_turn, "text", "") or "")[:180] if last_candidate_turn else "",
    }

def _build_vision_analysis(session_id: str) -> dict:
    ve = sessions_vision.get(session_id)
    if not ve:
        return {"disponible": False}

    timeline = ve.get_emotion_timeline()
    if not timeline:
        return {
            "disponible": True, "timeline": [],
            "emotion_dominante": "neutre", "pics_stress": [], "evolution": [],
        }

    from collections import Counter
    counts    = Counter(e["emotion"] for e in timeline)
    dominante = counts.most_common(1)[0][0]

    stress_emotions = {"tendu", "anxieux", "découragé"}
    pics = [
        f"{e['phase']} @ {e['timestamp']}"
        for e in timeline
        if e["emotion"] in stress_emotions and e["confidence"] >= 65.0
    ]

    evolution = []
    prev = None
    for e in timeline:
        if e["emotion"] != prev:
            evolution.append(e["emotion"])
            prev = e["emotion"]

    conf_avg = round(sum(e["confidence"] for e in timeline) / len(timeline), 1)

    return {
        "disponible":        True,
        "emotion_dominante": dominante,
        "pics_stress":       pics,
        "confiance_globale": conf_avg,
        "evolution":         " → ".join(evolution),
        "nb_frames":         len(timeline),
        "timeline":          timeline,
    }


async def _build_full_report(
    brain: HRInteractiveBrain,
    inline_report: dict | None,
    session_id: str = "",
) -> dict:
    import datetime as _dt

    base = inline_report or {}
    vision_analysis = _build_vision_analysis(session_id)

    if not base and hasattr(brain, "_build_inline_report"):
        try:
            base = brain._build_inline_report() or {}
        except Exception as e:
            print(f"⚠️  Impossible de générer le rapport inline : {e}")
            base = {}

    nlu = {}
    try:
        evaluator = InterviewEvaluator()
        evaluator.set_job_context(getattr(brain, "job_offer_text", "") or "")
        evaluator.set_vision_data(vision_analysis)
        log_file = getattr(brain, "log_file", None)
        if log_file and Path(log_file).exists():
            nlu = evaluator.evaluate_file(str(log_file))
        else:
            nlu = evaluator.evaluate_latest()[0]
        print("✅ NLU Evaluator : rapport généré.")
    except Exception as e:
        print(f"⚠️ NLU Evaluator indisponible : {e}")
        nlu = {}

    score_total = nlu.get("score_total", base.get("score_total", 0))
    score_global = nlu.get("score_total", nlu.get("score_global", base.get("pourcentage", score_total)))
    score_technique = nlu.get("score_technical", nlu.get("score_technique", base.get("pourcentage", 0)))
    score_behavioral = nlu.get("score_behavioral", base.get("pourcentage", 0))
    score_communication = nlu.get("score_communication", 0)

    scores_by_phase_nlu = nlu.get("scores_by_phase", {})
    scores_par_phase = {}
    for key, val in scores_by_phase_nlu.items():
        scores_par_phase[key] = {
            "obtenu": val.get("obtained", 0),
            "max": val.get("max", 0),
            "avg_turn": val.get("avg_turn", 0),
            "n_turns": val.get("n_turns", 0),
            "status": val.get("status", ""),
            "commentaire": val.get("comment", ""),
        }

    emotion_analysis = nlu.get("emotion_analysis", {})
    if emotion_analysis and emotion_analysis.get("available"):
        vision_analysis = {
            "disponible": True,
            "emotion_dominante": emotion_analysis.get("dominant_fr") or emotion_analysis.get("dominant_emotion") or vision_analysis.get("emotion_dominante", "neutre"),
            "pics_stress": [
                f"{p.get('phase','?')} @ {p.get('timestamp','?')}"
                for p in emotion_analysis.get("stress_peaks", [])
            ] or vision_analysis.get("pics_stress", []),
            "confiance_globale": vision_analysis.get("confiance_globale"),
            "evolution": " → ".join(emotion_analysis.get("emotional_evolution", [])) if emotion_analysis.get("emotional_evolution") else vision_analysis.get("evolution", ""),
            "nb_frames": emotion_analysis.get("n_frames", vision_analysis.get("nb_frames", 0)),
            "timeline": vision_analysis.get("timeline", []),
            "etat_global": emotion_analysis.get("overall_emotional_state", ""),
            "correlation_reponses": emotion_analysis.get("emotion_answer_correlation", ""),
        }

    merged = {
        "username": getattr(brain, "candidate_username", None),
        "date": base.get("date", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
        "langue": base.get("langue", brain.target_lang),
        "duree_minutes": base.get("duree_minutes", brain.duration_minutes),
        "meta": nlu.get("meta", {
            "date_entretien": base.get("date", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
            "langue": base.get("langue", brain.target_lang),
            "duree": f"{base.get('duree_minutes', brain.duration_minutes)} min",
        }),
        "phases": base.get("phases", []),
        "score_total": score_total,
        "score_max": base.get("score_max", 100),
        "pourcentage": base.get("pourcentage", score_global),
        "score_global": score_global,
        "score_technique": score_technique,
        "score_behavioral": score_behavioral,
        "score_communication": score_communication,
        "scores_by_phase": scores_by_phase_nlu,
        "scores_par_phase": scores_par_phase,
        "coverage_rate_pct": nlu.get("coverage_rate_pct"),
        "is_partial_evaluation": nlu.get("is_partial_evaluation", False),
        "competences_detectees": nlu.get("competencies_detected", nlu.get("competences_detectees", [])),
        "lacunes_identifiees": nlu.get("gaps_identified", nlu.get("lacunes_identifiees", [])),
        "analyse_motivation": nlu.get("motivation_analysis", nlu.get("analyse_motivation", "")),
        "analyse_soft_skills": nlu.get("soft_skills_analysis", nlu.get("analyse_soft_skills", "")),
        "points_forts": nlu.get("strengths", nlu.get("points_forts", base.get("points_forts", []))),
        "points_amelioration": nlu.get("improvement_areas", nlu.get("points_amelioration", base.get("points_faibles", []))),
        "verdict": nlu.get("verdict", ""),
        "verdict_final": nlu.get("verdict", nlu.get("verdict_final", base.get("recommandation", ""))),
        "recommandation_detail": nlu.get("recommendation_detail", nlu.get("recommandation_detail", base.get("recommandation", ""))),
        "recommandation": base.get("recommandation", nlu.get("recommendation_detail", "")),
        "sources_rag": base.get("sources_rag", brain.ingested_docs),
        "analyse_emotion_vision": vision_analysis,
        "emotion_analysis": emotion_analysis,
        "evaluation_technique_points": base.get("evaluation_technique_points", []),
        "evaluation_communication_points": base.get("evaluation_communication_points", []),
        "points_detectes": base.get("points_detectes", []),
        "points_manquants": base.get("points_manquants", []),
        "answer_by_answer": nlu.get("answer_by_answer", []),
        "answer_status_distribution": nlu.get("answer_status_distribution", {}),
        "turns_summary": nlu.get("turns_summary", {}),
        "raw_nlu_report": nlu,
    }

    try:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        username_tag = f"_{merged.get('username','')}" if merged.get("username") else ""
        path = DATA_DIR / "reports" / f"full_report_{ts}{username_tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        merged["report_file"] = path.name
        merged["report_path"] = str(path)
        print(f"📄 Rapport complet sauvegardé : {path}")

        if merged.get("username"):
            try:
                import json as _json
                users_file = DATA_DIR / "users.json"
                if users_file.exists():
                    users_data = _json.loads(users_file.read_text(encoding="utf-8"))
                    uname = merged["username"]
                    if uname in users_data:
                        users_data[uname]["session_status"] = "termine"
                        users_data[uname]["report_file"] = path.name
                        users_file.write_text(_json.dumps(users_data, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"✅ users.json mis à jour pour {uname} → terminé")
            except Exception as e:
                print(f"⚠️ Mise à jour users.json échouée : {e}")
    except Exception as e:
        print(f"⚠️ Sauvegarde rapport échouée : {e}")

    return merged

# =============================================================================
# SESSION : RÉPONSE CANDIDAT (HTTP fallback)
# =============================================================================
@app.post("/session/{session_id}/respond")
async def respond(session_id: str, user_text: str = Form(...)):
    try:
        brain = sessions.get(session_id)
        if not brain:
            return JSONResponse({"error": "Session introuvable."}, status_code=404)

        result     = await brain.generate_response_async(user_text)
        uid        = uuid.uuid4().hex[:8]
        tts_result = await tts_engine.generate_speech(
            result["text"], brain.target_lang,
            filename=f"resp_{session_id}_{uid}.wav",
        )

        final_report = None
        if result.get("interview_ended"):
            final_report = await _build_full_report(brain, result.get("report"), session_id)

        return {
            "text":            result["text"],
            "sentiment":       result.get("candidate_sentiment", "neutre"),
            "audio_url":       f"/temp_audio/{tts_result['filename']}" if tts_result["success"] else None,
            "duration_s":      tts_result["estimated_duration_s"],
            "phase":           result.get("phase", brain.steps[brain.current_step_index]),
            "time_left":       result.get("time_left", brain.get_time_remaining()),
            "interview_ended": result.get("interview_ended", False),
            "candidate_assessment": _build_live_candidate_assessment(session_id, brain),
            "report":          final_report,
        }
    except Exception as e:
        import traceback
        print(f"❌ /respond error: {e}")
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# SESSION : TRANSCRIPTION AUDIO → STT (HTTP fallback)
# =============================================================================
@app.post("/session/{session_id}/transcribe")
async def transcribe_audio(session_id: str, audio: UploadFile = File(...)):
    if not sessions.get(session_id):
        return JSONResponse({"error": "Session introuvable."}, status_code=404)

    raw_bytes = await audio.read()
    tmp_webm  = TEMP_DIR / f"audio_{session_id}_{uuid.uuid4().hex[:6]}.webm"
    tmp_wav   = tmp_webm.with_suffix(".wav")
    tmp_webm.write_bytes(raw_bytes)

    try:
        import subprocess as _sp
        _sp.run(
            ["ffmpeg", "-y", "-i", str(tmp_webm),
             "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp_wav)],
            check=True, capture_output=True,
        )
        import wave, numpy as np
        with wave.open(str(tmp_wav), "rb") as wf:
            frames   = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        segments, lang = stt_engine.transcribe_stream(audio_np)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {"text": text, "language": lang, "success": bool(text)}
    except Exception as e:
        import traceback
        print(f"❌ Transcription error : {e}")
        print(traceback.format_exc())
        return JSONResponse({"error": str(e), "success": False}, status_code=500)
    finally:
        for p in (tmp_webm, tmp_wav):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

# =============================================================================
# SESSION : ÉVALUATION NLU FINALE
# =============================================================================
@app.get("/session/{session_id}/evaluate")
async def evaluate_session(session_id: str):
    brain = sessions.get(session_id)
    if not brain:
        return JSONResponse({"error": "Session introuvable."}, status_code=404)
    try:
        report = await _build_full_report(brain, None, session_id)
        return {
            "status": "ok",
            "report": report,
            "report_path": report.get("report_path"),
            "report_file": report.get("report_file"),
        }
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# SESSION : STATUT
# =============================================================================
@app.get("/session/{session_id}/status")
async def session_status(session_id: str):
    brain = sessions.get(session_id)
    if not brain:
        return JSONResponse({"error": "Session introuvable."}, status_code=404)
    return {
        "phase":     brain.steps[brain.current_step_index],
        "time_left": brain.get_time_remaining(),
        "scores":    brain.scores,
        "vision":    _build_live_candidate_assessment(session_id, brain),
        "rag_sources": {
            "cv":           brain.ingested_docs.get("cv",           []),
            "job_offer":    brain.ingested_docs.get("job_offer",    []),
            "company_docs": brain.ingested_docs.get("company_info", []),
        },
    }

# =============================================================================
# SESSION : FERMETURE
# =============================================================================
@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    ve = sessions_vision.pop(session_id, None)
    if ve:
        ve.stop_background_analysis()
    sessions.pop(session_id, None)
    return {"status": "session fermée"}

# =============================================================================
# AUDIO SERVING
# =============================================================================
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = TEMP_DIR / filename
    if path.exists():
        media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(str(path), media_type=media_type)
    return JSONResponse({"error": "Fichier audio introuvable."}, status_code=404)

# =============================================================================
# WEBSOCKET VISION
# =============================================================================
@app.websocket("/ws/vision/{session_id}")
async def vision_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    ve = sessions_vision.get(session_id)
    if not ve:
        await websocket.send_json({"error": "Session vision introuvable.", "valid": False})
        await websocket.close()
        return

    print(f"📷 Vision WebSocket ouvert : {session_id}")
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=10.0)
            except asyncio.TimeoutError:
                continue
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text", "")
            if text == "stop":
                break
            if not text:
                continue
            try:
                data      = json.loads(text)
                frame_b64 = data.get("frame", "")
            except Exception:
                continue
            if not frame_b64:
                continue
            snapshot = ve.process_frame_now(frame_b64)
            await websocket.send_json(snapshot)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Vision WebSocket erreur : {e}")
    finally:
        print(f"📷 Vision WebSocket fermé : {session_id}")

# =============================================================================
# WEBSOCKET STT — streaming PCM Float32 depuis Web Audio API
# Architecture simplifiée : tout l'audio est bufferisé jusqu'au signal "end",
# puis transcrit en une seule passe Whisper → pas de troncature, pas de race condition.
#
# Protocole CLIENT → SERVEUR :
#   bytes Float32Array (PCM 16kHz mono)  → accumulation dans audio_buffer
#   texte "end"                          → transcription + envoi résultat final
#   texte "cancel"                       → vider le buffer sans transcrire
#   JSON {"action":"set_lang","lang":"fr"} → forcer la langue
#
# Protocole SERVEUR → CLIENT :
#   {"text":"...", "language":"fr", "success":true,  "final":true}
#   {"text":"",   "language":"fr", "success":false, "error":"..."}
# =============================================================================
@app.websocket("/ws/stt/{session_id}")
async def stt_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if not sessions.get(session_id):
        await websocket.send_json({"error": "Session introuvable.", "success": False})
        await websocket.close()
        return

    import numpy as np

    audio_buffer = np.array([], dtype=np.float32)
    session_lang: str | None = None

    LANG_MAP = {
        "Français": "fr", "Anglais": "en", "Arabe": "ar",
        "French":   "fr", "English": "en", "Arabic": "ar",
        "fr": "fr", "en": "en", "ar": "ar",
    }

    print(f"STT WebSocket ouvert : {session_id}")

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"error": "Timeout STT.", "success": False})
                audio_buffer = np.array([], dtype=np.float32)
                continue

            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                break

            # ── Chunk PCM binaire → accumulation ─────────────────────────────
            if msg_type == "websocket.receive" and message.get("bytes"):
                chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                audio_buffer = np.concatenate([audio_buffer, chunk])

            elif msg_type == "websocket.receive" and message.get("text"):
                raw_text = message["text"]

                # ── Commandes JSON ────────────────────────────────────────────
                if raw_text.startswith("{"):
                    try:
                        cmd = json.loads(raw_text)
                        if cmd.get("action") == "set_lang":
                            received = cmd.get("lang", "")
                            mapped   = LANG_MAP.get(received)
                            if mapped:
                                session_lang = mapped
                                print(f"[STT] Langue fixée : {session_lang}")
                            else:
                                print(f"[STT] Langue inconnue '{received}' → auto-détection")
                    except Exception:
                        pass
                    continue

                # ── Fin de parole → transcription complète ────────────────────
                elif raw_text == "end":
                    # APRÈS — ajouter final: True pour que le JS sache que c'est terminé
                    # mais success: False pour qu'il ne tente pas d'envoyer au LLM
                    if len(audio_buffer) < 1600:
                        await websocket.send_json({
                            "text": "",
                            "display_text": "[silence]",
                            "is_silence": True,
                            "language": session_lang or "fr",
                            "success": False,
                            "error": "Audio trop court",
                            "final": True,
                        })
                        audio_buffer = np.array([], dtype=np.float32)
                        continue

                    buf_copy = audio_buffer.copy()
                    audio_buffer = np.array([], dtype=np.float32)   # vider immédiatement

                    try:
                        loop  = asyncio.get_event_loop()
                        brain = sessions.get(session_id)

                        # Résolution de la langue depuis la session si pas encore fixée
                        lang_hint = session_lang
                        if not lang_hint and brain:
                            _lang_map = {"Français": "fr", "Anglais": "en", "Arabe": "ar"}
                            lang_hint = _lang_map.get(brain.target_lang)

                        def _do_transcribe():
                            result = stt_engine.get_full_text(buf_copy, language=lang_hint)
                            return result["text"].strip(), result["language"]

                        text, detected_lang = await loop.run_in_executor(None, _do_transcribe)

                        print(f"[STT] ✅ Final : {text[:80]}{'…' if len(text)>80 else ''}")
                        await websocket.send_json({
                            "text": text,
                            "display_text": text if text else "[silence]",
                            "is_silence": not bool(text),
                            "language": detected_lang,
                            "success": bool(text),
                            "final": True,
                        })

                    except Exception as e:
                        print(f"[STT] ❌ Transcription error : {e}")
                        await websocket.send_json({
                            "text": "",
                            "display_text": "[silence]",
                            "is_silence": True,
                            "language": session_lang or "fr",
                            "success": False,
                            "error": str(e),
                            "final": True,
                        })

                # ── Annulation ────────────────────────────────────────────────
                elif raw_text == "cancel":
                    audio_buffer = np.array([], dtype=np.float32)

    except Exception as e:
        print(f"STT WebSocket erreur : {e}")
    finally:
        print(f"STT WebSocket fermé : {session_id}")

# =============================================================================
# WEBSOCKET LLM (streaming)
# =============================================================================

async def _ws_drain_control_messages(websocket: WebSocket, inbox: asyncio.Queue, brain: HRInteractiveBrain) -> bool:
    interrupted = False
    buffered = []
    while True:
        try:
            item = inbox.get_nowait()
        except asyncio.QueueEmpty:
            break

        action = item.get("action")
        if action == "interrupt":
            interrupted = True
        elif action == "ping":
            await websocket.send_json({
                "type": "pong",
                "time_left": brain.get_time_remaining(),
                "phase": brain.steps[brain.current_step_index],
            })
        else:
            buffered.append(item)

    for item in buffered:
        await inbox.put(item)
    return interrupted


async def _ws_send_progress(websocket: WebSocket, brain: HRInteractiveBrain, stage: str, detail: str) -> None:
    await websocket.send_json({
        "type": "progress",
        "stage": stage,
        "detail": detail,
        "phase": brain.steps[brain.current_step_index],
        "time_left": brain.get_time_remaining(),
    })


async def _ws_stream_brain_response(websocket: WebSocket, inbox: asyncio.Queue, brain: HRInteractiveBrain, user_text: str):
    event_q: asyncio.Queue = asyncio.Queue()

    async def _producer():
        try:
            async for event in brain.generate_response_stream(user_text):
                await event_q.put(("event", event))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await event_q.put(("error", exc))
        finally:
            await event_q.put(("eof", None))

    producer = asyncio.create_task(_producer())

    speech_text = ""
    meta_event = {}
    interview_ended = False
    sentence_idx = 0
    first_token_seen = False
    started_at = time.monotonic()
    last_activity = started_at
    last_progress = 0.0

    try:
        while True:
            if await _ws_drain_control_messages(websocket, inbox, brain):
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer
                await websocket.send_json({"type": "interrupted"})
                return {
                    "speech_text": speech_text,
                    "meta_event": meta_event,
                    "interview_ended": False,
                    "sentence_idx": sentence_idx,
                    "interrupted": True,
                }

            try:
                kind, payload = await asyncio.wait_for(event_q.get(), timeout=WS_EVENT_POLL_S)
            except asyncio.TimeoutError:
                now = time.monotonic()
                if now - started_at >= WS_LLM_HARD_TIMEOUT_S:
                    producer.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await producer

                    phase = brain.steps[brain.current_step_index]
                    fallback_text = brain._fallback_question(phase=phase, avoid_repeat=True)
                    try:
                        brain._append_turn(
                            phase,
                            brain._recruiter_speaker_label(),
                            fallback_text,
                            emotion=brain.vision_emotion_label if brain.vision_stress_flag else "neutre",
                        )
                    except Exception:
                        pass

                    await _ws_send_progress(
                        websocket,
                        brain,
                        "timeout_fallback",
                        "Le modèle a pris trop de temps. Relance de secours envoyée.",
                    )
                    for tok in (re.findall(r"\S+\s*", fallback_text) or [fallback_text]):
                        await websocket.send_json({"type": "token", "token": tok})
                    await websocket.send_json({"type": "tts_start", "index": 0, "text": fallback_text})
                    try:
                        async for chunk in tts_engine.stream_speech(fallback_text, brain.target_lang):
                            await websocket.send_bytes(chunk)
                    except Exception as exc:
                        print(f"[TTS stream timeout fallback] erreur : {exc}")
                    await websocket.send_json({"type": "tts_end", "index": 0})
                    meta_event = {
                        "candidate_sentiment": "neutre",
                        "phase": brain.steps[brain.current_step_index],
                        "time_left": brain.get_time_remaining(),
                        "interview_ended": False,
                    }
                    return {
                        "speech_text": fallback_text,
                        "meta_event": meta_event,
                        "interview_ended": False,
                        "sentence_idx": 1,
                        "interrupted": False,
                    }

                if now - last_progress >= WS_PROGRESS_INTERVAL_S:
                    stage = "llm_waiting" if not first_token_seen else "streaming_keepalive"
                    detail = (
                        "Le recruteur prépare sa relance…"
                        if not first_token_seen
                        else "Réponse en cours de génération…"
                    )
                    await _ws_send_progress(websocket, brain, stage, detail)
                    last_progress = now

                if (not first_token_seen) and (now - started_at >= WS_FIRST_TOKEN_SOFT_TIMEOUT_S) and (now - last_progress >= 0.8):
                    await _ws_send_progress(
                        websocket,
                        brain,
                        "first_token_delayed",
                        "Le modèle répond lentement, mais la session reste active.",
                    )
                    last_progress = now
                continue

            if kind == "error":
                raise payload

            if kind == "eof":
                break

            event = payload
            last_activity = time.monotonic()

            if event["type"] == "token":
                first_token_seen = True
                await websocket.send_json({"type": "token", "token": event["token"]})

            elif event["type"] == "sentence":
                phrase = event["text"].strip()
                idx = event["index"]
                if phrase:
                    await websocket.send_json({"type": "tts_start", "index": idx, "text": phrase})
                    try:
                        async for chunk in tts_engine.stream_speech(phrase, brain.target_lang):
                            await websocket.send_bytes(chunk)
                    except Exception as exc:
                        print(f"[TTS stream phrase {idx}] erreur : {exc}")
                    await websocket.send_json({"type": "tts_end", "index": idx})
                    sentence_idx += 1

            elif event["type"] == "stream_done":
                speech_text = event.get("full_text", "")

            elif event["type"] == "progress":
                await websocket.send_json(event)

            elif event["type"] == "meta":
                meta_event = event
                interview_ended = event.get("interview_ended", False)

        return {
            "speech_text": speech_text,
            "meta_event": meta_event,
            "interview_ended": interview_ended,
            "sentence_idx": sentence_idx,
            "interrupted": False,
        }
    finally:
        if not producer.done():
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"🔌 WebSocket connecté : {session_id}")

    brain = sessions.get(session_id)
    if not brain:
        await websocket.send_json({"type": "error", "message": "Session introuvable."})
        await websocket.close()
        return

    _inbox: asyncio.Queue = asyncio.Queue()
    _stop = asyncio.Event()

    async def _reader():
        try:
            while not _stop.is_set():
                try:
                    raw = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if raw.get("type") == "websocket.disconnect":
                    break
                text = raw.get("text")
                if text:
                    try:
                        await _inbox.put(json.loads(text))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        finally:
            _stop.set()

    async def _handler():
        try:
            while not _stop.is_set():
                try:
                    data = await asyncio.wait_for(_inbox.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                action = data.get("action")

                if action == "ping":
                    await websocket.send_json({
                        "type":      "pong",
                        "time_left": brain.get_time_remaining(),
                        "phase":     brain.steps[brain.current_step_index],
                    })

                elif action == "text_message":
                    user_text = data.get("text", "").strip()
                    if not user_text:
                        await websocket.send_json({"type": "error", "message": "Texte vide reçu."})
                        continue

                    result = await _ws_stream_brain_response(websocket, _inbox, brain, user_text)
                    speech_text = result["speech_text"]
                    meta_event = result["meta_event"]
                    interview_ended = result["interview_ended"]
                    sentence_idx = result["sentence_idx"]

                    if result.get("interrupted"):
                        continue

                    if sentence_idx == 0 and speech_text.strip():
                        await websocket.send_json({"type": "tts_start", "index": 0, "text": speech_text})
                        try:
                            async for chunk in tts_engine.stream_speech(speech_text, brain.target_lang):
                                await websocket.send_bytes(chunk)
                        except Exception as e:
                            print(f"[TTS stream fallback] erreur : {e}")
                        await websocket.send_json({"type": "tts_end", "index": 0})

                    await websocket.send_json({
                        "type":                "meta",
                        "full_text":           speech_text,
                        "candidate_sentiment": meta_event.get("candidate_sentiment", "neutre"),
                        "phase":               meta_event.get("phase", brain.steps[brain.current_step_index]),
                        "time_left":           meta_event.get("time_left", brain.get_time_remaining()),
                        "interview_ended":     interview_ended,
                        "candidate_assessment": _build_live_candidate_assessment(session_id, brain),
                    })
                    await websocket.send_json({"type": "done"})

                    if interview_ended:
                        report = await _build_full_report(brain, None, session_id)
                        await websocket.send_json({"type": "report", **report})
                        await websocket.close()
                        print(f"🏁 Entretien terminé, WS fermé : {session_id}")
                        _stop.set()
                        break

                elif action == "interrupt":
                    pass

                else:
                    await websocket.send_json({
                        "type":    "error",
                        "message": f"Action inconnue : '{action}'",
                    })

        except WebSocketDisconnect:
            print(f"🔌 WebSocket déconnecté : {session_id}")
        except Exception as e:
            import traceback
            print(f"❌ WebSocket erreur : {e}")
            print(traceback.format_exc())
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            _stop.set()

    reader_task  = asyncio.create_task(_reader(),  name=f"ws-reader-{session_id[:8]}")
    handler_task = asyncio.create_task(_handler(), name=f"ws-handler-{session_id[:8]}")

    done, pending = await asyncio.wait(
        [reader_task, handler_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    _stop.set()
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    print(f"🔌 WebSocket session terminée proprement : {session_id}")

# =============================================================================
# LANCEMENT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    import subprocess

    loop = "uvloop" if sys.platform != "win32" else "asyncio"

    cert_file = BASE_DIR / "cert.pem"
    key_file  = BASE_DIR / "key.pem"

    if not cert_file.exists() or not key_file.exists():
        print("🔐 Génération du certificat SSL auto-signé...")
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_file),
                "-out",    str(cert_file),
                "-days",   "365",
                "-nodes",
                "-subj",   "/C=FR/ST=IDF/L=Paris/O=Avatar-RH/CN=localhost",
                "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"
            ], check=True, capture_output=True)
            print(f"✅ Certificat créé : {cert_file}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️  openssl indisponible ({e}) — lancement en HTTP")
            cert_file = None
            key_file  = None

    if cert_file and cert_file.exists():
        print("🔒 Démarrage en HTTPS sur https://localhost:8000")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            loop=loop,
            ssl_certfile=str(cert_file),
            ssl_keyfile=str(key_file),
        )
    else:
        print("🌐 Démarrage en HTTP sur http://localhost:8000")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, loop=loop)