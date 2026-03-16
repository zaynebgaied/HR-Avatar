import os
import sys
import json
import asyncio
import uuid
from pathlib import Path
from typing import List

# ── Garantit que tous les modules locaux sont trouvés quel que soit le CWD
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from stt_engine          import STTEngine
from llm_chain           import HRInteractiveBrain
from tts_engine          import TTSEngine
from interview_evaluator import InterviewEvaluator

# =============================================================================
# APP & DOSSIERS
# =============================================================================
app = FastAPI(title="Avatar RH Interactif", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR         = Path(__file__).parent
STATIC_DIR       = BASE_DIR / "static"
DATA_DIR         = BASE_DIR / "data"
TEMP_DIR         = BASE_DIR / "temp_audio"
COMPANY_INFO_DIR = DATA_DIR / "company_info"
REPORTS_DIR      = DATA_DIR / "reports"

for d in [STATIC_DIR, DATA_DIR, TEMP_DIR, COMPANY_INFO_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CV_OFFRE_EXTS = {".pdf", ".docx"}
COMPANY_EXTS  = {".pdf", ".docx", ".txt"}

app.mount("/static",     StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/temp_audio", StaticFiles(directory=str(TEMP_DIR)),   name="temp_audio")
app.mount("/data",       StaticFiles(directory=str(DATA_DIR)),   name="data")

# =============================================================================
# INSTANCES GLOBALES
# =============================================================================
stt_engine = STTEngine()
tts_engine = TTSEngine(output_dir=str(TEMP_DIR))

sessions: dict[str, HRInteractiveBrain] = {}

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
    print(f"✅  [{doc_type.upper()}] '{file.filename}' → {dest.name}  ({round(len(content)/1024,1)} Ko)")
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
        print(f"✅  [KB] Uploadé : {file.filename}  ({round(len(content)/1024,1)} Ko)")
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
    print(f"🗑️   [KB] Supprimé : {filename}")
    return {"status": "ok", "deleted": filename}

# =============================================================================
# SESSION : DÉMARRAGE  (HTTP — reste en HTTP car inclut l'ingestion RAG lourde)
# =============================================================================
@app.post("/session/start")
async def start_session(lang: str = Form(...), duration: int = Form(20)):
    session_id = uuid.uuid4().hex
    brain      = HRInteractiveBrain(target_lang=lang, duration_minutes=duration)

    for ext in CV_OFFRE_EXTS:
        p = DATA_DIR / f"cv{ext}"
        if p.exists():
            brain.ingest_document(str(p), "cv")
            break

    for ext in CV_OFFRE_EXTS:
        p = DATA_DIR / f"offre{ext}"
        if p.exists():
            brain.ingest_document(str(p), "job_offer")
            break

    for doc in sorted(COMPANY_INFO_DIR.iterdir()):
        if doc.suffix.lower() in COMPANY_EXTS:
            brain.ingest_document(str(doc), "company_info")

    kb_loaded = brain.ingested_docs.get("company_info", [])
    if kb_loaded:
        print(f"🏢  KB : {len(kb_loaded)} doc(s) injecté(s) — {', '.join(kb_loaded)}")
    else:
        print("⚠️   KB vide — RAG limité à CV + Offre.")

    sessions[session_id] = brain

    greeting_text = brain.get_initial_greeting()
    tts_result    = await tts_engine.generate_speech(
        greeting_text, lang, filename=f"greeting_{session_id}.wav"
    )

    return {
        "session_id":  session_id,
        "greeting":    greeting_text,
        "audio_url":   f"/temp_audio/{tts_result['filename']}" if tts_result["success"] else None,
        "duration_s":  tts_result["estimated_duration_s"],
        "phase":       brain.steps[brain.current_step_index],
        "time_left":   brain.get_time_remaining(),
        "rag_sources": {
            "cv":           brain.ingested_docs.get("cv",           []),
            "job_offer":    brain.ingested_docs.get("job_offer",    []),
            "company_docs": brain.ingested_docs.get("company_info", []),
        },
    }

# =============================================================================
# HELPER : RAPPORT FINAL
# =============================================================================
async def _build_full_report(brain: HRInteractiveBrain, inline_report: dict | None) -> dict:
    import datetime as _dt

    base = inline_report or {}

    nlu = {}
    try:
        evaluator = InterviewEvaluator()
        # Lire le fichier log de cette session precise
        raw_text, file_path = evaluator.load_latest_interview(
            specific_file=getattr(brain, "log_file", None)
        )
        parsed = evaluator.parse_interview(raw_text)
        # Passer brain.scores comme filet de secours si le LLM echoue
        nlu    = evaluator.run_nlu_assessment(parsed, brain_scores=brain.scores)
        evaluator.save_json_report(nlu, source_file=file_path)
        print("NLU Evaluator : rapport genere.")
    except Exception as e:
        print(f"NLU Evaluator indisponible : {e}")

    merged = {
        "date":              base.get("date", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
        "langue":            base.get("langue", brain.target_lang),
        "duree_minutes":     base.get("duree_minutes", brain.duration_minutes),
        "meta":              nlu.get("meta", {}),
        "phases":            base.get("phases", []),
        "score_total":       base.get("score_total", 0),
        "score_max":         base.get("score_max", 100),
        "pourcentage":       base.get("pourcentage", 0),
        "scores_par_phase":    nlu.get("scores_par_phase",    {}),
        "score_technique":     nlu.get("score_technique",     base.get("pourcentage", 0)),
        "score_communication": nlu.get("score_communication", 0),
        "score_global":        nlu.get("score_global",        base.get("pourcentage", 0)),
        "competences_detectees": nlu.get("competences_detectees", []),
        "lacunes_identifiees":   nlu.get("lacunes_identifiees",   []),
        "analyse_motivation":    nlu.get("analyse_motivation",    ""),
        "analyse_soft_skills":   nlu.get("analyse_soft_skills",   ""),
        "points_forts":          nlu.get("points_forts",          base.get("points_forts",   [])),
        "points_amelioration":   nlu.get("points_amelioration",   base.get("points_faibles", [])),
        "verdict_final":         nlu.get("verdict_final",         base.get("recommandation", "")),
        "recommandation_detail": nlu.get("recommandation_detail", base.get("recommandation", "")),
        "recommandation":        base.get("recommandation", ""),
        "sources_rag":           base.get("sources_rag", brain.ingested_docs),
    }

    try:
        ts   = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DATA_DIR / "reports" / f"full_report_{ts}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📄  Rapport complet sauvegardé : {path}")
    except Exception as e:
        print(f"⚠️  Sauvegarde échouée : {e}")

    return merged

# =============================================================================
# SESSION : RÉPONSE CANDIDAT (HTTP — fallback si WebSocket non disponible)
# =============================================================================
@app.post("/session/{session_id}/respond")
async def respond(session_id: str, user_text: str = Form(...)):
    brain = sessions.get(session_id)
    if not brain:
        return JSONResponse({"error": "Session introuvable."}, status_code=404)

    result     = brain.generate_response(user_text)
    uid        = uuid.uuid4().hex[:8]
    tts_result = await tts_engine.generate_speech(
        result["text"], brain.target_lang,
        filename=f"resp_{session_id}_{uid}.wav",
    )

    final_report = None
    if result.get("interview_ended"):
        final_report = await _build_full_report(brain, result.get("report"))

    return {
        "text":            result["text"],
        "sentiment":       result.get("candidate_sentiment", "neutre"),
        "audio_url":       f"/temp_audio/{tts_result['filename']}" if tts_result["success"] else None,
        "duration_s":      tts_result["estimated_duration_s"],
        "phase":           brain.steps[brain.current_step_index],
        "time_left":       brain.get_time_remaining(),
        "interview_ended": result.get("interview_ended", False),
        "report":          final_report,
    }

# =============================================================================
# SESSION : TRANSCRIPTION AUDIO → STT  (toujours HTTP — fichier binaire)
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
    if session_id not in sessions:
        return JSONResponse({"error": "Session introuvable."}, status_code=404)

    evaluator = InterviewEvaluator()
    try:
        raw_text, file_path = evaluator.load_latest_interview()
        parsed              = evaluator.parse_interview(raw_text)
        report              = evaluator.run_nlu_assessment(parsed)
        out_path            = evaluator.save_json_report(report, source_file=file_path)
        return {"status": "ok", "report": report, "report_path": out_path}
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# SESSION : STATUT (HTTP — utilisable hors WS pour debug)
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
# WEBSOCKET STT — streaming PCM Float32 depuis Web Audio API
# Evite le HTTP POST + ffmpeg : les chunks PCM arrivent directement
# CLIENT → SERVEUR : bytes Float32Array (PCM 16kHz mono)
# CLIENT → SERVEUR : texte "end" = fin de parole, lancer transcription
# SERVEUR → CLIENT : {"text":"...", "language":"fr", "success":true}
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

            # Chunk PCM binaire
            if msg_type == "websocket.receive" and message.get("bytes"):
                chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                audio_buffer = np.concatenate([audio_buffer, chunk])

            # Fin de parole : transcrire
            elif msg_type == "websocket.receive" and message.get("text") == "end":
                if len(audio_buffer) < 1600:
                    await websocket.send_json({"text": "", "language": "fr", "success": False, "error": "Audio trop court"})
                    audio_buffer = np.array([], dtype=np.float32)
                    continue
                try:
                    loop  = asyncio.get_event_loop()
                    brain = sessions.get(session_id)
                    lang_map  = {"Français": "fr", "Anglais": "en", "Arabe": "ar"}
                    lang_hint = lang_map.get(brain.target_lang if brain else "Français", None)

                    buf_copy = audio_buffer.copy()
                    def _do_transcribe():
                        segs, lang = stt_engine.transcribe_stream(buf_copy, language=lang_hint)
                        return " ".join(s.text.strip() for s in segs).strip(), lang

                    text, detected_lang = await loop.run_in_executor(None, _do_transcribe)
                    await websocket.send_json({"text": text, "language": detected_lang, "success": bool(text)})
                except Exception as e:
                    await websocket.send_json({"text": "", "language": "fr", "success": False, "error": str(e)})
                finally:
                    audio_buffer = np.array([], dtype=np.float32)

            elif msg_type == "websocket.receive" and message.get("text") == "cancel":
                audio_buffer = np.array([], dtype=np.float32)

    except Exception as e:
        print(f"STT WebSocket erreur : {e}")
    finally:
        print(f"STT WebSocket ferme : {session_id}")


# =============================================================================
# WEBSOCKET — streaming temps réel de l'entretien
#
# Protocole messages CLIENT → SERVEUR :
#   {"action": "text_message", "text": "réponse du candidat"}
#   {"action": "ping"}
#
# Protocole messages SERVEUR → CLIENT :
#   {"type": "token",     "token": "mot"}              ← streaming LLM
#   {"type": "tts_ready", "audio_url": "/temp_audio/x.mp3",
#                         "duration_s": 4.2}            ← audio prêt
#   {"type": "meta",      "full_text": "...",
#                         "candidate_sentiment": "...",
#                         "phase": "...", "time_left": 18.3,
#                         "interview_ended": false}      ← fin de génération
#   {"type": "done"}                                    ← fin du tour
#   {"type": "pong",      "time_left": 18.3,
#                         "phase": "..."}               ← réponse au ping
#   {"type": "report",    ...}                          ← rapport final
#   {"type": "error",     "message": "..."}
# =============================================================================
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"🔌 WebSocket connecté : {session_id}")

    brain = sessions.get(session_id)
    if not brain:
        await websocket.send_json({"type": "error", "message": "Session introuvable."})
        await websocket.close()
        return

    try:
        while True:
            data   = await websocket.receive_json()
            action = data.get("action")

            # ── PING — mise à jour timer sans échange LLM ─────────────────
            if action == "ping":
                await websocket.send_json({
                    "type":      "pong",
                    "time_left": brain.get_time_remaining(),
                    "phase":     brain.steps[brain.current_step_index],
                    "scores":    brain.scores,
                })

            # ── TEXT_MESSAGE — cœur du streaming ─────────────────────────
            elif action == "text_message":
                user_text = data.get("text", "").strip()
                if not user_text:
                    await websocket.send_json({"type": "error", "message": "Texte vide reçu."})
                    continue

                speech_text     = ""
                meta_event      = {}
                interview_ended = False
                uid             = uuid.uuid4().hex[:8]

                # ── Collecte des phrases émises par le LLM stream ────────────
                # On accumule dans pending_sentences au fil du stream,
                # et on streame le TTS de chaque phrase en binaire dès qu'elle arrive.
                sentence_idx    = 0
                tts_meta_task   = None  # lancé après la fin du stream LLM

                async for event in brain.generate_response_stream(user_text):

                    if event["type"] == "token":
                        await websocket.send_json({"type": "token", "token": event["token"]})

                    elif event["type"] == "sentence":
                        # ── Stream TTS binaire pour cette phrase ─────────────
                        # Protocole :
                        #   → {"type":"tts_start", "index": N}   JSON
                        #   → <bytes> <bytes> <bytes> …           MP3 chunks
                        #   → {"type":"tts_end",   "index": N}   JSON
                        phrase = event["text"].strip()
                        idx    = event["index"]
                        if phrase:
                            await websocket.send_json({
                                "type":  "tts_start",
                                "index": idx,
                                "text":  phrase,
                            })
                            try:
                                async for chunk in tts_engine.stream_speech(
                                    phrase, brain.target_lang
                                ):
                                    await websocket.send_bytes(chunk)
                            except Exception as e:
                                print(f"[TTS stream phrase {idx}] erreur : {e}")
                            await websocket.send_json({
                                "type":  "tts_end",
                                "index": idx,
                            })
                            sentence_idx += 1

                    elif event["type"] == "stream_done":
                        speech_text = event.get("full_text", "")

                    elif event["type"] == "meta":
                        meta_event      = event
                        interview_ended = event.get("interview_ended", False)

                # ── Si aucune phrase n'a été émise (réponse très courte),
                #    on streame le texte complet en fallback ──────────────────
                if sentence_idx == 0 and speech_text.strip():
                    await websocket.send_json({"type": "tts_start", "index": 0, "text": speech_text})
                    try:
                        async for chunk in tts_engine.stream_speech(speech_text, brain.target_lang):
                            await websocket.send_bytes(chunk)
                    except Exception as e:
                        print(f"[TTS stream fallback] erreur : {e}")
                    await websocket.send_json({"type": "tts_end", "index": 0})

                # ── Métadonnées finales ──────────────────────────────────────
                await websocket.send_json({
                    "type":                "meta",
                    "full_text":           speech_text,
                    "candidate_sentiment": meta_event.get("candidate_sentiment", "neutre"),
                    "phase":               meta_event.get("phase", brain.steps[brain.current_step_index]),
                    "time_left":           meta_event.get("time_left", brain.get_time_remaining()),
                    "scores":              brain.scores,
                    "interview_ended":     interview_ended,
                })

                # Signal fin de tour
                await websocket.send_json({"type": "done"})

                # Fin d'entretien
                if interview_ended:
                    report = await _build_full_report(brain, None)
                    await websocket.send_json({"type": "report", **report})
                    await websocket.close()
                    print(f"🏁 Entretien terminé, WS fermé : {session_id}")
                    break

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

# =============================================================================
# LANCEMENT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)