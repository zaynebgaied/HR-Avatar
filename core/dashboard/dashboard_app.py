"""
dashboard_app.py — Avatar RH Interactif — Plateforme Dynamique v3.1
=================================================================
Corrections v3.1 :
  - Cause 1 : Bouton "Rouvrir fenêtre" sans changer le token
  - Cause 2 : Avertissement avant régénération + affichage lien actuel
  - Cause 3 : URL de base persistante + détection IP automatique
"""

import streamlit as st
import json
import os
import hashlib
import uuid
import datetime
import shutil
import socket
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Avatar RH — Plateforme",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"  # core/data/
USERS_FILE    = DATA_DIR / "users.json"
REPORTS_DIR   = DATA_DIR / "reports"
CV_DIR        = DATA_DIR / "cv_offres"
BASE_URL_FILE = DATA_DIR / "base_url.txt"

for d in [DATA_DIR, REPORTS_DIR, CV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8f9fb; }
[data-testid="stSidebar"] { background: #1a1f2e; border-right: 1px solid #2d3348; }
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
.card {
    background: #fff; border-radius: 14px; padding: 1.4rem 1.6rem;
    border: 1px solid #e8ecf4; margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
.card-metric {
    background: #fff; border-radius: 12px; padding: 1.2rem 1.4rem;
    border: 1px solid #e8ecf4; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1a1f2e; margin: 0; }
.metric-label { font-size: 0.8rem; color: #8a93a8; margin: 0; text-transform: uppercase; letter-spacing: .04em; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
}
.badge-admin   { background: #ede9fe; color: #5b21b6; }
.badge-user    { background: #dbeafe; color: #1d4ed8; }
.badge-ok      { background: #dcfce7; color: #166534; }
.badge-warn    { background: #fef3c7; color: #92400e; }
.badge-danger  { background: #fee2e2; color: #991b1b; }
.badge-info    { background: #e0f2fe; color: #075985; }
.badge-purple  { background: #ede9fe; color: #5b21b6; }
.badge-blocked { background: #374151; color: #fff; }
.link-box {
    background: #f0f4ff; border: 1.5px solid #4f46e5; border-radius: 10px;
    padding: 1rem 1.2rem; margin-top: .8rem;
}
.link-box code { font-size: .85rem; color: #1a1f2e; word-break: break-all; }
.notif-banner {
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    border-radius: 12px; padding: 1rem 1.4rem; color:#fff;
    margin-bottom: 1rem; display:flex; align-items:center; gap:12px;
}
.schedule-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
    padding: 4px 12px; font-size: .82rem; color: #1e40af; font-weight: 600;
}
.score-bar-bg {
    background: #f1f3f8; border-radius: 6px; height: 8px;
    width: 100%; overflow: hidden; margin-top: 4px;
}
.score-bar-fill { height: 8px; border-radius: 6px; }
.required-badge {
    display: inline-block; background: #fee2e2; color: #991b1b;
    border-radius: 6px; padding: 2px 8px; font-size: .72rem; font-weight: 600;
}
.ok-badge {
    display: inline-block; background: #dcfce7; color: #166534;
    border-radius: 6px; padding: 2px 8px; font-size: .72rem; font-weight: 600;
}
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
thead th {
    text-align: left; padding: 10px 12px; background: #f8f9fb;
    color: #6b7280; font-weight: 600; border-bottom: 1px solid #e8ecf4;
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em;
}
tbody td { padding: 10px 12px; border-bottom: 1px solid #f1f3f8; color: #374151; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: #fafbff; }
.avatar {
    width: 40px; height: 40px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
}
.sidebar-user {
    background: rgba(255,255,255,.06); border-radius: 12px;
    padding: 0.8rem 1rem; margin: 1rem 0; border: 1px solid rgba(255,255,255,.08);
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# URL DE BASE — PERSISTANCE + DÉTECTION IP  [CORRECTIF CAUSE 3]
# ─────────────────────────────────────────────────────────────────────────────
def load_base_url() -> str:
    if BASE_URL_FILE.exists():
        return BASE_URL_FILE.read_text(encoding="utf-8").strip()
    return "http://localhost:8000"


def save_base_url(url: str):
    BASE_URL_FILE.write_text(url.strip(), encoding="utf-8")


def detect_local_ip() -> str:
    """Retourne l'URL avec l'IP LAN de la machine (utile pour accès réseau local)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return f"http://{ip}:8000"
    except Exception:
        return "http://localhost:8000"


# ─────────────────────────────────────────────────────────────────────────────
# USER STORE
# ─────────────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _default_fields(u: dict) -> dict:
    defaults = {
        "login_history": [], "is_blocked": False,
        "session_id": None, "session_status": "en_attente",
        "poste": None, "langue": None, "duree": None,
        "scheduled_at": None,
        "scheduled_window_min": 30,
        "access_token": None,
        "cv_file": None,
        "offre_file": None,
        "report_file": None, "report_read_at": None,
        "rh_comment": "", "decision_rh": "", "notifications": [],
    }
    for k, v in defaults.items():
        if k not in u:
            u[k] = v
    return u


def load_users() -> dict:
    if USERS_FILE.exists():
        raw = USERS_FILE.read_text(encoding="utf-8").strip()
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Fichier corrompu → sauvegarde et reconstruction
                import shutil as _shutil
                _shutil.copy2(str(USERS_FILE), str(USERS_FILE.with_suffix(".bak")))
                USERS_FILE.unlink()
    default = {"admin": _default_fields({
        "id": str(uuid.uuid4()), "name": "Administrateur",
        "email": "admin@rh.local", "password": _hash("admin123"),
        "role": "admin", "created_at": datetime.datetime.now().isoformat(),
        "last_login": None,
    })}
    save_users(default)
    return default


def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


def authenticate(username: str, password: str):
    users = load_users()
    u = users.get(username)
    if not u or u["password"] != _hash(password):
        return None
    if u.get("is_blocked"):
        return "BLOCKED"
    u = _default_fields(u)
    now = datetime.datetime.now().isoformat()
    u["last_login"] = now
    u["login_history"].append({"at": now, "ok": True})
    u["login_history"] = u["login_history"][-20:]
    users[username] = u
    save_users(users)
    return u


def register_user(username: str, name: str, email: str, password: str) -> tuple[bool, str]:
    users = load_users()
    if username in users:
        return False, "Nom d'utilisateur déjà pris."
    if any(u["email"] == email for u in users.values()):
        return False, "Email déjà utilisé."
    users[username] = _default_fields({
        "id": str(uuid.uuid4()), "name": name, "email": email,
        "password": _hash(password), "role": "user",
        "created_at": datetime.datetime.now().isoformat(), "last_login": None,
    })
    for uname, u in users.items():
        if u.get("role") == "admin":
            u.setdefault("notifications", []).append({
                "type": "new_candidate",
                "message": f"👤 Nouveau candidat inscrit : {name} (@{username})",
                "at": datetime.datetime.now().isoformat(), "read": False,
            })
    save_users(users)
    return True, "Compte créé avec succès !"


def get_candidates() -> list:
    users = load_users()
    return [{**_default_fields(u), "username": uname}
            for uname, u in users.items() if u.get("role") == "user"]


def assign_session_to_user(username: str, session_id: str, poste: str, langue: str, duree: int,
                            scheduled_at: str | None = None, window_min: int = 30):
    users = load_users()
    if username not in users:
        return
    u = _default_fields(users[username])
    access_token = uuid.uuid4().hex
    u.update({
        "session_id":           session_id,
        "poste":                poste,
        "langue":               langue,
        "duree":                duree,
        "session_status":       "configure",
        "scheduled_at":         scheduled_at,
        "scheduled_window_min": window_min,
        "access_token":         access_token,
    })
    u["notifications"].append({
        "type":    "session_ready",
        "message": f"🎤 Votre entretien est planifié ! Poste : {poste} | Langue : {langue}"
                   + (f" |  {scheduled_at[:16].replace('T',' ')}" if scheduled_at else ""),
        "at":      datetime.datetime.now().isoformat(),
        "read":    False,
    })
    users[username] = u
    save_users(users)
    return access_token


def extend_session_window(username: str, new_scheduled_at: str | None, new_window_min: int) -> bool:
    """[CORRECTIF CAUSE 1] Repousse la fenêtre d'accès SANS changer le token ni le session_id."""
    users = load_users()
    if username not in users:
        return False
    u = _default_fields(users[username])
    u["scheduled_at"]         = new_scheduled_at
    u["scheduled_window_min"] = new_window_min
    u["notifications"].append({
        "type":    "session_ready",
        "message": " Votre fenêtre d'entretien a été mise à jour"
                   + (f" — nouveau créneau : {new_scheduled_at[:16].replace('T',' ')}" if new_scheduled_at else " — accès immédiat activé"),
        "at":  datetime.datetime.now().isoformat(),
        "read": False,
    })
    users[username] = u
    save_users(users)
    return True


def block_user(username: str, block: bool):
    users = load_users()
    if username in users:
        users[username]["is_blocked"] = block
        save_users(users)


def mark_report_read(username: str):
    users = load_users()
    if username in users:
        users[username]["report_read_at"] = datetime.datetime.now().isoformat()
        save_users(users)


def mark_notifications_read(username: str):
    users = load_users()
    if username in users:
        for n in users[username].get("notifications", []):
            n["read"] = True
        save_users(users)


def get_unread_count(username: str) -> int:
    u = load_users().get(username, {})
    return sum(1 for n in u.get("notifications", []) if not n.get("read"))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS PLANIFICATION & FICHIERS
# ─────────────────────────────────────────────────────────────────────────────
def build_access_link(session_id: str, access_token: str, base_url: str | None = None) -> str:
    if base_url is None:
        base_url = load_base_url()
    return f"{base_url}/interview/{access_token}"


def schedule_status(u: dict) -> dict:
    scheduled_at = u.get("scheduled_at")
    if not scheduled_at:
        return {"label": "Pas d'horaire — accès immédiat", "ok": True,
                "future": False, "expired": False, "minutes_until": 0}
    try:
        dt_start = datetime.datetime.fromisoformat(scheduled_at)
    except Exception:
        return {"label": "Horaire invalide", "ok": False, "future": False, "expired": True, "minutes_until": 0}
    
    window   = int(u.get("scheduled_window_min", 30))
    dt_end   = dt_start + datetime.timedelta(minutes=window)
    
    # ✅ CORRECTION : forcer les deux datetime en naive (sans timezone)
    # pour éviter tout décalage UTC vs heure locale
    now = datetime.datetime.now().replace(tzinfo=None)
    dt_start = dt_start.replace(tzinfo=None)
    dt_end   = dt_end.replace(tzinfo=None)
    
    if now < dt_start:
        diff = int((dt_start - now).total_seconds() / 60)
        return {"label": f"Planifié le {dt_start.strftime('%d/%m/%Y à %H:%M')} (dans {diff} min)",
                "ok": False, "future": True, "expired": False, "minutes_until": diff}
    elif now <= dt_end:
        left = int((dt_end - now).total_seconds() / 60)
        return {"label": f"🟢 Fenêtre ouverte — ferme dans {left} min",
                "ok": True, "future": False, "expired": False, "minutes_until": 0}
    else:
        return {"label": f"⛔ Fenêtre expirée (était le {dt_start.strftime('%d/%m/%Y à %H:%M')})",
                "ok": False, "future": False, "expired": True, "minutes_until": 0}


def _copy_candidate_files(username: str):
    """Copie les CV/offre du candidat depuis cv_offres/ vers data/ pour le moteur de session."""
    users      = load_users()
    u          = users.get(username, {})
    cv_file    = u.get("cv_file")
    offre_file = u.get("offre_file")
    if cv_file and (CV_DIR / cv_file).exists():
        ext = Path(cv_file).suffix
        shutil.copy2(str(CV_DIR / cv_file), str(DATA_DIR / f"cv{ext}"))
    if offre_file and (CV_DIR / offre_file).exists():
        ext = Path(offre_file).suffix
        shutil.copy2(str(CV_DIR / offre_file), str(DATA_DIR / f"offre{ext}"))


def _candidate_files_status(u: dict) -> dict:
    """Retourne le statut des fichiers CV et Offre pour un candidat."""
    cv_file    = u.get("cv_file")
    offre_file = u.get("offre_file")
    cv_ok      = bool(cv_file and (CV_DIR / cv_file).exists())
    offre_ok   = bool(offre_file and (CV_DIR / offre_file).exists())
    return {
        "cv_ok":       cv_ok,
        "offre_ok":    offre_ok,
        "cv_file":     cv_file,
        "offre_file":  offre_file,
        "both_ok":     cv_ok and offre_ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS RAPPORTS
# ─────────────────────────────────────────────────────────────────────────────
PHASE_LABELS_FR = {
    "OPENING":                 "Accueil & Présentation",
    "JOB_ALIGNED_EXPLORATION": "Exploration Technique",
    "PROJECT_DEEP_DIVE":       "Analyse de Projets",
    "TECHNICAL_DEPTH":         "Profondeur Technique",
    "SOFT_SKILLS_BEHAVIORAL":  "Soft Skills & Comportemental",
    "CANDIDATE_QUESTIONS":     "Questions Candidat",
    "FINAL_CHECK":             "Vérification Finale",
    "CLOSING":                 "Clôture",
}

VERDICT_STYLES = {
    "HIRE":         ("#dcfce7", "#166534"),
    "RECONSIDER":   ("#fef3c7", "#92400e"),
    "TO_COMPLETE":  ("#e0f2fe", "#075985"),
    "NOT_RETAINED": ("#fee2e2", "#991b1b"),
}


def _verdict_to_status(v: str) -> str:
    v = v.upper()
    if "HIRE" in v:                               return "Recommandé"
    if "NOT_RETAINED" in v:                       return "Refusé"
    if "RECONSIDER" in v or "TO_COMPLETE" in v:   return "En attente"
    if "RECOMMAND" in v:                          return "Recommandé"
    if "REFUS" in v or "NON" in v:                return "Refusé"
    return "En attente"


def load_real_reports() -> list:
    reports = []
    if not REPORTS_DIR.exists():
        return reports
    for p in sorted(REPORTS_DIR.glob("*.json"),
                    key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data       = json.loads(p.read_text(encoding="utf-8"))
            meta       = data.get("meta") or {}
            score      = int(data.get("score_total")  or data.get("score_global")  or data.get("pourcentage") or 0)
            score_tech = int(data.get("score_technical") or data.get("score_technique") or 0)
            score_beh  = int(data.get("score_behavioral") or 0)
            score_comm = int(data.get("score_communication") or 0)
            verdict_raw = data.get("verdict") or data.get("verdict_final") or ""
            vision  = data.get("analyse_emotion_vision") or {}
            emotion = vision.get("emotion_dominante", "neutre") if isinstance(vision, dict) else "neutre"
            phases  = data.get("scores_by_phase") or data.get("scores_par_phase") or {}
            candidate = (data.get("candidate_name") or meta.get("candidat")
                         or data.get("username") or p.stem)
            reports.append({
                "id":          p.stem,
                "file":        str(p),
                "candidate":   candidate,
                "username":    data.get("username", ""),
                "poste":       meta.get("poste") or data.get("poste") or "—",
                "date":        data.get("date") or p.stem,
                "score":       score,
                "score_tech":  score_tech,
                "score_beh":   score_beh,
                "score_comm":  score_comm,
                "langue":      data.get("langue") or meta.get("language") or "—",
                "duree":       f"{data.get('duree_minutes') or meta.get('duration') or '?'} min",
                "status":      _verdict_to_status(verdict_raw),
                "verdict_raw": verdict_raw,
                "emotion_dominante": emotion,
                "phases":      phases,
                "_raw":        data,
            })
        except Exception:
            pass
    return reports


def get_score_color(s: int) -> str:
    return "#16a34a" if s >= 75 else "#d97706" if s >= 55 else "#dc2626"


def badge(label: str, cls: str) -> str:
    return f'<span class="badge {cls}">{label}</span>'


def status_badge(s: str) -> str:
    return badge(s, {"Recommandé":"badge-ok","En attente":"badge-warn","Refusé":"badge-danger"}.get(s,"badge-info"))


def verdict_badge(v: str) -> str:
    icons = {"HIRE":"✅ HIRE","RECONSIDER":"⚠️ RECONSIDER",
             "TO_COMPLETE":"🔁 TO_COMPLETE","NOT_RETAINED":"❌ NOT_RETAINED"}
    cls   = {"HIRE":"badge-ok","RECONSIDER":"badge-warn",
             "TO_COMPLETE":"badge-info","NOT_RETAINED":"badge-danger"}
    return badge(icons.get(v.upper(), v or "—"), cls.get(v.upper(),"badge-info"))


def emotion_badge(e: str) -> str:
    return badge(e, {"détendu":"badge-ok","neutre":"badge-info",
                     "anxieux":"badge-warn","tendu":"badge-danger"}.get(e,"badge-info"))


# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────────────────────────────────────
def page_login():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1rem;">
            <div style="font-size:3rem;"></div>
            <h2 style="margin:.3rem 0;color:#1a1f2e;">Avatar RH Interactif</h2>
            <p style="color:#8a93a8;font-size:.88rem;">Connectez-vous à votre espace</p>
        </div>""", unsafe_allow_html=True)
        with st.form("login_form"):
            username  = st.text_input("👤 Nom d'utilisateur", placeholder="admin")
            password  = st.text_input("🔒 Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter →", use_container_width=True)
        if submitted:
            result = authenticate(username.strip(), password)
            if result == "BLOCKED":
                st.error("❌ Compte désactivé. Contactez l'administrateur.")
            elif result:
                st.session_state.user     = result
                st.session_state.username = username.strip()
                st.rerun()
            else:
                st.error("Identifiants incorrects.")
        st.markdown("---")
        if st.button("✨ Créer un compte candidat", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()
        st.caption("Admin : `admin` / `admin123`")


def page_register():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-size:2rem;">✨</div>
            <h2 style="margin:0;color:#1a1f2e;">Créer un compte</h2>
            <p style="color:#8a93a8;font-size:.85rem;">Espace candidat</p>
        </div>""", unsafe_allow_html=True)
        with st.form("register_form"):
            name     = st.text_input("Nom complet",       placeholder="Yassine Mrabet")
            username = st.text_input("Nom d'utilisateur", placeholder="ymrabet")
            email    = st.text_input("Email",             placeholder="yassine@email.com")
            pw1      = st.text_input("Mot de passe",      type="password")
            pw2      = st.text_input("Confirmer",         type="password")
            ok       = st.form_submit_button("Créer mon compte →", use_container_width=True)
        if ok:
            if not all([name, username, email, pw1, pw2]):
                st.warning("Tous les champs sont obligatoires.")
            elif pw1 != pw2:
                st.error("Mots de passe différents.")
            elif len(pw1) < 6:
                st.error("Mot de passe trop court (min. 6 car.).")
            else:
                success, msg = register_user(username.strip(), name.strip(), email.strip(), pw1)
                if success:
                    st.success(msg + " L'admin sera notifié.")
                    st.balloons()
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(msg)
        if st.button("← Retour"):
            st.session_state.page = "login"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(user: dict, role: str, username: str) -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 .5rem;display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.6rem;"></span>
            <span style="font-size:1rem;font-weight:700;color:#fff;">Avatar RH</span>
        </div>""", unsafe_allow_html=True)

        initials = "".join(w[0].upper() for w in user["name"].split()[:2])
        st.markdown(f"""
        <div class="sidebar-user">
            <div style="display:flex;align-items:center;gap:10px;">
                <div class="avatar" style="background:#4f46e5;color:#fff;">{initials}</div>
                <div>
                    <div style="font-weight:600;font-size:.88rem;color:#fff;">{user['name']}</div>
                    <div style="font-size:.75rem;color:#8a93a8;">{'🛡️ Admin' if role=='admin' else '👤 Candidat'}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        unread = get_unread_count(username)
        if unread:
            st.markdown(f"""<div style="background:#dc2626;color:#fff;border-radius:20px;
                padding:4px 12px;font-size:.75rem;font-weight:700;text-align:center;margin-bottom:8px;">
                🔔 {unread} notification(s) non lue(s)</div>""", unsafe_allow_html=True)

        st.markdown("---")

        if role == "admin":
            pages = [
                ("accueil",      "", "Tableau de bord"),
                ("entretiens",   "", "Entretiens"),
                ("rapports",     "", "Rapports"),
                ("config",       "", "Configuration"),
                ("utilisateurs", "", "Utilisateurs"),
                ("notifications","", f"Notifications{' ('+str(unread)+')' if unread else ''}"),
            ]
        else:
            pages = [
                ("accueil",      "", "Mon espace"),
                ("notifications","", f"Notifications{' ('+str(unread)+')' if unread else ''}"),
                ("entretien",    "", "Mon entretien"),
                ("resultats",    "", "Mes résultats"),
                ("profil",       "", "Mon profil"),
            ]

        for page_key, icon, label in pages:
            if st.button(f"{icon}  {label}", key=f"nav_{page_key}", use_container_width=True):
                st.session_state.active_page = page_key
                if page_key == "notifications":
                    mark_notifications_read(username)
                st.rerun()

        st.markdown("---")
        if st.button("  Déconnexion", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    return st.session_state.get("active_page", "accueil")


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
def page_notifications(username: str):
    st.markdown("## 🔔 Notifications")
    u      = load_users().get(username, {})
    notifs = list(reversed(u.get("notifications", [])))
    if not notifs:
        st.info("Aucune notification.")
        return
    type_icons = {"new_candidate":"👤","session_ready":"🎤","report_ready":"📊","info":"ℹ️"}
    for n in notifs:
        read   = n.get("read", False)
        bg     = "#fff" if read else "#f0f4ff"
        border = "#e8ecf4" if read else "#4f46e5"
        icon   = type_icons.get(n.get("type","info"), "🔔")
        at     = n.get("at","")[:16].replace("T"," ")
        st.markdown(f"""
        <div style="background:{bg};border-left:3px solid {border};border-radius:8px;
             padding:.85rem 1rem;margin-bottom:.6rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div style="display:flex;gap:10px;">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <span style="font-size:.88rem;color:#1a1f2e;">{n.get('message','')}</span>
                </div>
                <span style="font-size:.75rem;color:#8a93a8;white-space:nowrap;margin-left:12px;">{at}</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def _status_config(status: str) -> tuple[str, str, str]:
    return {
        "en_attente": ("⏳", "Pas encore fait",  "#d97706"),
        "configure":  ("📋", "Configuré",        "#4f46e5"),
        "en_cours":   ("🎤", "En cours",          "#059669"),
        "termine":    ("✅", "Terminé",           "#16a34a"),
    }.get(status, ("❓", status, "#6b7280"))


def admin_dashboard():
    reports    = load_real_reports()
    users      = load_users()
    candidates = {u: d for u, d in users.items() if d.get("role") == "user"}
    total        = len(reports)
    avg_score    = round(sum(r["score"] for r in reports) / total) if total else 0
    recommandes  = sum(1 for r in reports if r["status"] == "Recommandé")
    nb_candidats = len(candidates)

    has_live = any(
        _default_fields(d).get("session_status") == "en_cours"
        for d in candidates.values()
    )
    if has_live:
        st.markdown(
            '<div class="notif-banner">🎤 <strong>Entretien en cours</strong> — '
            'Cette page se rafraîchit automatiquement toutes les 20 secondes.</div>',
            unsafe_allow_html=True
        )

    st.markdown("## Tableau de bord")
    cols = st.columns(4)
    for col, (label, val, color, sub) in zip(cols, [
        ("Candidats inscrits", str(nb_candidats), "#4f46e5", "comptes actifs"),
        ("Score moyen",        f"{avg_score}/100", "#16a34a", "tous entretiens"),
        ("Recommandés",        str(recommandes),   "#059669", f"sur {total} entretiens"),
        ("Entretiens total",   str(total),          "#d97706", "rapports disponibles"),
    ]):
        with col:
            st.markdown(f"""<div class="card-metric">
                <p class="metric-value" style="color:{color};">{val}</p>
                <p class="metric-label">{label}</p>
                <p style="font-size:.72rem;color:#8a93a8;margin:4px 0 0;">{sub}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    missing_files = [
        (u, d) for u, d in candidates.items()
        if not _default_fields(d).get("cv_file") or not _default_fields(d).get("offre_file")
    ]
    if missing_files:
        names = ", ".join(_default_fields(d)["name"] for _, d in missing_files[:3])
        st.warning(
            f"⚠️ **{len(missing_files)} candidat(s) sans CV ou Offre** : {names} — "
            f"allez dans **Configuration** pour uploader les documents."
        )

    pending = [(u, d) for u, d in candidates.items()
               if _default_fields(d).get("session_status") == "en_attente"]
    if pending:
        names = ", ".join(_default_fields(d)["name"] for _, d in pending[:3])
        st.warning(f"⚠️ **{len(pending)} candidat(s) en attente** : {names} — allez dans **Configuration**.")

    st.markdown("### 🔴 Suivi temps réel des entretiens")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    configured_candidates = [
        (uname, _default_fields(d)) for uname, d in candidates.items()
        if _default_fields(d).get("session_id") or _default_fields(d).get("session_status") != "en_attente"
    ]

    if not configured_candidates:
        st.markdown('<p style="color:#8a93a8;text-align:center;padding:1rem;">Aucune session configurée.</p>',
                    unsafe_allow_html=True)
    else:
        rows_html = ""
        for uname, d in sorted(configured_candidates,
                                key=lambda x: x[1].get("session_status",""), reverse=True):
            status   = d.get("session_status", "en_attente")
            emoji, label, color = _status_config(status)
            sched    = schedule_status(d)
            poste    = d.get("poste", "—")
            name     = d.get("name", uname)
            sched_txt = sched["label"][:50]
            fs       = _candidate_files_status(d)
            files_html = (
                '<span class="ok-badge">CV ✓</span> <span class="ok-badge">Offre ✓</span>'
                if fs["both_ok"] else
                f'{"<span class=\"ok-badge\">CV ✓</span>" if fs["cv_ok"] else "<span class=\"required-badge\">CV ✗</span>"} '
                f'{"<span class=\"ok-badge\">Offre ✓</span>" if fs["offre_ok"] else "<span class=\"required-badge\">Offre ✗</span>"}'
            )

            if status == "en_cours":
                status_cell = (
                    f'<span class="badge badge-purple" style="background:#d1fae5;color:#065f46;border:1px solid #6ee7b7;">'
                    f'{emoji} {label}</span>'
                )
            elif status == "termine":
                status_cell = f'<span class="badge badge-ok">{emoji} {label}</span>'
            elif status == "configure":
                status_cell = f'<span class="badge badge-info">{emoji} {label}</span>'
            else:
                status_cell = f'<span class="badge badge-warn">{emoji} {label}</span>'

            rows_html += f"""
            <tr>
                <td><strong>{name}</strong><br>
                    <span style="font-size:.72rem;color:#8a93a8;">@{uname}</span></td>
                <td>{poste}</td>
                <td>{status_cell}</td>
                <td>{files_html}</td>
                <td><span style="font-size:.75rem;color:#8a93a8;">{sched_txt}</span></td>
            </tr>"""

        st.markdown(f"""
        <table><thead><tr>
            <th>Candidat</th><th>Poste</th><th>Statut</th><th>Documents</th><th>Horaire</th>
        </tr></thead><tbody>{rows_html}</tbody></table>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("### Derniers entretiens")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if not reports:
            st.markdown('<p style="color:#8a93a8;text-align:center;padding:1.5rem;">Aucun entretien réalisé.</p>',
                        unsafe_allow_html=True)
        else:
            rows = "".join(f"""
            <tr>
                <td><strong>{r['candidate']}</strong><br>
                    <span style="font-size:.75rem;color:#8a93a8;">{r['poste']}</span></td>
                <td>{r['date'][:10]}</td>
                <td><strong style="color:{get_score_color(r['score'])};">{r['score']}</strong>/100</td>
                <td>{verdict_badge(r['verdict_raw'])}</td>
                <td>{emotion_badge(r['emotion_dominante'])}</td>
                <td>{r['duree']}</td>
            </tr>""" for r in reports[:5])
            st.markdown(f"""<table><thead><tr>
                <th>Candidat</th><th>Date</th><th>Score</th>
                <th>Verdict</th><th>Émotion</th><th>Durée</th>
            </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("### 👥 Statuts globaux")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sc = {}
        for d in candidates.values():
            s = _default_fields(d).get("session_status","en_attente")
            sc[s] = sc.get(s,0)+1
        for key, (label, color) in [
            ("en_attente",("⏳ Pas encore fait","#d97706")),
            ("configure", ("📋 Configuré",      "#4f46e5")),
            ("en_cours",  ("🎤 En cours",        "#059669")),
            ("termine",   ("✅ Terminé",         "#16a34a")),
        ]:
            count = sc.get(key, 0)
            pct   = round(count/nb_candidats*100) if nb_candidats else 0
            st.markdown(f"""<div style="margin-bottom:.8rem;">
                <div style="display:flex;justify-content:space-between;font-size:.83rem;">
                    <span>{label}</span>
                    <span style="color:{color};font-weight:600;">{count}</span>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        if not candidates:
            st.caption("Aucun candidat inscrit.")
        st.markdown("---")
        if st.button("🔄 Rafraîchir maintenant", use_container_width=True):
            st.rerun()
        st.caption(f"Mis à jour : {datetime.datetime.now().strftime('%H:%M:%S')}")
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — ENTRETIENS
# ─────────────────────────────────────────────────────────────────────────────
def admin_entretiens():
    st.markdown("##  Sessions d'entretien")
    c_ref, c_ts = st.columns([1, 3])
    with c_ref:
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()
    with c_ts:
        st.caption(f"Dernière mise à jour : {datetime.datetime.now().strftime('%H:%M:%S')}")

    candidates = [c for c in get_candidates() if c.get("session_id")]
    if not candidates:
        st.info("Aucune session configurée. Allez dans **Configuration**.")
        return

    st_badges = {
        "configure":  ("badge-info",   "📋 Configuré"),
        "en_cours":   ("badge-purple", "🎤 En cours"),
        "termine":    ("badge-ok",     "✅ Terminé"),
        "en_attente": ("badge-warn",   "⏳ Pas encore fait"),
    }

    for c in candidates:
        sched = schedule_status(c)
        s     = c.get("session_status", "en_attente")
        badge_cls, badge_label = st_badges.get(s, ("badge-info", s))
        fs    = _candidate_files_status(c)

        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([2.5, 1.5, 1.5, 1.5, 2, 2])
            with col1:
                blocked = " 🚫" if c.get("is_blocked") else ""
                st.markdown(f"**{c['name']}{blocked}** — *{c.get('poste','—')}*")
                st.caption(f"@{c['username']} · {c.get('email','')}")
            with col2:
                st.caption(c.get("created_at","")[:10] or "—")
            with col3:
                st.markdown(
                    f'<span class="badge {badge_cls}">{badge_label}</span>',
                    unsafe_allow_html=True
                )
            with col4:
                cv_tag    = '<span class="ok-badge">CV ✓</span>'    if fs["cv_ok"]    else '<span class="required-badge">CV ✗</span>'
                offre_tag = '<span class="ok-badge">Offre ✓</span>' if fs["offre_ok"] else '<span class="required-badge">Offre ✗</span>'
                st.markdown(f"{cv_tag} {offre_tag}", unsafe_allow_html=True)
            with col5:
                st.caption(f"🌐 {c.get('langue','—')} | ⏱ {c.get('duree','—')} min")
                st.markdown(f'<span class="schedule-badge">📅 {sched["label"][:40]}</span>',
                            unsafe_allow_html=True)
            with col6:
                if s == "termine":
                    if st.button("Rapport →", key=f"rep_{c['username']}"):
                        st.session_state.active_page = "rapports"
                        st.rerun()

                elif s == "configure" and c.get("access_token"):
                    lien = build_access_link(c["session_id"], c["access_token"])

                    # ── [CORRECTIF CAUSE 1] Bouton Rouvrir si fenêtre expirée ──
                    if sched["expired"]:
                        if st.button("🔄 Rouvrir +30 min", key=f"reopen_{c['username']}",
                                     help="Repousse la fenêtre sans changer le lien"):
                            new_dt = datetime.datetime.now().isoformat()
                            extend_session_window(c["username"], new_dt, 30)
                            st.success(f"✅ Fenêtre réouverte pour {c['name']} — même lien valide.")
                            st.rerun()
                    else:
                        if st.button("🔗 Lien", key=f"lnk_{c['username']}"):
                            st.session_state[f"show_link_{c['username']}"] = True
                        if st.session_state.get(f"show_link_{c['username']}"):
                            st.code(lien, language=None)

                elif s == "en_cours":
                    st.markdown('<span style="color:#059669;font-size:.8rem;">⏳ En attente fin…</span>',
                                unsafe_allow_html=True)
                else:
                    if st.button("Configurer", key=f"cfg_{c['username']}"):
                        st.session_state.active_page    = "config"
                        st.session_state.configure_user = c["username"]
                        st.rerun()

            # ── [CORRECTIF CAUSE 1] Fenêtre expirée : expander Modifier horaire ──
            if s == "configure" and sched["expired"] and c.get("access_token"):
                lien = build_access_link(c["session_id"], c["access_token"])
                with st.expander(f"🕐 Modifier la fenêtre de {c['name']} (sans changer le lien)"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        new_date = st.date_input(
                            "Nouvelle date", value=datetime.date.today(),
                            min_value=datetime.date.today(),
                            key=f"reopen_date_{c['username']}"
                        )
                    with col_b:
                        new_time = st.time_input(
                            "Heure", value=datetime.time(9, 0), step=300,
                            key=f"reopen_time_{c['username']}"
                        )
                    with col_c:
                        new_win = st.select_slider(
                            "Durée", [10, 15, 20, 30, 45, 60, 90, 120], value=30,
                            format_func=lambda x: f"{x} min",
                            key=f"reopen_win_{c['username']}"
                        )
                    if st.button("✅ Appliquer", key=f"apply_reopen_{c['username']}",
                                 use_container_width=True):
                        new_iso = datetime.datetime.combine(new_date, new_time).isoformat()
                        extend_session_window(c["username"], new_iso, new_win)
                        st.success("Fenêtre mise à jour. Le lien reste valide.")
                        st.code(lien, language=None)
                        st.rerun()

            st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — RAPPORTS
# ─────────────────────────────────────────────────────────────────────────────
def admin_rapports():
    st.markdown("## Rapports d'évaluation")
    reports = load_real_reports()
    if not reports:
        st.markdown('<div class="card"><div style="text-align:center;padding:2.5rem;">'
                    '<div style="font-size:2.5rem;">📭</div>'
                    '<div style="font-weight:600;margin-top:.5rem;">Aucun rapport disponible</div>'
                    '<div style="color:#6b7280;font-size:.85rem;margin-top:.3rem;">'
                    'Les rapports apparaissent après chaque entretien terminé.</div>'
                    '</div></div>', unsafe_allow_html=True)
        return

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1: filter_status = st.selectbox("Statut", ["Tous","Recommandé","En attente","Refusé"])
    with col_f2: filter_langue = st.selectbox("Langue", ["Toutes","Français","Anglais","Arabe"])
    with col_f3: filter_score  = st.slider("Score min", 0, 100, 0)

    filtered = [r for r in reports
                if (filter_status == "Tous" or r["status"] == filter_status)
                and (filter_langue == "Toutes" or r["langue"] == filter_langue)
                and r["score"] >= filter_score]

    st.caption(f"{len(filtered)} rapport(s)")
    st.markdown("---")
    selected = st.session_state.get("selected_report")

    if not selected:
        for r in filtered:
            c1,c2,c3,c4,c5,c6 = st.columns([3,1.5,1,1.5,1.5,1])
            with c1: st.markdown(f"**{r['candidate']}**  \n*{r['poste']}*")
            with c2: st.caption(r["date"][:10])
            with c3:
                color = get_score_color(r["score"])
                st.markdown(f"<span style='color:{color};font-size:1.2rem;font-weight:700;'>{r['score']}</span>/100",
                            unsafe_allow_html=True)
            with c4: st.markdown(verdict_badge(r["verdict_raw"]), unsafe_allow_html=True)
            with c5: st.markdown(emotion_badge(r["emotion_dominante"]), unsafe_allow_html=True)
            with c6:
                if st.button("Voir →", key=f"btn_{r['id']}"):
                    st.session_state.selected_report = r["id"]
                    if r.get("username"):
                        mark_report_read(r["username"])
                    st.rerun()
            st.divider()
    else:
        _render_report_detail(reports, selected)


def _render_report_detail(reports: list, selected: str):
    r = next((x for x in reports if x["id"] == selected), None)
    if not r:
        st.session_state.selected_report = None
        st.rerun()

    if st.button("← Retour"):
        st.session_state.selected_report = None
        st.rerun()

    raw = r["_raw"]
    st.markdown(f"###  {r['candidate']} — *{r['poste']}*")

    for col, (val, label, color) in zip(st.columns(4), [
        (f"{r['score']}/100",       "Score Total",         get_score_color(r["score"])),
        (f"{r['score_tech']}/100",  "Score Technique",     "#4f46e5"),
        (f"{r['score_beh']}/100",   "Score Comportemental","#059669"),
        (f"{r['score_comm']}/100",  "Score Communication", "#d97706"),
    ]):
        with col:
            st.markdown(f"""<div class="card-metric">
                <p class="metric-value" style="color:{color};font-size:1.5rem;">{val}</p>
                <p class="metric-label">{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("####  Scores par phase")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        phases = r.get("phases") or {}
        if phases:
            for pk, pdata in phases.items():
                if isinstance(pdata, dict):
                    obtained = pdata.get("obtained", 0)
                    max_pts  = pdata.get("max", 10)
                    avg      = pdata.get("avg_turn", 0)
                    n        = pdata.get("n_turns", 0)
                    pct      = round(obtained/max_pts*100) if max_pts else 0
                else:
                    obtained = int(pdata); max_pts = 10; pct = obtained; avg = n = 0
                label = PHASE_LABELS_FR.get(pk, pk)
                sc2   = get_score_color(pct)
                st.markdown(f"""<div style="margin-bottom:.9rem;">
                    <div style="display:flex;justify-content:space-between;font-size:.83rem;">
                        <span>{label}</span>
                        <span style="color:{sc2};font-weight:600;">{obtained}/{max_pts} pts</span>
                    </div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{pct}%;background:{sc2};"></div>
                    </div>
                    {f'<div style="font-size:.72rem;color:#8a93a8;">moy {avg:.1f}/10 · {n} tours</div>' if n else ''}
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("Scores par phase non disponibles.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🏷️ Verdict")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        verdict = raw.get("verdict") or raw.get("verdict_final") or "—"
        bg, txt = VERDICT_STYLES.get(verdict.upper(), ("#f1f3f8","#374151"))
        reco    = raw.get("recommendation_detail") or raw.get("recommandation_detail") or ""
        st.markdown(f"""<div style="background:{bg};border-radius:10px;padding:1rem;text-align:center;">
            <div style="font-size:1.3rem;font-weight:700;color:{txt};">{verdict}</div>
            <div style="font-size:.82rem;color:{txt};margin-top:.4rem;">{reco[:120] if reco else ''}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        vision = raw.get("analyse_emotion_vision") or {}
        if isinstance(vision, dict) and vision.get("disponible"):
            st.markdown("#### 😶 Émotions")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            emo  = r["emotion_dominante"]
            eico = {"détendu":"😊","neutre":"😐","anxieux":"😰","tendu":"😤"}.get(emo,"🙂")
            st.markdown(f"""<div style="text-align:center;padding:.5rem 0;">
                <div style="font-size:2rem;">{eico}</div>
                <div style="font-weight:600;">{emo.capitalize()}</div>
                <div style="font-size:.78rem;color:#8a93a8;">{vision.get('nb_frames',0)} frames | conf. {vision.get('confiance_globale',0)}%</div>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💬 Commentaire RH & Décision")
    users    = load_users()
    username = r.get("username","")
    u        = users.get(username, {})
    c1, c2   = st.columns([3, 1])
    with c1:
        comment = st.text_area("Commentaire", value=u.get("rh_comment",""), height=80)
    with c2:
        opts    = ["—","Recommandé","En attente","Refusé"]
        cur_dec = u.get("decision_rh","—") if u.get("decision_rh") in opts else "—"
        decision = st.selectbox("Décision RH", opts, index=opts.index(cur_dec))
        if st.button(" Enregistrer", type="primary", use_container_width=True):
            if username and username in users:
                users[username]["rh_comment"] = comment
                if decision != "—":
                    users[username]["decision_rh"] = decision
                    users[username].setdefault("notifications",[]).append({
                        "type":"report_ready",
                        "message":f"📊 Votre rapport est disponible. Décision : {decision}",
                        "at":datetime.datetime.now().isoformat(),"read":False,
                    })
                save_users(users)
                st.success("✅ Enregistré et candidat notifié.")
            else:
                st.warning("Username introuvable dans users.json.")


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
def admin_config():
    st.markdown("## ⚙️ Configuration entretien")
    users     = load_users()
    candidats = {u: _default_fields(d) for u, d in users.items() if d.get("role") == "user"}

    if not candidats:
        st.info("Aucun candidat inscrit.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👤 Candidat & Documents")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        candidate_options  = {f"{d['name']} (@{u})": u for u, d in candidats.items()}
        selected_label     = st.selectbox("Candidat", list(candidate_options.keys()))
        selected_uname     = candidate_options[selected_label]
        selected_user      = candidats[selected_uname]

        st_map = {"en_attente":("⏳","#d97706"),"configure":("⚙️","#4f46e5"),
                  "en_cours":("🎤","#059669"),"termine":("✅","#6b7280")}
        ico, col = st_map.get(selected_user.get("session_status","en_attente"),("—","#6b7280"))
        st.markdown(f'<div style="font-size:.83rem;margin:.5rem 0;">Statut : '
                    f'<span style="color:{col};font-weight:600;">{ico} '
                    f'{selected_user.get("session_status","—").replace("_"," ").capitalize()}'
                    f'</span></div>', unsafe_allow_html=True)

        poste = st.text_input("Poste visé *", value=selected_user.get("poste") or "",
                              placeholder="Ingénieur ML")

        fs = _candidate_files_status(selected_user)
        col_cv, col_offre = st.columns(2)
        with col_cv:
            if fs["cv_ok"]:
                st.markdown(f'<span class="ok-badge">✓ CV : {fs["cv_file"]}</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="required-badge">⚠️ CV manquant — requis</span>',
                            unsafe_allow_html=True)
        with col_offre:
            if fs["offre_ok"]:
                st.markdown(f'<span class="ok-badge">✓ Offre : {fs["offre_file"]}</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="required-badge">⚠️ Offre manquante — requise</span>',
                            unsafe_allow_html=True)

        st.markdown("---")

        cv_f = st.file_uploader(
            "📄 CV du candidat * (PDF/DOCX — obligatoire)",
            type=["pdf","docx"], key=f"cv_upload_{selected_uname}"
        )
        if cv_f:
            cv_bytes    = cv_f.read()
            cv_ext      = Path(cv_f.name).suffix.lower()
            cv_filename = f"cv_{selected_uname}{cv_ext}"
            (CV_DIR / cv_filename).write_bytes(cv_bytes)
            (DATA_DIR / f"cv{cv_ext}").write_bytes(cv_bytes)
            _u_tmp = load_users()
            if selected_uname in _u_tmp:
                _u_tmp[selected_uname]["cv_file"] = cv_filename
                save_users(_u_tmp)
                selected_user = _default_fields(_u_tmp[selected_uname])
            st.success(f"✅ CV enregistré : {cv_f.name}")

        of_f = st.file_uploader(
            "Offre d'emploi * (PDF/DOCX — obligatoire)",
            type=["pdf","docx"], key=f"offre_upload_{selected_uname}"
        )
        if of_f:
            offre_bytes    = of_f.read()
            offre_ext      = Path(of_f.name).suffix.lower()
            offre_filename = f"offre_{selected_uname}{offre_ext}"
            (CV_DIR / offre_filename).write_bytes(offre_bytes)
            (DATA_DIR / f"offre{offre_ext}").write_bytes(offre_bytes)
            _u_tmp = load_users()
            if selected_uname in _u_tmp:
                _u_tmp[selected_uname]["offre_file"] = offre_filename
                save_users(_u_tmp)
                selected_user = _default_fields(_u_tmp[selected_uname])
            st.success(f"✅ Offre enregistrée : {of_f.name}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("###  Paramètres entretien")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        langue = st.selectbox("Langue *", ["Français","Anglais","Arabe"])
        duree  = st.slider("Durée (min) *", 10, 60,
                           int(selected_user.get("duree") or 30), step=5)
        st.select_slider("Difficulté", ["Débutant","Intermédiaire","Senior","Expert"],
                         value="Intermédiaire")
        st.toggle("Analyse émotionnelle", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Planification ────────────────────────────────────────────────────
        st.markdown("###  Planification")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        use_schedule     = st.toggle("Fixer un horaire précis",
                                     value=bool(selected_user.get("scheduled_at")))
        scheduled_at_iso = None
        window_min       = 30

        if use_schedule:
            col_d, col_h = st.columns(2)
            with col_d:
                default_date = datetime.date.today() + datetime.timedelta(days=1)
                if selected_user.get("scheduled_at"):
                    try:
                        default_date = datetime.datetime.fromisoformat(
                            selected_user["scheduled_at"]).date()
                    except Exception:
                        pass
                today = datetime.date.today()
                if default_date < today:
                    default_date = today
                sched_date = st.date_input("📅 Date", value=default_date,
                                           min_value=today)
            with col_h:
                default_time = datetime.time(9, 0)
                if selected_user.get("scheduled_at"):
                    try:
                        default_time = datetime.datetime.fromisoformat(
                            selected_user["scheduled_at"]).time()
                    except Exception:
                        pass
                sched_time = st.time_input("🕐 Heure", value=default_time, step=300)

            window_min = st.select_slider(
                "Fenêtre d'accès",
                options=[10, 15, 20, 30, 45, 60, 90, 120],
                value=int(selected_user.get("scheduled_window_min", 30)),
                format_func=lambda x: f"{x} min",
            )
            scheduled_dt     = datetime.datetime.combine(sched_date, sched_time)
            scheduled_at_iso = scheduled_dt.isoformat()
            st.markdown(f"""<div style="background:#eff6ff;border-radius:8px;padding:.7rem 1rem;
                font-size:.85rem;color:#1e40af;margin-top:.5rem;">
                📌 Accès du <strong>{scheduled_dt.strftime('%d/%m/%Y à %H:%M')}</strong>
                au <strong>{(scheduled_dt + datetime.timedelta(minutes=window_min)).strftime('%H:%M')}</strong>
            </div>""", unsafe_allow_html=True)
        else:
            st.caption("Pas d'horaire — le lien sera actif immédiatement.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── URL de base [CORRECTIF CAUSE 3] ──────────────────────────────────
        st.markdown("### 🌐 Serveur")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        saved_url = load_base_url()
        col_url, col_detect = st.columns([3, 1])
        with col_url:
            base_url = st.text_input(
                "URL de base FastAPI",
                value=saved_url,
                placeholder="https://mon-serveur.com",
                help="Cette URL est intégrée dans tous les liens candidats. "
                     "Elle est sauvegardée automatiquement entre les sessions."
            )
        with col_detect:
            st.write("")
            if st.button("🔍 IP locale", use_container_width=True,
                         help="Détecte l'IP de cette machine sur le réseau local"):
                st.session_state["detected_url"] = detect_local_ip()
                st.rerun()

        if "detected_url" in st.session_state:
            detected = st.session_state["detected_url"]
            st.info(f"IP locale détectée : `{detected}`")
            col_apply, col_dismiss = st.columns(2)
            with col_apply:
                if st.button(f"✅ Utiliser cette IP", use_container_width=True):
                    save_base_url(detected)
                    del st.session_state["detected_url"]
                    st.success("URL sauvegardée.")
                    st.rerun()
            with col_dismiss:
                if st.button("✕ Ignorer", use_container_width=True):
                    del st.session_state["detected_url"]
                    st.rerun()

        if base_url.strip() and base_url.strip() != saved_url:
            if st.button(" Sauvegarder cette URL", use_container_width=True):
                save_base_url(base_url.strip())
                st.success("✅ URL sauvegardée — utilisée pour tous les prochains liens.")
                st.rerun()

        if "localhost" in base_url:
            st.warning("⚠️ `localhost` n'est accessible que sur cette machine. "
                       "Les candidats sur d'autres postes ne pourront pas ouvrir le lien. "
                       "Utilisez **🔍 IP locale** ou renseignez l'URL publique du serveur.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Générer le lien ─────────────────────────────────────────────────
        st.markdown("###  Générer le lien")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        fs_current = _candidate_files_status(selected_user)
        can_generate = fs_current["both_ok"] and bool(poste.strip())

        if not fs_current["cv_ok"]:
            st.error("❌ CV manquant — uploadez le CV du candidat avant de continuer.")
        if not fs_current["offre_ok"]:
            st.error("❌ Offre manquante — uploadez l'offre d'emploi avant de continuer.")
        if not poste.strip():
            st.warning("⚠️ Renseignez le poste visé.")

        # ── [CORRECTIF CAUSE 2] Avertissement si token existant ──────────────
        existing_token = selected_user.get("access_token")
        existing_sid   = selected_user.get("session_id")
        existing_status = selected_user.get("session_status", "en_attente")

        if existing_token and existing_status not in ("en_attente",):
            st.warning(
                "⚠️ **Ce candidat a déjà un lien actif.**  \n"
                "Cliquer sur **▶ Créer** va générer un **nouveau token** et "
                "**invalider l'ancien lien** envoyé au candidat.  \n"
                "Utilisez **'Modifier la fenêtre'** si vous souhaitez juste décaler l'horaire."
            )
            lien_actuel = build_access_link(existing_sid, existing_token, base_url)
            sched_actuel = schedule_status(selected_user)
            with st.expander("🔗 Lien actuellement actif (cliquez pour voir)"):
                st.code(lien_actuel, language=None)
                st.caption(f"Statut fenêtre : {sched_actuel['label']}")

                # ── Modifier la fenêtre sans changer le lien ──────────────
                st.markdown("** Modifier la fenêtre sans changer le lien**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    reo_date = st.date_input(
                        "Nouvelle date", value=datetime.date.today(),
                        min_value=datetime.date.today(),
                        key=f"reo_date_{selected_uname}"
                    )
                with col_b:
                    reo_time = st.time_input(
                        "Heure", value=datetime.time(9, 0), step=300,
                        key=f"reo_time_{selected_uname}"
                    )
                with col_c:
                    reo_win = st.select_slider(
                        "Durée", [10, 15, 20, 30, 45, 60, 90, 120], value=30,
                        format_func=lambda x: f"{x} min",
                        key=f"reo_win_{selected_uname}"
                    )
                if st.button("✅ Appliquer sans changer le lien",
                             key=f"apply_reo_{selected_uname}",
                             use_container_width=True):
                    new_iso = datetime.datetime.combine(reo_date, reo_time).isoformat()
                    extend_session_window(selected_uname, new_iso, reo_win)
                    st.success("✅ Fenêtre mise à jour. Le lien existant reste valide.")
                    st.rerun()

        if st.button("▶ Créer et générer le lien", use_container_width=True,
                     type="primary", disabled=not can_generate):
            import requests as _req

            _copy_candidate_files(selected_uname)

            try:
                resp = _req.post(
                    f"{base_url}/session/start",
                    data={"lang": langue, "duration": duree},
                    timeout=60,
                )
                if resp.ok:
                    sid = resp.json().get("session_id", "")
                else:
                    sid = str(uuid.uuid4())[:12].upper()
                    st.warning(f"FastAPI réponse {resp.status_code} — ID local généré.")
            except Exception as e:
                st.warning(f"FastAPI non joignable ({e}) — ID local généré.")
                sid = str(uuid.uuid4())[:12].upper()

            mf = DATA_DIR / "session_users.json"
            m  = json.loads(mf.read_text()) if mf.exists() else {}
            m[sid] = selected_uname
            mf.write_text(json.dumps(m, indent=2))

            token = assign_session_to_user(
                selected_uname, sid, poste.strip(), langue, duree,
                scheduled_at=scheduled_at_iso, window_min=window_min,
            )

            # Sauvegarder l'URL utilisée
            if base_url.strip():
                save_base_url(base_url.strip())

            lien = build_access_link(sid, token, base_url)
            st.success(f"✅ Session créée pour **{selected_user['name']}**")
            st.session_state[f"generated_link_{selected_uname}"] = lien
            st.rerun()

        # Afficher le lien généré
        lien_key = f"generated_link_{selected_uname}"
        if lien_key in st.session_state:
            lien = st.session_state[lien_key]
            st.markdown("---")
            st.markdown("#### 🔗 Lien à envoyer au candidat")
            st.markdown("""<div style="background:#fafffe;border:1.5px solid #4f46e5;
                 border-radius:10px;padding:1rem 1.2rem;margin-top:.5rem;">
                <div style="font-size:.78rem;color:#6b7280;margin-bottom:.4rem;">
                    ✅ Copiez ce lien et envoyez-le au candidat (email, SMS, WhatsApp…)
                </div>
            </div>""", unsafe_allow_html=True)
            st.code(lien, language=None)
            if scheduled_at_iso:
                dt_s = datetime.datetime.fromisoformat(scheduled_at_iso)
                dt_e = dt_s + datetime.timedelta(minutes=window_min)
                st.info(f" Actif du **{dt_s.strftime('%d/%m/%Y à %H:%M')}** "
                        f"au **{dt_e.strftime('%H:%M')}** ({window_min} min).")
            else:
                st.info(" Ce lien est actif immédiatement et sans limite de temps.")

        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — UTILISATEURS
# ─────────────────────────────────────────────────────────────────────────────
def admin_utilisateurs():
    st.markdown("## 👥 Gestion des utilisateurs")
    preselect     = st.session_state.pop("configure_user", None)
    tab_list, tab_config, tab_hist = st.tabs(["📋 Candidats","⚙️ Configurer","🕐 Historique"])

    with tab_list:
        users = load_users()
        cols  = st.columns([3,2,1.5,1.5,1.5,2.5])
        for col, h in zip(cols, ["Utilisateur","Email","Rôle","Statut","Poste","Actions"]):
            col.markdown(f"**{h}**")
        st.divider()
        for uname, u in users.items():
            u = _default_fields(u)
            c1,c2,c3,c4,c5,c6 = st.columns([3,2,1.5,1.5,1.5,2.5])
            initials  = "".join(w[0].upper() for w in u["name"].split()[:2])
            blocked   = u.get("is_blocked", False)
            role_html = badge("Admin","badge-admin") if u["role"]=="admin" else badge("Candidat","badge-user")
            ss   = u.get("session_status","—") if u["role"]=="user" else "—"
            smap = {"en_attente":"badge-warn","configure":"badge-info",
                    "en_cours":"badge-purple","termine":"badge-ok"}
            sess_html = (badge(ss.replace("_"," ").capitalize(), smap.get(ss,"badge-info"))
                         if u["role"]=="user" else "—")
            with c1:
                bl_tag = f' {badge("🚫 Bloqué","badge-blocked")}' if blocked else ""
                st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;">
                    <div class="avatar" style="background:{'#4f46e5' if u['role']=='admin' else '#0ea5e9'};
                         color:#fff;width:32px;height:32px;font-size:.75rem;">{initials}</div>
                    <div>
                        <div style="font-weight:600;font-size:.88rem;">{u['name']}{bl_tag}</div>
                        <div style="font-size:.75rem;color:#8a93a8;">@{uname}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with c2: st.caption(u["email"])
            with c3: st.markdown(role_html, unsafe_allow_html=True)
            with c4: st.markdown(sess_html, unsafe_allow_html=True)
            with c5: st.caption(u.get("poste","—") or "—")
            with c6:
                b1,b2,b3 = st.columns(3)
                with b1:
                    if u["role"]=="user" and st.button("⚙️",key=f"cfg_{uname}",help="Configurer"):
                        st.session_state.configure_user_tab = uname
                        st.rerun()
                with b2:
                    if u["role"]=="user":
                        lbl = "🔓" if blocked else "🚫"
                        if st.button(lbl, key=f"blk_{uname}"):
                            block_user(uname, not blocked); st.rerun()
                with b3:
                    if uname!="admin" and st.button("🗑",key=f"del_{uname}"):
                        fresh = load_users(); del fresh[uname]; save_users(fresh); st.rerun()
            st.divider()

        st.markdown("### ➕ Créer un compte")
        with st.expander("Nouveau compte", expanded=False):
            with st.form("admin_create_user"):
                n_name  = st.text_input("Nom complet")
                n_uname = st.text_input("Nom d'utilisateur")
                n_email = st.text_input("Email")
                n_pw    = st.text_input("Mot de passe", type="password")
                n_role  = st.selectbox("Rôle", ["user","admin"])
                if st.form_submit_button("Créer"):
                    if all([n_name,n_uname,n_email,n_pw]):
                        ok, msg = register_user(n_uname,n_name,n_email,n_pw)
                        if ok:
                            if n_role=="admin":
                                u2=load_users(); u2[n_uname]["role"]="admin"; save_users(u2)
                            st.success(f"✅ Compte {n_uname} créé."); st.rerun()
                        else: st.error(msg)
                    else: st.warning("Tous les champs sont requis.")

    with tab_config:
        candidates = get_candidates()
        if not candidates:
            st.info("Aucun candidat inscrit.")
        else:
            opts = {f"{c['name']} (@{c['username']})": c["username"] for c in candidates}
            default_idx = 0
            presel = preselect or st.session_state.pop("configure_user_tab", None)
            if presel:
                for i, uname in enumerate(opts.values()):
                    if uname == presel: default_idx = i; break

            sel_label    = st.selectbox("Candidat", list(opts.keys()), index=default_idx)
            sel_username = opts[sel_label]
            sel_user     = next(c for c in candidates if c["username"]==sel_username)
            fs           = _candidate_files_status(sel_user)

            c1, c2 = st.columns(2)
            with c1:
                poste  = st.text_input("Poste *", value=sel_user.get("poste","") or "")
                langue = st.selectbox("Langue *", ["Français","Anglais","Arabe"])
                duree  = st.slider("Durée (min) *", 10, 60,
                                   int(sel_user.get("duree") or 30), step=5)
            with c2:
                col_a, col_b = st.columns(2)
                with col_a:
                    if fs["cv_ok"]:
                        st.markdown(f'<span class="ok-badge">CV ✓ {fs["cv_file"]}</span>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="required-badge">CV ✗ manquant</span>',
                                    unsafe_allow_html=True)
                with col_b:
                    if fs["offre_ok"]:
                        st.markdown(f'<span class="ok-badge">Offre ✓ {fs["offre_file"]}</span>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="required-badge">Offre ✗ manquante</span>',
                                    unsafe_allow_html=True)

                cv_f = st.file_uploader("CV * (obligatoire)", type=["pdf","docx"],
                                        key=f"cv_{sel_username}")
                if cv_f:
                    cv_bytes = cv_f.read()
                    cv_ext   = Path(cv_f.name).suffix.lower()
                    cv_fn    = f"cv_{sel_username}{cv_ext}"
                    (CV_DIR / cv_fn).write_bytes(cv_bytes)
                    _u_tmp = load_users()
                    if sel_username in _u_tmp:
                        _u_tmp[sel_username]["cv_file"] = cv_fn
                        save_users(_u_tmp)
                    st.success("CV ok.")

                of_f = st.file_uploader("Offre * (obligatoire)", type=["pdf","docx"],
                                        key=f"offre_{sel_username}")
                if of_f:
                    offre_bytes = of_f.read()
                    offre_ext   = Path(of_f.name).suffix.lower()
                    offre_fn    = f"offre_{sel_username}{offre_ext}"
                    (CV_DIR / offre_fn).write_bytes(offre_bytes)
                    _u_tmp = load_users()
                    if sel_username in _u_tmp:
                        _u_tmp[sel_username]["offre_file"] = offre_fn
                        save_users(_u_tmp)
                    st.success("Offre ok.")

            fs2 = _candidate_files_status(_default_fields(load_users().get(sel_username, {})))
            can_assign = fs2["both_ok"] and bool(poste.strip())

            if not fs2["cv_ok"]:
                st.error("❌ CV manquant.")
            if not fs2["offre_ok"]:
                st.error("❌ Offre manquante.")

            if st.button("▶ Assigner la session", type="primary",
                         use_container_width=True, disabled=not can_assign):
                import requests as _req
                _copy_candidate_files(sel_username)
                base_url = load_base_url()
                try:
                    resp = _req.post(f"{base_url}/session/start",
                                     data={"lang":langue,"duration":duree}, timeout=60)
                    sid = resp.json().get("session_id","") if resp.ok else ""
                except Exception:
                    sid = str(uuid.uuid4())[:12].upper()
                    st.warning("FastAPI non joignable — ID local généré.")
                mf = DATA_DIR/"session_users.json"
                m  = json.loads(mf.read_text()) if mf.exists() else {}
                m[sid] = sel_username
                mf.write_text(json.dumps(m,indent=2))
                token = assign_session_to_user(sel_username, sid, poste.strip(), langue, duree)
                lien  = build_access_link(sid, token, base_url)
                st.success(f"✅ Session assignée à {sel_user['name']}")
                st.code(lien, language=None)
                st.rerun()

    with tab_hist:
        st.markdown("### 🕐 Historique des connexions")
        users = load_users()
        for uname, u in users.items():
            hist = u.get("login_history",[])
            if not hist: continue
            with st.expander(f"**{u['name']}** (@{uname}) — {len(hist)} connexion(s)"):
                for entry in reversed(hist[-10:]):
                    at  = entry.get("at","")[:16].replace("T"," ")
                    ico = "✅" if entry.get("ok") else "❌"
                    st.markdown(f"<div style='font-size:.82rem;'>- {ico} {at}</div>",
                                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDAT — ACCUEIL
# ─────────────────────────────────────────────────────────────────────────────
def user_accueil(user: dict, username: str):
    st.markdown(f"## 👋 Bienvenue, {user['name'].split()[0]} !")
    users  = load_users()
    u      = _default_fields(users.get(username, user))
    status = u.get("session_status","en_attente")
    poste  = u.get("poste") or "—"
    langue = u.get("langue") or "—"
    duree  = u.get("duree") or "—"

    notif_session = [n for n in u.get("notifications",[])
                     if n.get("type")=="session_ready" and not n.get("read")]
    if notif_session:
        sched = schedule_status(u)
        st.markdown(f"""<div class="notif-banner">
            <span style="font-size:1.8rem;">🎤</span>
            <div>
                <strong>Votre session d'entretien est configurée !</strong><br>
                <span style="font-size:.85rem;">{notif_session[-1].get('message','')}</span><br>
                <span style="font-size:.82rem;opacity:.85;">📅 {sched['label']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📌 Conseils")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for ico, tip in [
            ("🎙️","Parlez clairement et distinctement"),
            ("💡","Structurez vos réponses (Situation → Action → Résultat)"),
            ("📸","Regardez la caméra pour la meilleure analyse"),
            ("⏱️","Respectez les temps impartis par phase"),
            ("🌐","Répondez dans la langue configurée par le recruteur"),
        ]:
            st.markdown(f'<div style="display:flex;gap:10px;margin-bottom:.7rem;"><span>{ico}</span>'
                        f'<span style="font-size:.85rem;color:#374151;">{tip}</span></div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("### ℹ️ Ma session")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sc    = {"en_attente":"#d97706","configure":"#4f46e5","en_cours":"#059669","termine":"#6b7280"}
        sched = schedule_status(u)
        scheduled_at = u.get("scheduled_at")
        sched_str = ""
        if scheduled_at:
            try:
                dt_s = datetime.datetime.fromisoformat(scheduled_at)
                dt_e = dt_s + datetime.timedelta(minutes=int(u.get("scheduled_window_min",30)))
                sched_str = (f'<div>📅 Planifié le : <strong>{dt_s.strftime("%d/%m/%Y à %H:%M")}</strong></div>'
                             f'<div>🔓 Fenêtre : <strong>{dt_s.strftime("%H:%M")} → {dt_e.strftime("%H:%M")}</strong></div>')
            except Exception:
                pass
        st.markdown(f"""<div style="font-size:.85rem;line-height:2.1;">
            <div>📌 Poste : <strong>{poste}</strong></div>
            <div>🌐 Langue : <strong>{langue}</strong></div>
            <div>⏱ Durée : <strong>{duree} min</strong></div>
            <div>📊 Statut : <strong style="color:{sc.get(status,'#374151')};">
                {status.replace('_',' ').capitalize()}</strong></div>
            {sched_str}
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDAT — MON ENTRETIEN
# ─────────────────────────────────────────────────────────────────────────────
def user_entretien(username: str):
    st.markdown("## 🎤 Mon entretien")
    u      = _default_fields(load_users().get(username, {}))
    sid    = u.get("session_id")
    status = u.get("session_status","en_attente")
    token  = u.get("access_token")
    sched  = schedule_status(u)

    if not sid or status == "en_attente":
        st.markdown('<div class="card"><div style="text-align:center;padding:2rem;">'
                    '<div style="font-size:3rem;">⏳</div>'
                    '<div style="font-size:1.1rem;font-weight:600;margin-top:.5rem;">'
                    'Session non encore configurée</div>'
                    '<div style="font-size:.85rem;color:#6b7280;margin-top:.4rem;">'
                    'Le recruteur configurera votre entretien prochainement.</div>'
                    '</div></div>', unsafe_allow_html=True)
        return

    if not sched["ok"]:
        if sched["future"]:
            st.markdown(f"""<div style="background:#fef3c7;border:1px solid #fcd34d;border-radius:10px;
                padding:1rem 1.2rem;margin-bottom:1rem;">
                <div style="font-weight:700;color:#92400e;"> Votre entretien est planifié</div>
                <div style="color:#92400e;font-size:.88rem;margin-top:.3rem;">{sched['label']}</div>
            </div>""", unsafe_allow_html=True)
        elif sched["expired"]:
            st.error("⛔ La fenêtre d'accès a expiré. Contactez votre recruteur.")
            return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    poste  = u.get("poste","—"); langue = u.get("langue","—"); duree = u.get("duree","—")
    st.markdown(f"""<div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:1.2rem;">
        <div><span style="color:#8a93a8;font-size:.8rem;">📌 Poste</span><br><strong>{poste}</strong></div>
        <div><span style="color:#8a93a8;font-size:.8rem;">🌐 Langue</span><br><strong>{langue}</strong></div>
        <div><span style="color:#8a93a8;font-size:.8rem;">⏱ Durée</span><br><strong>{duree} min</strong></div>
    </div>""", unsafe_allow_html=True)

    if sched["ok"]:
        base_url = load_base_url()
        url = f"{base_url}/interview/{token}" if token else f"{base_url}/?session={sid}"
        st.markdown(f"**🔗 Lien d'entretien :** [{url}]({url})")
        st.code(url, language=None)
        if st.button("▶ Rejoindre l'entretien", type="primary", use_container_width=True):
            st.markdown(f'<meta http-equiv="refresh" content="0;url={url}">', unsafe_allow_html=True)
    else:
        st.button("▶ Rejoindre l'entretien", type="primary", use_container_width=True, disabled=True)
        st.caption(f"🔒 Disponible à partir de l'heure planifiée — {sched['label']}")

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDAT — MES RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────
def user_resultats(user: dict, username: str):
    st.markdown("##  Mes résultats")
    u      = _default_fields(load_users().get(username, {}))
    status = u.get("session_status", "en_attente")

    if status not in ("termine",):
        st.markdown('<div class="card"><div style="text-align:center;padding:2.5rem;">'
                    '<div style="font-size:3rem;">⏳</div>'
                    '<div style="font-weight:600;margin-top:.5rem;">Entretien pas encore réalisé</div>'
                    '</div></div>', unsafe_allow_html=True)
        return

    st.markdown("""
    <div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:14px;
         padding:1.5rem 2rem;text-align:center;margin-bottom:1.5rem;">
        <div style="font-size:2.5rem;margin-bottom:.5rem;">✅</div>
        <div style="font-size:1.2rem;font-weight:700;color:#166534;">Entretien réalisé avec succès</div>
        <div style="font-size:.88rem;color:#166534;margin-top:.4rem;max-width:420px;margin-inline:auto;">
            Votre évaluation a été transmise à l'équipe RH.
            Vous serez contacté(e) pour la suite du processus.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info(" Les détails de votre évaluation sont disponibles auprès de votre recruteur.")


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDAT — MON PROFIL
# ─────────────────────────────────────────────────────────────────────────────
def user_profil(username: str):
    st.markdown("## 👤 Mon profil")
    users = load_users()
    u     = _default_fields(users.get(username, {}))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###  Modifier mes informations")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.form("profil_form"):
            new_name  = st.text_input("Nom complet", value=u.get("name",""))
            new_email = st.text_input("Email",        value=u.get("email",""))
            st.markdown("---")
            st.markdown("**Changer le mot de passe**")
            old_pw  = st.text_input("Mot de passe actuel", type="password")
            new_pw1 = st.text_input("Nouveau mot de passe", type="password")
            new_pw2 = st.text_input("Confirmer",            type="password")
            submit  = st.form_submit_button("💾 Enregistrer", use_container_width=True)
        if submit:
            errors = []
            fresh  = load_users()
            ue     = fresh.get(username, {})
            if not new_name.strip():  errors.append("Le nom ne peut pas être vide.")
            if not new_email.strip(): errors.append("L'email ne peut pas être vide.")
            if new_email != ue.get("email"):
                if any(uu.get("email")==new_email for un,uu in fresh.items() if un!=username):
                    errors.append("Email déjà utilisé.")
            if old_pw or new_pw1:
                if ue.get("password") != _hash(old_pw):
                    errors.append("Mot de passe actuel incorrect.")
                elif new_pw1 != new_pw2:
                    errors.append("Mots de passe différents.")
                elif len(new_pw1) < 6:
                    errors.append("Trop court (min. 6 car.).")
            if errors:
                for e in errors: st.error(e)
            else:
                ue["name"]  = new_name.strip()
                ue["email"] = new_email.strip()
                if new_pw1: ue["password"] = _hash(new_pw1)
                fresh[username] = ue
                save_users(fresh)
                st.session_state.user = ue
                st.success("✅ Profil mis à jour.")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("###  Informations")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        last    = (u.get("last_login","") or "")[:16].replace("T"," ") or "Jamais"
        nb_conn = len(u.get("login_history",[]))
        st.markdown(f"""<div style="font-size:.88rem;line-height:2.2;">
            <div>📅 Inscrit le : <strong>{u.get('created_at','')[:10] or '—'}</strong></div>
            <div>🔐 Dernière connexion : <strong>{last}</strong></div>
            <div>🔢 Connexions : <strong>{nb_conn}</strong></div>
            <div>📊 Statut session : <strong>{u.get('session_status','—').replace('_',' ').capitalize()}</strong></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "active_page" not in st.session_state:
        st.session_state.active_page = "accueil"

    if "user" not in st.session_state:
        if st.session_state.page == "register":
            page_register()
        else:
            page_login()
        return

    user     = st.session_state.user
    username = st.session_state.get("username","")
    role     = user.get("role","user")
    active   = render_sidebar(user, role, username)

    if role == "admin":
        if active == "accueil":           admin_dashboard()
        elif active == "entretiens":      admin_entretiens()
        elif active == "rapports":        admin_rapports()
        elif active == "config":          admin_config()
        elif active == "utilisateurs":    admin_utilisateurs()
        elif active == "notifications":   page_notifications(username)
    else:
        if active == "accueil":           user_accueil(user, username)
        elif active == "notifications":   page_notifications(username)
        elif active == "entretien":       user_entretien(username)
        elif active == "resultats":       user_resultats(user, username)
        elif active == "profil":          user_profil(username)


if __name__ == "__main__":
    main()