import streamlit as st
import threading
import time
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import speech
import random
import audio
import string
import os
import socket
import requests

# Configuration de la page Streamlit
st.set_page_config(
    page_title="OpenPronounce - Pronunciation Analysis",
    page_icon="🎤",
    layout="wide"
)

# Créer l'application FastAPI
app = FastAPI()

# Ajouter CORS pour permettre les requêtes depuis Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)

def upload_webp(file):
    tempname_random = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    destination = f'/tmp/{tempname_random}.webm'

    with open(destination, 'wb') as buffer:
        buffer.write(file.file.read())

    return audio.webp2wav(destination)

@app.post("/pronunciation")
async def api_analyze_pronunciation(file: UploadFile = File(...), expected_text: str = Form(...)):
    wav_file = upload_webp(file)
    try:
        sound = audio.load(wav_file)
        return speech.compare_audio_with_text(sound, expected_text)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail='Something went wrong')

@app.post("/speech2text")
async def api_speech2text(file: UploadFile = File(...)):
    wav_file = upload_webp(file)
    try:
        sound = audio.load(wav_file)
        return {
            "transcript": speech.transcribe(sound),
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail='Something went wrong')
    
@app.post("/phonemes")
async def api_phonemes(text: str = Form(...)):
    try:
        phonemes, words = speech.get_phonemes_with_word_mapping(text)
        return {
            "phonemes": phonemes,
            "words": list(words.values())
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail='Something went wrong')

@app.post("/tts")
async def api_tts(text: str = Form(...)):
    try:
        filename = audio.text2speech(text)
        return FileResponse(filename, media_type="audio/wav")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail='Something went wrong')

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )

def find_free_port():
    """Trouve un port libre"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def run_server(port):
    """Lance le serveur FastAPI"""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server.run()

def check_server_running(url, max_retries=10):
    """Vérifie si le serveur est en cours d'exécution"""
    for _ in range(max_retries):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except:
            time.sleep(0.5)
    return False

# Initialisation du serveur
if 'server_port' not in st.session_state:
    port = find_free_port()
    st.session_state.server_port = port
    st.session_state.server_started = False
    st.session_state.server_ready = False

# Démarrer le serveur si ce n'est pas déjà fait
if not st.session_state.server_started:
    server_thread = threading.Thread(target=run_server, args=(st.session_state.server_port,), daemon=True)
    server_thread.start()
    st.session_state.server_started = True

# Vérifier que le serveur est prêt
if st.session_state.server_started and not st.session_state.server_ready:
    server_url = f"http://127.0.0.1:{st.session_state.server_port}"
    if check_server_running(server_url):
        st.session_state.server_ready = True

# Afficher la page HTML dans un iframe
if st.session_state.server_ready:
    server_url = f"http://127.0.0.1:{st.session_state.server_port}"
    
    st.markdown(f"""
    <div style="width: 100%; height: 1200px; border: none;">
        <iframe src="{server_url}" width="100%" height="1200px" frameborder="0" 
                allow="microphone; camera; autoplay; encrypted-media"
                style="border-radius: 10px; border: 2px solid #e5e7eb;"></iframe>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("⏳ Démarrage du serveur en cours... Veuillez patienter quelques secondes.")
    st.rerun()

# Sidebar avec informations
with st.sidebar:
    st.header("ℹ️ Informations")
    if st.session_state.server_ready:
        st.success(f"✅ Serveur actif sur le port {st.session_state.server_port}")
    else:
        st.warning("⏳ Le serveur démarre...")
    
    st.markdown("---")
    st.markdown("### 📚 Liens")
    st.markdown("[📖 Article de blog](https://blog.lepine.pro/en/ai-wav2vec-pronunciation-vectorization/)")
    st.markdown("[🐙 GitHub](https://github.com/Halleck45/OpenPronounce)")
    
    st.markdown("---")
    st.markdown("### 🔧 Dépannage")
    st.markdown("""
    Si l'application ne se charge pas :
    1. Attendez quelques secondes
    2. Rafraîchissez la page
    3. Vérifiez que les ports ne sont pas bloqués
    
    **Pour le microphone :**
    - Cliquez sur l'icône de cadenas dans la barre d'adresse
    - Autorisez l'accès au microphone
    - Si cela ne fonctionne pas, utilisez l'upload de fichier audio
    """)
