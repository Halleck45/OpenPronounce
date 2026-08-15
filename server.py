"""FastAPI server exposing the OpenPronounce web UI and JSON API.

Run with: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from openpronounce import __version__, audio, speech
from openpronounce.languages import DEFAULT_LANGUAGE, LANGUAGES, get_language

logger = logging.getLogger("openpronounce.server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="OpenPronounce",
    description="Phoneme-level pronunciation assessment (Wav2Vec2 + DTW). English by default, see /languages.",
    version=__version__,
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def _save_upload_as_wav(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".webm"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="openpronounce-upload-")
    with os.fdopen(fd, "wb") as buffer:
        buffer.write(upload.file.read())
    try:
        return audio.webm2wav(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _validate_lang(lang: str) -> str:
    try:
        return get_language(lang).code
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/pronunciation")
async def api_analyze_pronunciation(file: UploadFile = File(...), expected_text: str = Form(...),
                                    lang: str = Form(DEFAULT_LANGUAGE)):
    """Score ``file`` against ``expected_text`` in ``lang``. Returns the full analysis (score, errors, prosody)."""
    lang = _validate_lang(lang)
    try:
        wav_file = _save_upload_as_wav(file)
        sound = audio.load(wav_file)
        return speech.compare_audio_with_text(sound, expected_text, lang=lang)
    except Exception:
        logger.exception("pronunciation analysis failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/speech2text")
async def api_speech2text(file: UploadFile = File(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Transcribe ``file`` with the Wav2Vec2 model of ``lang``."""
    lang = _validate_lang(lang)
    try:
        wav_file = _save_upload_as_wav(file)
        sound = audio.load(wav_file)
        return {"transcript": speech.transcribe(sound, lang)}
    except Exception:
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/phonemes")
async def api_phonemes(text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Return the IPA phonemes of ``text`` in ``lang`` and the word each phoneme belongs to."""
    lang = _validate_lang(lang)
    try:
        phonemes, words = speech.get_phonemes_with_word_mapping(text, lang)
        return {"phonemes": phonemes, "words": list(words.values())}
    except Exception:
        logger.exception("phonemization failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/tts")
async def api_tts(text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Return a 16 kHz wav reference pronunciation of ``text`` in ``lang``."""
    lang = _validate_lang(lang)
    try:
        return FileResponse(audio.text2speech(text, lang=lang), media_type="audio/wav")
    except Exception:
        logger.exception("tts failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.get("/languages")
async def api_languages():
    """List the supported languages (``code`` is the value of the ``lang`` form field)."""
    languages = [{"code": language.code, "name": language.name} for language in LANGUAGES.values()]
    return {"default": DEFAULT_LANGUAGE, "languages": languages}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})
