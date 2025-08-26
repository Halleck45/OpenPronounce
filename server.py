from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


import speech
import random
import audio
import string

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

@app.route('/')
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )