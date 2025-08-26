from gtts import gTTS
import os

def generate_audio_google(text, lang="en", filename="reference.mp3"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    os.system(f"play {filename}")  # Joue le fichier audio
    return filename

generate_audio_google("Hello, how are you?")
