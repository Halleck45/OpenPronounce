import librosa
from pydub import AudioSegment
from gtts import gTTS
import torchaudio
import torchaudio.transforms as transforms
import os
import uuid

def load(file_path):
    """ Charge un fichier audio et le convertit en mono 16kHz """
    audio, sr = librosa.load(file_path, sr=16000)
    return audio

def webp2wav(file_path):
    """ Convertit un fichier audio webp en wav """
    audio = AudioSegment.from_file(file_path, format="webm")
    audio.export(file_path.replace('.webm', '.wav'), format="wav")
    print(file_path.replace('.webm', '.wav'))
    return file_path.replace('.webm', '.wav')



def text2speech(text, lang="en", filename="reference.mp3", target_sr=16000):
    """
    Convertit un texte en audio et force le sampling rate à 16 kHz.
    """

    if not filename:
        filename = f"/tmp/{uuid.uuid4()}.wav"

    # if filename exists, remove it
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    # Générer le fichier avec gTTS (format MP3)
    tts = gTTS(text=text, lang=lang, slow=False)
    temp_filename = filename.replace(".wav", ".mp3")
    tts.save(temp_filename)

    # Charger l’audio avec torchaudio
    waveform, sample_rate = torchaudio.load(temp_filename)

    # Vérifier si on doit resampler
    if sample_rate != target_sr:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    # Sauvegarder en WAV avec le bon taux d’échantillonnage
    torchaudio.save(filename, waveform, target_sr)
    return filename
