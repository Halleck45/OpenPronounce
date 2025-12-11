import librosa
from pydub import AudioSegment
from gtts import gTTS
import torchaudio
import torchaudio.transforms as transforms
import soundfile as sf
import os
import uuid

def load(file_path):
    """ Charge un fichier audio et le convertit en mono 16kHz """
    audio, sr = librosa.load(file_path, sr=16000)
    return audio

def webp2wav(file_path):
    """ Convertit un fichier audio webm en wav """
    output_path = file_path.replace('.webm', '.wav')
    
    try:
        # Essayer avec pydub (nécessite ffmpeg)
        audio = AudioSegment.from_file(file_path, format="webm")
        audio.export(output_path, format="wav")
    except Exception as e:
        # Fallback: utiliser torchaudio si pydub échoue
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            # Resampler à 16kHz si nécessaire
            if sample_rate != 16000:
                resampler = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            # Convertir en mono si stéréo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            torchaudio.save(output_path, waveform, 16000)
        except Exception as e2:
            # Dernier recours: utiliser librosa
            try:
                audio_data, sr = librosa.load(file_path, sr=16000, mono=True)
                sf.write(output_path, audio_data, 16000)
            except Exception as e3:
                raise RuntimeError(
                    f"Impossible de convertir le fichier webm. "
                    f"Erreurs: pydub={str(e)}, torchaudio={str(e2)}, librosa={str(e3)}. "
                    f"Assurez-vous que ffmpeg est installé."
                ) from e3
    
    return output_path



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
