"""Audio loading, conversion and reference-speech generation."""

import hashlib
import logging
import os
import tempfile

import librosa
import soundfile as sf
from gtts import gTTS

logger = logging.getLogger(__name__)

TARGET_SR = 16000

CACHE_DIR = os.environ.get(
    "OPENPRONOUNCE_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "openpronounce"),
)


def load(file_path, sr=TARGET_SR):
    """Load any audio file (wav, mp3, flac, ogg, webm, m4a...) as a mono float32 waveform at 16 kHz."""
    waveform, _ = librosa.load(file_path, sr=sr, mono=True)
    return waveform


def webm2wav(file_path):
    """Convert a browser-recorded file (webm/ogg/wav/...) to a 16 kHz mono ``*.16k.wav`` file next to it.

    Uses librosa (soundfile, then ffmpeg through audioread) so that ffmpeg is only
    needed for formats libsndfile cannot read.
    """
    output_path = os.path.splitext(file_path)[0] + ".16k.wav"
    try:
        waveform = load(file_path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to decode {file_path!r} ({e}). Make sure ffmpeg is installed."
        ) from e
    sf.write(output_path, waveform, TARGET_SR)
    return output_path


# Backward-compatible alias (the original name was a typo)
webp2wav = webm2wav


def text2speech(text, lang="en", filename=None, target_sr=TARGET_SR):
    """Generate a reference pronunciation of ``text`` as a 16 kHz mono wav file and return its path.

    Uses gTTS (Google Translate TTS), so an internet connection is required the first
    time a given text is requested. Results are cached in ``CACHE_DIR`` keyed by
    ``(lang, text)`` so that repeated comparisons against the same sentence are free.
    """
    if filename is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.sha1(f"{lang}\x00{target_sr}\x00{text}".encode("utf-8")).hexdigest()
        filename = os.path.join(CACHE_DIR, f"tts-{key}.wav")
        if os.path.exists(filename):
            return filename

    fd, mp3_path = tempfile.mkstemp(suffix=".mp3", prefix="openpronounce-tts-")
    os.close(fd)
    try:
        gTTS(text=text, lang=lang, slow=False).save(mp3_path)
        waveform = load(mp3_path, sr=target_sr)
    finally:
        try:
            os.remove(mp3_path)
        except OSError:
            pass

    sf.write(filename, waveform, target_sr)
    return filename
