"""Audio loading, conversion and reference-speech generation."""

import hashlib
import logging
import os
import tempfile

import librosa
import soundfile as sf

from openpronounce import tts

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


def text2speech(text, lang="en", filename=None, target_sr=TARGET_SR, *, backend=None, voice=None):
    """Generate a reference pronunciation of ``text`` as a 16 kHz mono wav file and return its path.

    The synthesizer is chosen with ``backend`` (``gtts``, ``piper`` or ``kokoro``), falling
    back to the ``OPENPRONOUNCE_TTS`` environment variable, then to gTTS; the voice with
    ``voice`` / ``OPENPRONOUNCE_TTS_VOICE``, then to a per-language default. See
    :mod:`openpronounce.tts`. Results are cached in ``CACHE_DIR`` keyed by
    ``(backend, voice, lang, text)`` so that repeated comparisons against the same
    sentence are free.
    """
    backend, voice = tts.resolve(lang, backend=backend, voice=voice)
    if filename is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.sha1(
            f"{backend}\x00{voice}\x00{lang}\x00{target_sr}\x00{text}".encode("utf-8")
        ).hexdigest()
        filename = os.path.join(CACHE_DIR, f"tts-{key}.wav")
        if os.path.exists(filename):
            return filename

    logger.info("Synthesizing reference with %s (voice %s, lang %s)", backend, voice, lang)
    waveform, sr = tts.synthesize(text, lang, backend, voice)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    sf.write(filename, waveform, target_sr)
    return filename
