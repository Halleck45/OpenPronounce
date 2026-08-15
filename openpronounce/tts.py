"""Text-to-speech backends used to synthesize the reference pronunciation.

Three backends are available, selected with the ``OPENPRONOUNCE_TTS`` environment
variable (or the ``backend`` argument of :func:`openpronounce.audio.text2speech`):

- ``gtts`` (default): Google Translate TTS, needs network access, no local model.
- ``piper``: Piper (VITS, ONNX), fully offline once the voice (~60 MB) is downloaded.
  ``pip install openpronounce[tts-piper]``.
- ``kokoro``: Kokoro-82M (PyTorch), fully offline once the model (~330 MB) and the
  voice are downloaded. ``pip install openpronounce[tts-kokoro]``.

The voice is selected with ``OPENPRONOUNCE_TTS_VOICE`` (or the ``voice`` argument):
a Piper voice id (``en_US-lessac-medium``), a Kokoro voice name (``af_heart``) or,
for gTTS, the Google domain used for the accent (``com``, ``co.uk``, ``com.au``...).

Each backend is a callable ``(text, lang, voice) -> (waveform, sample_rate)`` where
``waveform`` is a mono float32 numpy array. Optional dependencies are imported lazily
so that importing this module never requires piper or kokoro.
"""

import os
import tempfile

import numpy as np

DEFAULT_BACKEND = "gtts"

# Piper voices on Hugging Face (rhasspy/piper-voices), one medium-quality voice per language.
PIPER_VOICES_REPO = "rhasspy/piper-voices"
PIPER_DEFAULT_VOICES = {
    "ar": "ar_JO-kareem-medium",
    "de": "de_DE-thorsten-medium",
    "en": "en_US-lessac-medium",
    "es": "es_ES-davefx-medium",
    "fr": "fr_FR-siwis-medium",
    "hi": "hi_IN-pratham-medium",
    "it": "it_IT-paola-medium",
    "nl": "nl_NL-mls-medium",
    "pl": "pl_PL-gosia-medium",
    "pt": "pt_BR-faber-medium",
    "ru": "ru_RU-irina-medium",
    "sv": "sv_SE-nst-medium",
    "tr": "tr_TR-dfki-medium",
    "zh": "zh_CN-huayan-medium",
}

# Kokoro-82M: our 2-letter language codes -> (Kokoro lang code, default voice).
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
KOKORO_LANGUAGES = {
    "en": ("a", "af_heart"),
    "es": ("e", "ef_dora"),
    "fr": ("f", "ff_siwis"),
    "hi": ("h", "hf_alpha"),
    "it": ("i", "if_sara"),
    "ja": ("j", "jf_alpha"),
    "pt": ("p", "pf_dora"),
    "zh": ("z", "zf_xiaobei"),
}

_piper_voices = {}
_kokoro_pipelines = {}


def _import(module, extra):
    """Import an optional module and explain how to install it when missing."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"The {module!r} package is required for the {extra} TTS backend: "
            f"pip install 'openpronounce[tts-{extra}]'"
        ) from e


def gtts_default_voice(lang):
    """Return the default gTTS "voice", i.e. the Google Translate domain."""
    return "com"


def synthesize_gtts(text, lang, voice):
    """Synthesize with Google Translate TTS (network). Returns ``(waveform, sample_rate)``."""
    import librosa
    from gtts import gTTS

    fd, mp3_path = tempfile.mkstemp(suffix=".mp3", prefix="openpronounce-tts-")
    os.close(fd)
    try:
        gTTS(text=text, lang=lang, tld=voice, slow=False).save(mp3_path)
        waveform, sr = librosa.load(mp3_path, sr=None, mono=True)
    finally:
        try:
            os.remove(mp3_path)
        except OSError:
            pass
    return waveform, sr


def piper_default_voice(lang):
    """Return the default Piper voice for ``lang`` or raise if we have none."""
    try:
        return PIPER_DEFAULT_VOICES[lang]
    except KeyError:
        raise ValueError(
            f"No default Piper voice for language {lang!r}. Pick one at "
            f"https://huggingface.co/{PIPER_VOICES_REPO} (e.g. 'en_US-lessac-medium') and set "
            "OPENPRONOUNCE_TTS_VOICE or pass voice=... to text2speech()."
        ) from None


def _piper_voice_files(voice):
    """Download (once, into the Hugging Face cache) and return the .onnx and .json paths of a Piper voice."""
    from huggingface_hub import hf_hub_download

    try:
        lang_code, name, quality = voice.split("-", 2)
        family = lang_code.split("_")[0]
    except ValueError:
        raise ValueError(
            f"Invalid Piper voice {voice!r}, expected '<lang>_<REGION>-<name>-<quality>' like 'en_US-lessac-medium'."
        ) from None
    folder = f"{family}/{lang_code}/{name}/{quality}"
    model = hf_hub_download(PIPER_VOICES_REPO, f"{folder}/{voice}.onnx")
    config = hf_hub_download(PIPER_VOICES_REPO, f"{folder}/{voice}.onnx.json")
    return model, config


def synthesize_piper(text, lang, voice):
    """Synthesize with Piper (offline). Returns ``(waveform, sample_rate)``."""
    piper = _import("piper", "piper")

    if voice not in _piper_voices:
        model, config = _piper_voice_files(voice)
        _piper_voices[voice] = piper.PiperVoice.load(model, config)
    engine = _piper_voices[voice]

    chunks = list(engine.synthesize(text))
    if not chunks:
        raise RuntimeError(f"Piper produced no audio for {text!r}")
    waveform = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return waveform, chunks[0].sample_rate


def kokoro_default_voice(lang):
    """Return the default Kokoro voice for ``lang`` or raise if Kokoro does not support it."""
    return _kokoro_language(lang)[1]


def _kokoro_language(lang):
    try:
        return KOKORO_LANGUAGES[lang]
    except KeyError:
        raise ValueError(
            f"Kokoro does not support language {lang!r} "
            f"(supported: {', '.join(sorted(KOKORO_LANGUAGES))}). Use another TTS backend."
        ) from None


def synthesize_kokoro(text, lang, voice):
    """Synthesize with Kokoro-82M (offline). Returns ``(waveform, sample_rate)``."""
    kokoro = _import("kokoro", "kokoro")

    lang_code, _ = _kokoro_language(lang)
    if lang_code not in _kokoro_pipelines:
        # Share the 82M model between per-language pipelines.
        model = next((p.model for p in _kokoro_pipelines.values()), True)
        _kokoro_pipelines[lang_code] = kokoro.KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO, model=model)
    pipeline = _kokoro_pipelines[lang_code]

    parts = [r.audio.detach().cpu().numpy() for r in pipeline(text, voice=voice) if r.audio is not None]
    if not parts:
        raise RuntimeError(f"Kokoro produced no audio for {text!r}")
    return np.concatenate(parts).astype(np.float32), KOKORO_SAMPLE_RATE


BACKENDS = {
    "gtts": (synthesize_gtts, gtts_default_voice),
    "piper": (synthesize_piper, piper_default_voice),
    "kokoro": (synthesize_kokoro, kokoro_default_voice),
}


def resolve(lang, backend=None, voice=None):
    """Return the effective ``(backend, voice)`` from the arguments, the environment and the defaults.

    Raises ``ValueError`` for an unknown backend or a language the backend has no default voice for.
    """
    backend = backend or os.environ.get("OPENPRONOUNCE_TTS") or DEFAULT_BACKEND
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown TTS backend {backend!r} (OPENPRONOUNCE_TTS), expected one of: {', '.join(BACKENDS)}"
        )
    voice = voice or os.environ.get("OPENPRONOUNCE_TTS_VOICE") or BACKENDS[backend][1](lang)
    return backend, voice


def synthesize(text, lang, backend, voice):
    """Run the given backend and return ``(waveform, sample_rate)`` (mono float32)."""
    return BACKENDS[backend][0](text, lang, voice)
