"""OpenPronounce: open-source, phoneme-level pronunciation assessment (English by default).

Typical usage::

    from openpronounce import load_audio, compare_audio_with_text

    sound = load_audio("recording.wav")
    result = compare_audio_with_text(sound, "Hello, I am a developer")
    print(result["score"], result["differences"]["errors"])

    result = compare_audio_with_text(load_audio("bonjour.wav"), "Bonjour le monde", lang="fr")
"""

from .audio import load as load_audio, text2speech
from .languages import LANGUAGES, get_language
from .phones import compare_phones, recognize_phones, transcribe_phones
from .speech import (
    compare_audio_with_text,
    compare_transcriptions,
    get_phonemes,
    get_phonemes_with_word_mapping,
    transcribe,
)

__version__ = "0.2.1"

__all__ = [
    "__version__",
    "load_audio",
    "text2speech",
    "LANGUAGES",
    "get_language",
    "compare_audio_with_text",
    "compare_transcriptions",
    "compare_phones",
    "recognize_phones",
    "transcribe_phones",
    "get_phonemes",
    "get_phonemes_with_word_mapping",
    "transcribe",
]
