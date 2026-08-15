"""Registry of the supported languages.

Each language binds a 2-letter code (also the gTTS code) to the espeak voice used
for phonemization and to the Wav2Vec2 CTC checkpoint used for word transcription.
The phone recognizer (see :mod:`openpronounce.phones`) is multilingual and shared.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    espeak: str
    asr_model: str
    name: str


DEFAULT_LANGUAGE = "en"

LANGUAGES = {
    "en": Language("en", "en-us", "facebook/wav2vec2-large-960h", "English"),
    "fr": Language("fr", "fr-fr", "jonatasgrosman/wav2vec2-large-xlsr-53-french", "French"),
    "es": Language("es", "es", "jonatasgrosman/wav2vec2-large-xlsr-53-spanish", "Spanish"),
    "de": Language("de", "de", "jonatasgrosman/wav2vec2-large-xlsr-53-german", "German"),
    "it": Language("it", "it", "jonatasgrosman/wav2vec2-large-xlsr-53-italian", "Italian"),
    "pt": Language("pt", "pt-br", "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "Portuguese"),
    "nl": Language("nl", "nl", "jonatasgrosman/wav2vec2-large-xlsr-53-dutch", "Dutch"),
}


def get_language(code):
    """Return the :class:`Language` for ``code`` or raise ``ValueError`` listing the supported codes."""
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"unsupported language {code!r}, expected one of: {', '.join(LANGUAGES)}") from None
