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
    #: Mean per-frame DTW distance between the Wav2Vec2 embeddings of two good native
    #: voices of this language (gTTS vs Piper on three sentences, English checkpoint).
    #: The acoustic term of the score maps this value to 100 and ``acoustic_good + 9`` to 0.
    #: English is calibrated on human readings and speechocean762 (see benchmarks/);
    #: for the other languages the English embeddings put even two native voices
    #: further apart, hence the offset.
    acoustic_good: float = 6.0


DEFAULT_LANGUAGE = "en"

LANGUAGES = {
    "en": Language("en", "en-us", "facebook/wav2vec2-large-960h", "English", acoustic_good=6.0),
    "fr": Language("fr", "fr-fr", "jonatasgrosman/wav2vec2-large-xlsr-53-french", "French", acoustic_good=9.0),
    "es": Language("es", "es", "jonatasgrosman/wav2vec2-large-xlsr-53-spanish", "Spanish", acoustic_good=10.0),
    "de": Language("de", "de", "jonatasgrosman/wav2vec2-large-xlsr-53-german", "German", acoustic_good=10.0),
    "it": Language("it", "it", "jonatasgrosman/wav2vec2-large-xlsr-53-italian", "Italian", acoustic_good=9.0),
    "pt": Language("pt", "pt-br", "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "Portuguese", acoustic_good=11.0),
    "nl": Language("nl", "nl", "jonatasgrosman/wav2vec2-large-xlsr-53-dutch", "Dutch", acoustic_good=13.0),
}


def get_language(code):
    """Return the :class:`Language` for ``code`` or raise ``ValueError`` listing the supported codes."""
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"unsupported language {code!r}, expected one of: {', '.join(LANGUAGES)}") from None
