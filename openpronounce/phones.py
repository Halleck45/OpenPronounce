"""Phone-level recognition and comparison.

Uses ``facebook/wav2vec2-lv-60-espeak-cv-ft``, a Wav2Vec2 model fine-tuned to emit
espeak IPA phones directly from audio, so pronunciation errors are detected inside
words without going through a word transcription (which silently "corrects" what
the learner said).
"""

import logging
import os
import re
from functools import lru_cache

import Levenshtein
import torch
from phonemizer import phonemize
from phonemizer.separator import Separator

from .languages import DEFAULT_LANGUAGE, get_language

logger = logging.getLogger(__name__)

PHONE_MODEL_NAME = os.environ.get("OPENPRONOUNCE_PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
SAMPLING_RATE = 16000

# A word is reported when at least this share of its phones is wrong...
PHONE_ERROR_THRESHOLD = 0.5
# ...or when at least this many phones are wrong, whatever the length of the word.
PHONE_ERROR_MIN_EDITS = 3

# Phones that espeak and the recognizer use interchangeably, or that no learner
# should be penalised for, per language. Length marks are always dropped. Only
# English needs merges so far: ɔ/ɑ and ɾ/t are contrastive in French, German or Spanish.
_PHONE_MAP = {
    "en": {
        "ᵻ": "ɪ",   # espeak's reduced /ɪ/
        "ɐ": "ə",   # espeak's reduced /a/ (article "a")
        "ɔ": "ɑ",   # cot-caught merger (American English)
        "ɾ": "t",   # flapped t
        "ɫ": "l",
    },
}

# Function words with more than one accepted pronunciation (already normalized), per language.
ALTERNATE_PRONUNCIATIONS = {
    "en": {
        "a": [["ə"], ["eɪ"]],
        "an": [["ən"], ["æn"]],
        "the": [["ð", "ə"], ["ð", "i"], ["ð", "ɪ"]],
        "to": [["t", "ə"], ["t", "u"]],
        "of": [["ʌ", "v"], ["ə", "v"]],
        "and": [["æ", "n", "d"], ["ə", "n", "d"], ["ə", "n"]],
        "for": [["f", "ɔ", "ɹ"], ["f", "ɚ"]],
        "you": [["j", "u"], ["j", "ə"]],
        "are": [["ɑ", "ɹ"], ["ɚ"]],
        "was": [["w", "ʌ", "z"], ["w", "ə", "z"]],
        "that": [["ð", "æ", "t"], ["ð", "ə", "t"]],
        "can": [["k", "æ", "n"], ["k", "ə", "n"]],
        "have": [["h", "æ", "v"], ["h", "ə", "v"]],
        "or": [["ɔ", "ɹ"], ["ɚ"]],
    },
    # Schwa elision in French function words ("je suis" -> /ʒsɥi/).
    "fr": {
        "je": [["ʒ"]],
        "le": [["l"]],
        "de": [["d"]],
        "ne": [["n"]],
        "ce": [["s"]],
        "se": [["s"]],
        "me": [["m"]],
        "te": [["t"]],
        "que": [["k"]],
    },
}


def is_enabled():
    return PHONE_MODEL_NAME not in ("", "0", "off", "false", "no")


@lru_cache(maxsize=1)
def _load_model():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    logger.info("Loading %s", PHONE_MODEL_NAME)
    processor = Wav2Vec2Processor.from_pretrained(PHONE_MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(PHONE_MODEL_NAME)
    model.eval()
    return processor, model


def normalize_phone(phone, lang=DEFAULT_LANGUAGE):
    """Map a phone to its canonical form (drop length marks, merge near-identical phones of ``lang``)."""
    phone = phone.replace("ː", "")
    return _PHONE_MAP.get(lang, {}).get(phone, phone)


def normalize_phones(phones, lang=DEFAULT_LANGUAGE):
    """Normalize a phone sequence and collapse immediate repetitions ("ɚ ɹ" -> "ɚ", "d d" -> "d")."""
    out = []
    for phone in phones:
        phone = normalize_phone(phone, lang)
        if not phone:
            continue
        if out and (out[-1] == phone or (out[-1] == "ɚ" and phone == "ɹ")):
            continue
        out.append(phone)
    return out


def transcribe_phones(audio_waveform, sampling_rate=SAMPLING_RATE, normalize=True, lang=DEFAULT_LANGUAGE):
    """Recognize the phones of a 16 kHz waveform. Returns a list of IPA phones (normalized for ``lang``)."""
    processor, model = _load_model()
    inputs = processor(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    phones = processor.batch_decode(predicted_ids)[0].split()
    return normalize_phones(phones, lang) if normalize else phones


_WORD_RE = re.compile(r"\b[\w']+\b")


@lru_cache(maxsize=1024)
def _expected_phones_by_word(text, lang=DEFAULT_LANGUAGE):
    """Return ``(words, phones_per_word)`` for ``text``, phones normalized."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return (), ()
    language = get_language(lang).espeak
    try:
        out = phonemize(
            " ".join(words),
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=Separator(phone=" ", word=" | ", syllable=""),
        )
        groups = [g.split() for g in out.split("|")]
        if len(groups) != len(words):
            raise ValueError(f"expected {len(words)} words, phonemizer returned {len(groups)} groups")
    except Exception as e:  # noqa: BLE001 - fall back to one call per word
        logger.debug("batch phonemization failed (%s), falling back to per-word", e)
        groups = []
        for word in words:
            try:
                out = phonemize(word, language=language, backend="espeak", strip=True,
                                separator=Separator(phone=" ", word=" ", syllable=""))
                groups.append(out.split())
            except Exception:  # noqa: BLE001
                groups.append([])
    return tuple(words), tuple(tuple(normalize_phones(g, lang)) for g in groups)


def get_expected_phones(text, lang=DEFAULT_LANGUAGE):
    """Return ``(words, phones_per_word)`` for ``text``. Words with no phones are kept as empty tuples."""
    words, groups = _expected_phones_by_word(text, lang)
    return list(words), [list(g) for g in groups]


def _align(expected, heard):
    """Map every expected phone index to the heard phone indices it aligns with (insertions go to the previous phone)."""
    alignment = [set() for _ in expected]
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(list(expected), list(heard)):
        if tag == "equal":
            for k, l in zip(range(i1, i2), range(j1, j2)):
                alignment[k].add(l)
        elif tag == "replace":
            len_i, len_j = i2 - i1, j2 - j1
            for k in range(i1, i2):
                start = j1 + int((k - i1) * len_j / len_i)
                end = j1 + int((k - i1 + 1) * len_j / len_i)
                if start == end:
                    alignment[k].add(min(start, j2 - 1))
                else:
                    alignment[k].update(range(start, end))
        elif tag == "insert":
            k = i1 - 1 if i1 > 0 else i1
            if k < len(alignment):
                alignment[k].update(range(j1, j2))
    return alignment


def _word_distance(word, expected_seg, actual_seg, previous_last_phone=None, lang=DEFAULT_LANGUAGE):
    """Smallest edit distance between what was heard and any accepted pronunciation of ``word``.

    When the word starts with the phone the previous word ended with ("heat to"),
    speakers merge the two; the shortened form is accepted as well.
    """
    alternates = ALTERNATE_PRONUNCIATIONS.get(lang, {}).get(word, [])
    candidates = [list(expected_seg)] + [normalize_phones(alt, lang) for alt in alternates]
    if previous_last_phone is not None and len(expected_seg) > 1 and expected_seg[0] == previous_last_phone:
        candidates.append(list(expected_seg[1:]))
    return min(Levenshtein.distance(c, list(actual_seg)) for c in candidates)


def compare_phones(heard_phones, text_reference, lang=DEFAULT_LANGUAGE):
    """Compare recognized phones with the phones expected for ``text_reference`` in ``lang``, word by word.

    Returns a dict with ``expected_phones`` (per word), ``heard_phones``, ``phone_error_rate``,
    ``errors`` (same shape as the text-based errors: ``position``, ``word``, ``expected``,
    ``actual``, ``actual_word``) and ``words_with_errors``.
    """
    words, groups = get_expected_phones(text_reference, lang)
    expected = [p for g in groups for p in g]
    heard = list(heard_phones)

    alignment = _align(expected, heard)
    phone_error_rate = Levenshtein.distance(expected, heard) / max(1, len(expected))

    errors = []
    words_with_errors = []
    offset = 0
    previous_last_phone = None
    for position, (word, group) in enumerate(zip(words, groups)):
        if not group:
            continue
        indices = range(offset, offset + len(group))
        offset += len(group)

        matched = sorted(set().union(*(alignment[i] for i in indices)))
        actual = [heard[j] for j in matched]
        distance = _word_distance(word, group, actual, previous_last_phone, lang)
        previous_last_phone = group[-1]

        if distance and (distance / len(group) >= PHONE_ERROR_THRESHOLD or distance >= PHONE_ERROR_MIN_EDITS):
            errors.append({
                "position": position,
                "word": word,
                "expected": "".join(group),
                "actual": "".join(actual),
                "actual_word": "",
                "phone_distance": distance,
            })
            words_with_errors.append(word)

    return {
        "expected_phones": [list(g) for g in groups],
        "heard_phones": heard,
        "phone_error_rate": round(phone_error_rate, 4),
        "errors": errors,
        "words_with_errors": words_with_errors,
    }
