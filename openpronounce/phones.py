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
from typing import NamedTuple

import Levenshtein
import numpy as np
import torch
from phonemizer import phonemize
from phonemizer.separator import Separator

from .device import get_device

from .languages import DEFAULT_LANGUAGE, get_language

logger = logging.getLogger(__name__)

PHONE_MODEL_NAME = os.environ.get("OPENPRONOUNCE_PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
SAMPLING_RATE = 16000

# Every wrong phone of a word gets an error confidence (0-1, see :func:`compare_phones`).
# A word is reported when these confidences add up to at least this share of its phones...
PHONE_ERROR_THRESHOLD = 0.4
# ...or to at least this many phones, whatever the length of the word.
PHONE_ERROR_MIN_EDITS = 2
# Substituting a phone for a close one (voicing, tense/lax vowel...) counts for this
# much of a full error: raters tolerate them, and the recognizer confuses them too.
NEAR_PHONE_COST = 0.5
# Extra phones heard after the last phone of a word (epenthetic vowel, onset of the
# next word caught by the alignment) and a dropped final consonant count for this much.
FINAL_EXTRA_COST = 0.25
FINAL_DELETION_COST = 0.5
# An expected phone whose posterior reaches this value where the recognizer heard
# something else is not a confident error (Goodness-of-Pronunciation style); the
# confidence decreases linearly from 1 (posterior 0) to 0 (posterior at this value).
PHONE_PLAUSIBLE_POSTERIOR = 0.05

# Recognizer tokens that are not phones of any supported language: Mandarin tone
# numbers, aspiration marks and two-letter spellings of English diphthongs.
_TOKEN_ALIASES = {"th": "t", "kh": "k", "ph": "p", "tʰ": "t", "kʰ": "k", "pʰ": "p",
                  "ai": "aɪ", "ei": "eɪ", "au": "aʊ", "ou": "oʊ"}
_TONE_RE = re.compile(r"\d")

# Phones that espeak and the recognizer use interchangeably, or that no learner
# should be penalised for, per language. Length marks are always dropped. Only
# English needs merges so far: ɔ/ɑ and ɾ/t are contrastive in French, German or Spanish.
_PHONE_MAP = {
    "en": {
        "ᵻ": "ɪ",   # espeak's reduced /ɪ/
        "ɐ": "ə",   # espeak's reduced /a/ (article "a")
        "ɔ": "ɑ",   # cot-caught merger (American English)
        "ɔɹ": "oɹ",  # horse-hoarse merger ("for" / "four")
        "ɜ": "ɚ",   # nurse vowel: espeak writes ɜː, the recognizer ɚ
        "ɾ": "t",   # flapped t
        "ɫ": "l",
    },
}

# Groups of phones close enough that substituting one for another is a minor error
# (cost NEAR_PHONE_COST): voicing pairs, tense/lax vowels, dental fricatives said as
# stops or sibilants, and other typical L2 approximations that raters accept.
_NEAR_PHONES = {
    "en": [
        {"ɪ", "i"}, {"ʊ", "u"}, {"ɛ", "æ"}, {"ʌ", "ɑ", "ə", "a"}, {"æ", "a"}, {"ɛ", "eɪ"},
        {"ð", "z", "d"}, {"θ", "s", "t"}, {"b", "p"}, {"d", "t"}, {"ɡ", "k"}, {"z", "s"},
        {"v", "f"}, {"ʒ", "ʃ"}, {"dʒ", "tʃ"}, {"n", "ŋ"}, {"h", "x"}, {"ɹ", "r"}, {"v", "w"},
    ],
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
    model = Wav2Vec2ForCTC.from_pretrained(PHONE_MODEL_NAME).to(get_device())
    model.eval()
    return processor, model


def normalize_phone(phone, lang=DEFAULT_LANGUAGE):
    """Map a phone to its canonical form (drop length marks, merge near-identical phones of ``lang``)."""
    phone = _TONE_RE.sub("", phone.replace("ː", ""))
    phone = _TOKEN_ALIASES.get(phone, phone)
    return _PHONE_MAP.get(lang, {}).get(phone, phone)


def normalize_phones(phones, lang=DEFAULT_LANGUAGE):
    """Normalize a phone sequence and collapse immediate repetitions ("ɚ ɹ" -> "ɚ", "d d" -> "d")."""
    return [phone for phone, _ in _normalize_indexed(phones, lang)]


def _normalize_indexed(phones, lang):
    """Normalize ``phones`` and return ``(phone, source_indices)`` pairs, one per surviving phone.

    ``source_indices`` lists the positions of ``phones`` merged into that phone, so
    that per-phone data (confidences, frames) can be merged the same way.
    """
    out = []
    for index, phone in enumerate(phones):
        phone = normalize_phone(phone, lang)
        if not phone:
            continue
        if out and (out[-1][0] == phone or (out[-1][0] == "ɚ" and phone == "ɹ")):
            out[-1][1].append(index)
            continue
        out.append((phone, [index]))
    return out


class PhoneRecognition(NamedTuple):
    """Phones recognized in a waveform, with the frame posteriors they were decoded from.

    ``phones`` are normalized, ``confidences`` (0-1) is the peak posterior of each
    phone over the frames it was decoded from, ``spans`` the ``(start, end)`` frame
    range of each phone, ``log_posteriors`` the ``(frames, vocab)`` log posteriors of
    the model and ``vocab`` its tokens indexed by id.
    """

    phones: list
    confidences: list
    spans: list
    log_posteriors: np.ndarray
    vocab: tuple


def _is_special(token):
    return token.startswith("<") and token.endswith(">")


def decode_ctc(log_posteriors, vocab, blank_id=0, lang=DEFAULT_LANGUAGE, normalize=True):
    """Greedy CTC decoding of ``log_posteriors`` (frames x vocab) into a :class:`PhoneRecognition`.

    Repeated frames are collapsed, blanks and special tokens dropped. Each phone gets
    the peak posterior of its token over its frames as confidence; when normalization
    merges phones, the merged phone keeps the highest confidence and the union of the frames.
    """
    log_posteriors = np.asarray(log_posteriors, dtype=np.float32)
    ids = log_posteriors.argmax(axis=1)
    tokens, confidences, spans = [], [], []
    start = 0
    for end in range(1, len(ids) + 1):
        if end < len(ids) and ids[end] == ids[start]:
            continue
        token = vocab[ids[start]]
        if ids[start] != blank_id and not _is_special(token):
            tokens.append(token)
            confidences.append(float(np.exp(log_posteriors[start:end, ids[start]].max())))
            spans.append((start, end))
        start = end
    if not normalize:
        return PhoneRecognition(tokens, confidences, spans, log_posteriors, tuple(vocab))
    phones, merged_confidences, merged_spans = [], [], []
    for phone, sources in _normalize_indexed(tokens, lang):
        phones.append(phone)
        merged_confidences.append(max(confidences[i] for i in sources))
        merged_spans.append((spans[sources[0]][0], spans[sources[-1]][1]))
    return PhoneRecognition(phones, merged_confidences, merged_spans, log_posteriors, tuple(vocab))


def phone_log_posteriors(audio_waveform, sampling_rate=SAMPLING_RATE):
    """Frame-level log posteriors of the phone model for a waveform, as a ``(frames, vocab)`` numpy array."""
    processor, model = _load_model()
    inputs = processor(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(get_device())).logits[0]
    return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()


@lru_cache(maxsize=1)
def phone_vocab():
    """Vocabulary of the phone model as a tuple of tokens indexed by id."""
    processor, _ = _load_model()
    vocab = processor.tokenizer.get_vocab()
    return tuple(token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1]))


def recognize_phones(audio_waveform, sampling_rate=SAMPLING_RATE, normalize=True, lang=DEFAULT_LANGUAGE):
    """Recognize the phones of a 16 kHz waveform with their confidences and frame posteriors."""
    processor, _ = _load_model()
    log_posteriors = phone_log_posteriors(audio_waveform, sampling_rate)
    return decode_ctc(log_posteriors, phone_vocab(), processor.tokenizer.pad_token_id, lang, normalize)


def transcribe_phones(audio_waveform, sampling_rate=SAMPLING_RATE, normalize=True, lang=DEFAULT_LANGUAGE,
                      return_confidence=False):
    """Recognize the phones of a 16 kHz waveform. Returns a list of IPA phones (normalized for ``lang``).

    With ``return_confidence=True``, returns ``(phones, confidences)`` where each
    confidence (0-1) is the peak posterior of the phone over the frames it was decoded from.
    """
    recognition = recognize_phones(audio_waveform, sampling_rate, normalize, lang)
    if return_confidence:
        return recognition.phones, recognition.confidences
    return recognition.phones


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
                                preserve_punctuation=False,
                                separator=Separator(phone=" ", word="", syllable=""))
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


def _pronunciations(word, expected_seg, previous_last_phone=None, lang=DEFAULT_LANGUAGE):
    """Accepted phone sequences for ``word``: the expected one, its alternates, and, when the
    word starts with the phone the previous word ended with ("heat to"), the merged form."""
    alternates = ALTERNATE_PRONUNCIATIONS.get(lang, {}).get(word, [])
    candidates = [list(expected_seg)] + [normalize_phones(alt, lang) for alt in alternates]
    if previous_last_phone is not None and len(expected_seg) > 1 and expected_seg[0] == previous_last_phone:
        candidates.append(list(expected_seg[1:]))
    return candidates


def _substitution_cost(expected, heard, lang):
    """Cost of hearing ``heard`` instead of ``expected``: 1, or NEAR_PHONE_COST for close phones."""
    for group in _NEAR_PHONES.get(lang, []):
        if expected in group and heard in group:
            return NEAR_PHONE_COST
    return 1.0


@lru_cache(maxsize=8)
def _token_ids_by_phone(vocab, lang):
    """Map every normalized phone to the ids of the vocabulary tokens that normalize to it."""
    ids = {}
    for index, token in enumerate(vocab):
        if not _is_special(token):
            ids.setdefault(normalize_phone(token, lang), []).append(index)
    return ids


def _plausibility(recognition, phone, start, end, lang):
    """Highest posterior of ``phone`` (any of its tokens) over frames ``start:end``, 0 without posteriors."""
    if recognition is None or end <= start:
        return 0.0
    ids = _token_ids_by_phone(recognition.vocab, lang).get(phone)
    if not ids:
        return 0.0
    return float(np.exp(recognition.log_posteriors[start:end][:, ids].max()))


def _phone_reports(candidate, actual, matched, recognition, region, lang):
    """Per-phone report of one word: ``{expected, heard, confidence}`` for every phone of ``candidate``.

    ``actual`` are the heard phones of the word (indices ``matched`` in the recognition),
    ``region`` the frame range of the word. The confidence is how sure we are that the
    phone is wrong: 0 for a correct phone; otherwise its cost (1 for a substitution,
    a deletion or an extra phone, less for a close substitution or at the end of the
    word) scaled down by the plausibility of the expected phone in the frames where it
    should have been.
    """
    local = _align(candidate, actual)
    spans = recognition.spans if recognition is not None else None

    def frames(j):
        return spans[matched[j]] if spans is not None else (0, 0)

    reports = []
    last = len(candidate) - 1
    for k, expected in enumerate(candidate):
        heard_indices = sorted(local[k])
        heard = [actual[j] for j in heard_indices]
        if heard == [expected]:
            confidence = 0.0
        elif expected in heard:
            confidence = FINAL_EXTRA_COST if k == last else 1.0
        else:
            if heard:
                cost = max(_substitution_cost(expected, h, lang) for h in heard)
                start = min(frames(j)[0] for j in heard_indices)
                end = max(frames(j)[1] for j in heard_indices)
            else:
                cost = FINAL_DELETION_COST if k == last else 1.0
                previous = [j for kk in range(k) for j in local[kk]]
                following = [j for kk in range(k + 1, len(candidate)) for j in local[kk]]
                start = frames(max(previous))[0] if previous else region[0]
                end = frames(min(following))[1] if following else region[1]
            plausibility = _plausibility(recognition, expected, start, end, lang)
            confidence = cost * (1.0 - min(1.0, plausibility / PHONE_PLAUSIBLE_POSTERIOR))
        reports.append({"expected": expected, "heard": "".join(heard), "confidence": confidence})
    return reports


def _as_recognition(heard_phones):
    """Return ``(recognition or None, heard phones, heard confidences)`` for a list of phones or a recognition."""
    if isinstance(heard_phones, PhoneRecognition):
        return heard_phones, list(heard_phones.phones), list(heard_phones.confidences)
    heard = list(heard_phones)
    return None, heard, [1.0] * len(heard)


def _word_reports(heard_phones, text_reference, lang=DEFAULT_LANGUAGE):
    """Compare what was heard with ``text_reference`` word by word.

    Returns one dict per word with phones: ``position``, ``word``, ``expected`` (phones),
    ``actual`` (phones heard for the word), ``distance`` (edit distance to the closest
    accepted pronunciation), ``phones`` (per-phone reports, see :func:`_phone_reports`)
    and ``weighted_edits`` (sum of the per-phone confidences).
    """
    recognition, heard, _ = _as_recognition(heard_phones)
    n_frames = len(recognition.log_posteriors) if recognition is not None else 0
    words, groups = get_expected_phones(text_reference, lang)
    expected = [p for g in groups for p in g]
    alignment = _align(expected, heard)

    reports = []
    offset = 0
    previous_last_phone = None
    for position, (word, group) in enumerate(zip(words, groups)):
        if not group:
            continue
        indices = range(offset, offset + len(group))
        offset += len(group)
        matched = sorted(set().union(*(alignment[i] for i in indices)))
        actual = [heard[j] for j in matched]

        candidates = _pronunciations(word, group, previous_last_phone, lang)
        distances = [Levenshtein.distance(c, actual) for c in candidates]
        distance = min(distances)
        candidate = candidates[distances.index(distance)]
        previous_last_phone = group[-1]

        if recognition is None:
            region = (0, 0)
        elif matched:
            region = (recognition.spans[matched[0]][0], recognition.spans[matched[-1]][1])
        else:
            before = [j for i in range(indices[0]) for j in alignment[i]]
            after = [j for i in range(indices[-1] + 1, len(expected)) for j in alignment[i]]
            region = (recognition.spans[max(before)][0] if before else 0,
                      recognition.spans[min(after)][1] if after else n_frames)
        phone_reports = _phone_reports(candidate, actual, matched, recognition, region, lang) if distance else []
        reports.append({
            "position": position,
            "word": word,
            "expected": list(group),
            "actual": actual,
            "distance": distance,
            "phones": phone_reports,
            "weighted_edits": sum(p["confidence"] for p in phone_reports),
        })
    return reports


def compare_phones(heard_phones, text_reference, lang=DEFAULT_LANGUAGE):
    """Compare recognized phones with the phones expected for ``text_reference`` in ``lang``, word by word.

    ``heard_phones`` is either a plain list of phones or a :class:`PhoneRecognition`
    (from :func:`recognize_phones`). Every wrong phone of a word gets an error
    confidence: 1 for a substitution, ``NEAR_PHONE_COST`` when the two phones are close
    (voicing, tense/lax vowel...), less for a dropped or extra phone at the end of the
    word, and, with a recognition, scaled down when the expected phone was itself
    plausible in the frame posteriors. A word is reported when the confidences add up
    to ``PHONE_ERROR_THRESHOLD`` of its phones or to ``PHONE_ERROR_MIN_EDITS``.

    Returns a dict with ``expected_phones`` (per word), ``heard_phones``,
    ``heard_phones_confidence`` (0-1 per heard phone), ``phone_error_rate``, ``errors``
    (same shape as the text-based errors: ``position``, ``word``, ``expected``,
    ``actual``, ``actual_word``, plus ``phone_distance``, ``confidence`` (0-1, how sure
    we are the word is mispronounced) and ``phones``, a per-phone list of
    ``{expected, heard, confidence}``) and ``words_with_errors``.
    """
    _, heard, heard_confidences = _as_recognition(heard_phones)
    words, groups = get_expected_phones(text_reference, lang)
    expected = [p for g in groups for p in g]
    if expected:
        phone_error_rate = Levenshtein.distance(expected, heard) / len(expected)
    else:
        phone_error_rate = 1.0 if heard else 0.0

    errors = []
    words_with_errors = []
    for report in _word_reports(heard_phones, text_reference, lang):
        edits, length = report["weighted_edits"], len(report["expected"])
        if report["distance"] and (edits / length >= PHONE_ERROR_THRESHOLD or edits >= PHONE_ERROR_MIN_EDITS):
            errors.append({
                "position": report["position"],
                "word": report["word"],
                "expected": "".join(report["expected"]),
                "actual": "".join(report["actual"]),
                "actual_word": "",
                "phone_distance": report["distance"],
                "confidence": round(min(1.0, max(edits / length, edits / PHONE_ERROR_MIN_EDITS)), 3),
                "phones": [dict(p, confidence=round(p["confidence"], 3)) for p in report["phones"]],
            })
            words_with_errors.append(report["word"])

    return {
        "expected_phones": [list(g) for g in groups],
        "heard_phones": heard,
        "heard_phones_confidence": [round(c, 3) for c in heard_confidences],
        "phone_error_rate": round(phone_error_rate, 4),
        "errors": errors,
        "words_with_errors": words_with_errors,
    }
