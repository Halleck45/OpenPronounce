"""Pronunciation assessment: Wav2Vec2 embeddings, phonemization, alignment and scoring."""

import logging
import re
from functools import lru_cache

import Levenshtein
import librosa
import numpy as np
import torch
from fastdtw import fastdtw
from phonemizer import phonemize
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

from . import audio, phones
from .device import get_device
from .languages import DEFAULT_LANGUAGE, get_language

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/wav2vec2-large-960h"
SAMPLING_RATE = 16000

# Threshold above which a word is considered mispronounced:
# ratio of edited phonemes over the number of expected phonemes.
WORD_ERROR_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Models (loaded lazily, once)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_models(model_name=MODEL_NAME):
    """Load a Wav2Vec2 processor and CTC model on first use (cached per checkpoint).

    ``Wav2Vec2ForCTC`` embeds a ``Wav2Vec2Model`` (``.wav2vec2``), so a single
    checkpoint serves both transcription (CTC head) and embedding extraction.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    logger.info("Loading %s", model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model_ctc = Wav2Vec2ForCTC.from_pretrained(model_name).to(get_device())
    model_ctc.eval()
    return processor, model_ctc


def _get_processor(lang=DEFAULT_LANGUAGE):
    return _load_models(get_language(lang).asr_model)[0]


def _get_model_ctc(lang=DEFAULT_LANGUAGE):
    return _load_models(get_language(lang).asr_model)[1]


def _get_model():
    """Embedding extractor: always the English checkpoint, whatever the language."""
    return _load_models(MODEL_NAME)[1].wav2vec2


# ---------------------------------------------------------------------------
# Embeddings & transcription
# ---------------------------------------------------------------------------

def extract_embeddings(audio_waveform, sampling_rate=SAMPLING_RATE):
    """Extract raw Wav2Vec2 hidden states, shape (frames, features)."""
    inputs = _get_processor()(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values
    if len(input_values.shape) > 2:
        input_values = input_values.squeeze(0)

    with torch.no_grad():
        features = _get_model()(input_values.to(get_device())).last_hidden_state  # (batch, time, features)

    return features.squeeze(0).cpu().numpy()


def transcribe(audio_waveform, lang=DEFAULT_LANGUAGE):
    """Transcribe a 16 kHz waveform into text with the Wav2Vec2 CTC model of ``lang``.

    The English model emits upper-case text; the other checkpoints emit lower-case.
    """
    processor = _get_processor(lang)
    inputs = processor(audio_waveform, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = _get_model_ctc(lang)(inputs.input_values.to(get_device())).logits
    predicted_ids = torch.argmax(logits, dim=-1).cpu()
    return processor.batch_decode(predicted_ids)[0]


def clean_transcription(text):
    """Lower-case, strip and keep only letters, apostrophes and single spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z' ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Phonemization
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[\w']+\b")


def _words(text):
    """Split text into words, ignoring punctuation (mirrors the frontend logic)."""
    return _WORD_RE.findall(text)


@lru_cache(maxsize=4096)
def _phonemize_word(word, lang=DEFAULT_LANGUAGE):
    """Return the IPA phonemes of a single word (espeak first, festival as fallback).

    The word is lower-cased first: espeak spells upper-case words letter by letter
    ("IT" -> /aɪtiː/), which would flag every word of an upper-case sentence as wrong.
    """
    word = word.lower()
    language = get_language(lang).espeak
    for backend in ("espeak", "festival"):
        try:
            return tuple(
                phonemize(word, language=language, backend=backend, strip=True, preserve_punctuation=False).split()
            )
        except Exception as e:  # noqa: BLE001 - try the next backend
            logger.debug("phonemize(%r) failed with %s: %s", word, backend, e)
    return ()


def get_phonemes(text, lang=DEFAULT_LANGUAGE):
    """Return the flat list of phonemes for ``text``."""
    return get_phonemes_with_word_mapping(text, lang)[0]


def get_phonemes_with_word_mapping(text, lang=DEFAULT_LANGUAGE):
    """Return ``(phonemes, phoneme_to_word)`` where ``phoneme_to_word[i]`` is the word phoneme ``i`` belongs to."""
    phonemes = []
    phoneme_to_word = {}
    for word in _words(text):
        for phoneme in _phonemize_word(word, lang):
            phoneme_to_word[len(phonemes)] = word
            phonemes.append(phoneme)
    return phonemes, phoneme_to_word


def get_phoneme_embeddings(phoneme_seq):
    """Turn a phoneme sequence into a numeric (n, 1) array (codepoints) usable by DTW."""
    return np.array([ord(p) for p in phoneme_seq], dtype=float).reshape(-1, 1)


def compare_pronunciation(expected_phonemes, actual_phonemes):
    """Edit distance between two phoneme sequences."""
    return float(Levenshtein.distance(list(expected_phonemes), list(actual_phonemes)))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _align_phoneme_indices(expected_phonemes, transcribed_phonemes):
    """Map every expected phoneme index to the set of transcribed phoneme indices it aligns with."""
    alignment_map = [set() for _ in range(len(expected_phonemes))]

    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(expected_phonemes, transcribed_phonemes):
        if tag == "equal":
            for k, l in zip(range(i1, i2), range(j1, j2)):
                alignment_map[k].add(l)
        elif tag == "replace":
            # Spread the replaced range proportionally, so that "I'm" (3 phonemes)
            # can align with "I M" (4 phonemes) without mapping everything to everything.
            len_i, len_j = i2 - i1, j2 - j1
            for k in range(i1, i2):
                start_j = j1 + int((k - i1) * len_j / len_i)
                end_j = j1 + int((k - i1 + 1) * len_j / len_i)
                if start_j == end_j and len_j > 0:
                    alignment_map[k].add(min(start_j, j2 - 1))
                else:
                    alignment_map[k].update(range(start_j, end_j))
        elif tag == "insert":
            # Extra transcribed words ("hell no" for "hello") are attached to the
            # previous expected word so the feedback shows everything that was heard.
            k = i1 - 1 if i1 > 0 else i1
            if k < len(alignment_map):
                alignment_map[k].update(range(j1, j2))
        # "delete" (missing expected phonemes) produces no mapping.

    return alignment_map


def compare_transcriptions(transcription, text_reference, lang=DEFAULT_LANGUAGE):
    """Compare an automatic transcription with the expected text, word by word.

    Returns a JSON-serializable dict with distances, per-word errors and feedback.
    """
    transcription_clean = transcription.lower().strip()
    reference_clean = text_reference.lower().strip()

    word_distance = Levenshtein.distance(transcription_clean, reference_clean)

    expected_phonemes, _ = get_phonemes_with_word_mapping(text_reference, lang)
    transcribed_phonemes, transcribed_map = get_phonemes_with_word_mapping(transcription_clean, lang)

    # Global phoneme distance (DTW on codepoints, kept for the score and the charts)
    expected_seq = get_phoneme_embeddings(" ".join(expected_phonemes))
    transcribed_seq = get_phoneme_embeddings(" ".join(transcribed_phonemes))
    if len(expected_seq) and len(transcribed_seq):
        distance, _ = fastdtw(expected_seq, transcribed_seq, dist=euclidean)
    else:
        distance = float(max(len(expected_seq), len(transcribed_seq)))

    alignment_map = _align_phoneme_indices(expected_phonemes, transcribed_phonemes)

    errors = []
    words_with_errors = []
    current_phoneme_idx = 0

    for word in _words(text_reference):
        word_phonemes = _phonemize_word(word, lang)
        if not word_phonemes:
            continue

        word_indices = range(current_phoneme_idx, current_phoneme_idx + len(word_phonemes))
        current_phoneme_idx += len(word_phonemes)

        matched = set()
        for idx in word_indices:
            if idx < len(alignment_map):
                matched.update(alignment_map[idx])

        if not matched:
            errors.append({
                "position": word_indices.start,
                "expected": "".join(word_phonemes),
                "actual": "",
                "word": word,
                "actual_word": "",
            })
            words_with_errors.append(word)
            continue

        sorted_matched = sorted(matched)
        actual_words = []
        for tidx in sorted_matched:
            w = transcribed_map.get(tidx)
            if w is not None and (not actual_words or actual_words[-1] != w):
                actual_words.append(w)

        expected_seg = [expected_phonemes[i] for i in word_indices]
        actual_seg = [transcribed_phonemes[i] for i in sorted_matched]

        if Levenshtein.distance(expected_seg, actual_seg) > len(expected_seg) * WORD_ERROR_THRESHOLD:
            errors.append({
                "position": word_indices.start,
                "expected": "".join(expected_seg),
                "actual": "".join(actual_seg),
                "word": word,
                "actual_word": " ".join(actual_words),
            })
            words_with_errors.append(word)

    # De-duplicate while preserving order
    words_with_errors = list(dict.fromkeys(words_with_errors))

    expected_words = [w.lower() for w in _words(text_reference)]
    transcribed_words = [w.lower() for w in _words(transcription_clean)]
    word_error_rate = Levenshtein.distance(expected_words, transcribed_words) / max(1, len(expected_words))
    phoneme_error_rate = Levenshtein.distance(expected_phonemes, transcribed_phonemes) / max(1, len(expected_phonemes))

    feedback = _feedback(words_with_errors)

    expected_vector, transcribed_vector = align_sequences_dtw(expected_seq.tolist(), transcribed_seq.tolist())

    return {
        "word_distance": word_distance,
        "phoneme_distance": distance,
        "word_error_rate": round(word_error_rate, 4),
        "phoneme_error_rate": round(phoneme_error_rate, 4),
        "errors": errors,
        "feedback": feedback,
        "transcribe": transcription,
        "expected_vector": expected_vector.astype(float).tolist(),
        "transcribed_vector": transcribed_vector.astype(float).tolist(),
        "expected_phonemes": expected_phonemes,
        "transcribed_phonemes": transcribed_phonemes,
        "words_with_errors": words_with_errors,
    }


def _feedback(words_with_errors):
    feedback = "🔊 Feedback on your pronunciation:\n"
    if words_with_errors:
        feedback += "❌ You need to better pronounce these words: " + ", ".join(words_with_errors) + "\n"
    else:
        feedback += "✅ Your pronunciation is excellent! 🎉\n"
    return feedback


def align_sequences_dtw(seq1, seq2):
    """Align two 1-D numeric sequences (given as lists of ``[x]``) with DTW.

    Returns two arrays of identical length, so that curves of different
    durations can be plotted on top of each other.
    """
    if not len(seq1) or not len(seq2):
        return np.array([]), np.array([])

    _, path = fastdtw(seq1, seq2, dist=euclidean)
    aligned_seq1 = np.array([seq1[i][0] for i, _ in path])
    aligned_seq2 = np.array([seq2[j][0] for _, j in path])
    return aligned_seq1, aligned_seq2


# Mean per-step DTW distance between Wav2Vec2 frames of the learner and of the TTS
# reference. Measured on the bundled samples: ~6 for a clean native-like reading,
# ~10 for a good reading by a different voice, ~12-15 for a wrong sentence.
ACOUSTIC_DISTANCE_GOOD = 5.0
ACOUSTIC_DISTANCE_BAD = 15.0

SCORE_WEIGHTS = {"acoustic": 0.2, "phonemes": 0.5, "words": 0.3}


def compute_pronunciation_score(acoustic_distance, phoneme_error_rate, word_error_rate):
    """Combine length-independent measures into a 0-100 score.

    - ``acoustic_distance``: mean per-step DTW distance between Wav2Vec2 embeddings
      (see :data:`ACOUSTIC_DISTANCE_GOOD` / :data:`ACOUSTIC_DISTANCE_BAD`), 20%.
    - ``phoneme_error_rate``: edited phonemes / expected phonemes, 50%.
    - ``word_error_rate``: edited words / expected words, 30%.

    Every component is clipped to [0, 100] before weighting.
    """
    span = ACOUSTIC_DISTANCE_BAD - ACOUSTIC_DISTANCE_GOOD
    acoustic_score = 100 * (1 - (acoustic_distance - ACOUSTIC_DISTANCE_GOOD) / span)
    phoneme_score = 100 * (1 - phoneme_error_rate)
    word_score = 100 * (1 - word_error_rate)

    clip = lambda x: min(100.0, max(0.0, x))  # noqa: E731
    final_score = (
        SCORE_WEIGHTS["acoustic"] * clip(acoustic_score)
        + SCORE_WEIGHTS["phonemes"] * clip(phoneme_score)
        + SCORE_WEIGHTS["words"] * clip(word_score)
    )
    return round(clip(final_score), 2)


def compare_audio_with_text(audio_1, text_reference, sampling_rate=SAMPLING_RATE, use_phone_model=None,
                            lang=DEFAULT_LANGUAGE):
    """Assess how well ``audio_1`` (16 kHz mono waveform) pronounces ``text_reference``.

    ``lang`` selects the language (see :data:`openpronounce.languages.LANGUAGES`);
    it drives the reference TTS voice, the phonemizer and the word transcription model.

    Returns a JSON-serializable dict with ``score`` (0-100), ``distance``,
    ``differences`` (per-word errors, phonemes, feedback), ``transcribe``,
    ``language`` and ``prosody`` (``f0`` and ``energy`` contours).

    When the phone recognizer is enabled (default, see :mod:`openpronounce.phones`),
    ``differences.errors`` and ``differences.phoneme_error_rate`` come from phones
    recognized directly in the audio; otherwise they are derived from the word
    transcription.
    """
    if use_phone_model is None:
        use_phone_model = phones.is_enabled()
    lang = get_language(lang).code

    emb_1 = extract_embeddings(audio_1, sampling_rate)

    reference_file = audio.text2speech(text_reference, lang=lang)
    audio_2 = audio.load(reference_file, sr=sampling_rate)
    emb_2 = extract_embeddings(audio_2, sampling_rate)

    distance, path = fastdtw(emb_1, emb_2, dist=euclidean)
    acoustic_distance = distance / max(1, len(path))
    distance = int(distance)

    transcription = transcribe(audio_1, lang)
    differences = compare_transcriptions(transcription, text_reference, lang)

    if use_phone_model:
        recognition = phones.recognize_phones(audio_1, sampling_rate, lang=lang)
        phone_result = phones.compare_phones(recognition, text_reference, lang)
        differences.update({
            "errors": phone_result["errors"],
            "words_with_errors": phone_result["words_with_errors"],
            "phoneme_error_rate": phone_result["phone_error_rate"],
            "expected_phones": phone_result["expected_phones"],
            "heard_phones": phone_result["heard_phones"],
            "heard_phones_confidence": phone_result["heard_phones_confidence"],
            "feedback": _feedback(phone_result["words_with_errors"]),
        })

    score = compute_pronunciation_score(
        acoustic_distance, differences["phoneme_error_rate"], differences["word_error_rate"]
    )

    energy = extract_energy(audio_1)
    f0 = interpolate_f0(extract_f0(audio_1, sampling_rate))

    return {
        "score": score,
        "distance": distance,
        "acoustic_distance": round(acoustic_distance, 3),
        "differences": differences,
        "feedback": differences["feedback"],
        "transcribe": differences["transcribe"],
        "language": lang,
        "prosody": {
            "f0": f0.tolist(),
            "energy": energy.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Prosody
# ---------------------------------------------------------------------------

def extract_f0(audio_waveform, sr=SAMPLING_RATE):
    """Fundamental frequency (pitch) contour, unvoiced frames set to 0."""
    f0, _, _ = librosa.pyin(audio_waveform, fmin=50, fmax=300, sr=sr)
    return np.nan_to_num(f0)


def extract_energy(audio_waveform):
    """RMS energy contour scaled to 0-250 so it can share an axis with F0."""
    energy = librosa.feature.rms(y=audio_waveform)
    scaler = MinMaxScaler(feature_range=(0, 250))
    return scaler.fit_transform(energy.T).flatten()


def interpolate_f0(f0):
    """Linearly interpolate unvoiced (0) frames so the pitch curve has no gaps."""
    f0 = np.array(f0, dtype=float)
    mask = f0 > 0
    if not mask.any():
        return f0
    return np.interp(np.arange(len(f0)), np.where(mask)[0], f0[mask])
