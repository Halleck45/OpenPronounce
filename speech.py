import numpy as np
import torchaudio
import torch
import audio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from phonemizer import phonemize
import Levenshtein
import re
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Load Wav2Vec2
MODEL_NAME = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()

# for transcribing
#MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
modelCTC = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
modelCTC.eval()


def extract_embeddings(audio_waveform, sampling_rate=16000):
    """
    Extract raw Wav2Vec2 embeddings for a given audio input.
    """
    # Ensure audio is float32 and squeeze unnecessary dimensions
    #audio_waveform = audio_waveform.squeeze().float()

    # Transform audio into input for Wav2Vec2
    inputs = processor(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Check shape before sending to model
    input_values = inputs.input_values
    if len(input_values.shape) > 2:  # Remove unnecessary dimensions
        input_values = input_values.squeeze(0)

    with torch.no_grad():
        features = model(input_values).last_hidden_state  # (batch, time, features)

    return features.squeeze(0).numpy()


def get_phonemes_with_word_mapping(text):
    """ Return a list of phonemes and their associated words """
    words = text.split()  # List of words
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True, preserve_punctuation=False).split()

    # Associate each phoneme with a word (naively based on split)
    phoneme_to_word = {}
    phoneme_index = 0
    for word in words:
        word_phonemes = phonemize(word, language="en-us", backend="espeak", strip=True, preserve_punctuation=False).split()
        for phoneme in word_phonemes:
            phoneme_to_word[phoneme_index] = word
            phoneme_index += 1

    return phonemes, phoneme_to_word

def get_phonemes(text):
    """
    Convert a text into a sequence of phonemes using phonemizer.
    """
    # Clean the text to avoid errors
    text = text.strip().lower()

    # First test with `espeak`, then fallback to `espeak-ng` if error
    try:
        phonemes = phonemize(
            text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=False  # Disable punctuation that may cause issues
        )
    except Exception as e:
        print(f"⚠️ Error with espeak, switching to espeak-ng: {e}")
        phonemes = phonemize(
            text,
            language="en-us",
            backend="espeak-ng",
            strip=True,
            preserve_punctuation=False
        )

    return phonemes.split(" ")



def get_phoneme_embeddings(phoneme_seq):
    """ Convert a phoneme sequence into a numerical sequence """
    return np.array([ord(p) for p in phoneme_seq]).reshape(-1, 1)

def compare_pronunciation(expected, actual):
    """ Compare pronunciation with DTW and return a score """
    expected_seq = get_phoneme_embeddings(expected)
    actual_seq = get_phoneme_embeddings(actual)

    distance, _ = fastdtw(expected_seq, actual_seq, dist=euclidean)
    
    return distance

def compare_transcriptions(transcription, text_reference):
    """
    Compare automatic transcription with expected text.
    """

    transcription_clean = transcription.lower().strip()
    reference_clean = text_reference.lower().strip()

    # Check edit distance between transcription and reference text
    word_distance = Levenshtein.distance(transcription_clean, reference_clean)

    # Extract phonemes from both versions
    expected_phonemes, phoneme_to_word = get_phonemes_with_word_mapping(text_reference)
    transcribed_phonemes, _ = get_phonemes_with_word_mapping(transcription_clean)

    # Convert phonemes to numerical sequences
    expected_seq = get_phoneme_embeddings(" ".join(expected_phonemes))
    transcribed_seq = get_phoneme_embeddings(" ".join(transcribed_phonemes))

    # Apply DTW to align phonemes
    distance, path = fastdtw(expected_seq, transcribed_seq, dist=euclidean)

    # Identify words with pronunciation errors
    errors = []
    words_with_errors = set()
    for (i, j) in path:
        if i >= len(expected_phonemes) or j >= len(transcribed_phonemes):
            continue
        
        diff = Levenshtein.distance(expected_phonemes[i], transcribed_phonemes[j])
        if diff > 1:  # Ajuster le seuil selon la tolérance
            word = phoneme_to_word.get(i, "UNKNOWN")
            errors.append({"position": i, "expected": expected_phonemes[i], "actual": transcribed_phonemes[j], "word": word})
            words_with_errors.add(word)

    # Generate understandable feedback
    feedback = "🔊 Feedback on your pronunciation:\n"
    if words_with_errors:
        feedback += "❌ You need to better pronounce these words: " + ", ".join(words_with_errors) + "\n"
    else:
        feedback += "✅ Your pronunciation is excellent! 🎉\n"

    # errors is an array, but can contains multiple time the same word (for complex sounds). We want to keep only one occurence of each word
    errors = [dict(t) for t in {tuple(d.items()) for d in errors}]

    # Convert vectors to JSON for later display of expected and obtained traces
    expected_vector = expected_seq.tolist()
    transcribed_vector = transcribed_seq.tolist()

    # Alignement avec DTW (pour les durées différentes)
    expected_vector, transcribed_vector = align_sequences_dtw(expected_vector, transcribed_vector)

    return {
        "word_distance": word_distance,
        "phoneme_distance": distance,
        "errors": errors,
        "feedback": feedback,
        "transcribe": transcription,
        "expected_vector": expected_vector.astype(float).tolist(),
        "transcribed_vector": transcribed_vector.astype(float).tolist(),
        "expected_phonemes": expected_phonemes,
        "transcribed_phonemes": transcribed_phonemes,
        "words_with_errors": words_with_errors,
    }

def align_sequences_dtw(seq1, seq2):
    """
    Align two sequences of numerical values using Dynamic Time Warping (DTW).
    Returns the interpolated sequences to have the same length.
    This allows for easier comparison of the two sequences, as one may be faster than the other,
    or shorter.
    """
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    
    aligned_seq1 = []
    aligned_seq2 = []

    for i, j in path:
        aligned_seq1.append(seq1[i][0])  # Preserve the first dimension
        aligned_seq2.append(seq2[j][0])

    # we amplify the difference artificially, otherwise the two curves often overlap
    # aligned_seq2 = aligned_seq2 + (aligned_seq2 - aligned_seq1) * 2

    return np.array(aligned_seq1), np.array(aligned_seq2)

def compute_pronunciation_score(distance_dtw, phoneme_distance, word_distance, max_dtw=500, max_lev=30):
    """
    Calculate a score out of 100 by normalizing distances.
    """
    # Normalization of distances
    dtw_score = max(0, 100 - (distance_dtw / max_dtw) * 100)
    phoneme_score = max(0, 100 - (phoneme_distance / max_dtw) * 100)
    word_score = max(0, 100 - (word_distance / max_lev) * 100)
    
    # Ponderate the different components: DTW 40%, Phonemes 30%, Words 30%
    final_score = 0.4 * dtw_score + 0.3 * phoneme_score + 0.3 * word_score
    
    return round(final_score, 2)

def compare_audio_with_text(audio_1, text_reference, sampling_rate=16000):
    """
    Compare a user's pronunciation with a text reference.
    """

    # Extract Wav2Vec2 embeddings from user audio
    emb_1 = extract_embeddings(audio_1, sampling_rate)

    # Generate a reference audio (via TTS) and extract its embeddings
    reference_file = audio.text2speech(text_reference)

    # Generate the reference audio (via TTS) and extract its embeddings
    # Assume here that you already have a `reference.wav` file generated from the text.
    audio_2, sr = torchaudio.load(reference_file)
    emb_2 = extract_embeddings(audio_2, sr)

    # Apply DTW to align the embeddings
    distance, path = fastdtw(emb_1, emb_2, dist=euclidean)
    distance = int(distance)  # Convert to int for easier reading
    
    # Convert the reference text into phonemes and get the word-phoneme mapping
    expected_phonemes, phoneme_to_word = get_phonemes_with_word_mapping(text_reference)
    transcription = transcribe(audio_1)

    # Identify divergences between expected phonemes and transcribed phonemes
    differences = compare_transcriptions(transcription, text_reference)

    score = compute_pronunciation_score(distance, differences["phoneme_distance"], differences["word_distance"])

    # prosody
    energy = extract_energy(audio_1)
    f0 = interpolate_f0(extract_f0(audio_1, sampling_rate))

    return {
        "score": score,
        "distance": distance, 
        "differences": differences, 
        "feedback": differences["feedback"],
        "transcribe": differences["transcribe"],
        "prosody": {
            "f0": f0.tolist(),
            "energy": energy.tolist()
        }
    }


def extract_f0(audio_waveform, sr=16000):
    """ Extract the fundamental frequency F0 from the audio """
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_waveform, fmin=50, fmax=300)
    f0 = np.nan_to_num(f0)  # Replace NaNs with 0
    return f0

def extract_energy(audio_waveform):
    """ Extract and normalize the energy of the audio """
    energy = librosa.feature.rms(y=audio_waveform)
    scaler = MinMaxScaler(feature_range=(0, 250))  # Scale between 0 and 250 to match F0
    energy_scaled = scaler.fit_transform(energy.T).flatten()
    return energy_scaled

def interpolate_f0(f0):
    """ Interpolate missing F0 values to avoid gaps in the graph """
    f0 = np.array(f0)
    mask = f0 > 0  # Keep only valid values
    f0_interp = np.interp(np.arange(len(f0)), np.where(mask)[0], f0[mask])
    return f0_interp


def transcribe(audio):
    """ Transcribe the audio into text with Wav2Vec2 """
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = modelCTC(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

def clean_transcription(text):
    """ Clean the transcription text """
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z' ]+", "", text)  # Remove special characters
    text = text.replace("  ", " ")  # Avoid multiple spaces
    return text