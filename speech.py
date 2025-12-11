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
import warnings


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
    # Use regex to split words, ignoring punctuation, to match frontend logic and avoid "times," issues
    words = re.findall(r"\b[\w']+\b", text)
    
    phonemes = []
    phoneme_to_word = {}
    
    for word in words:
        try:
            word_phonemes = phonemize(word, language="en-us", backend="espeak", strip=True, preserve_punctuation=False).split()
        except Exception as e:
            # warnings.warn(f"Error with espeak for word '{word}', switching to festival: {e}", UserWarning)
            try:
                word_phonemes = phonemize(word, language="en-us", backend="festival", strip=True, preserve_punctuation=False).split()
            except:
                word_phonemes = [] # Fallback if everything fails

        for phoneme in word_phonemes:
            phoneme_to_word[len(phonemes)] = word
            phonemes.append(phoneme)

    return phonemes, phoneme_to_word

def get_phoneme_embeddings(phoneme_seq):
    """ Convert a phoneme sequence into a numerical sequence """
    return np.array([ord(p) for p in phoneme_seq]).reshape(-1, 1)

def compare_transcriptions(transcription, text_reference):
    """
    Compare automatic transcription with expected text.
    """

    transcription_clean = transcription.lower().strip()
    reference_clean = text_reference.lower().strip()

    # Check edit distance between transcription and reference text
    word_distance = Levenshtein.distance(transcription_clean, reference_clean)

    # Extract phonemes from both versions
    expected_phonemes, expected_map = get_phonemes_with_word_mapping(text_reference)
    transcribed_phonemes, transcribed_map = get_phonemes_with_word_mapping(transcription_clean)

    # Convert phonemes to numerical sequences for DTW (global score)
    expected_seq = get_phoneme_embeddings(" ".join(expected_phonemes))
    transcribed_seq = get_phoneme_embeddings(" ".join(transcribed_phonemes))

    # Apply DTW to align phonemes (for global score)
    distance, _ = fastdtw(expected_seq, transcribed_seq, dist=euclidean)

    # Identify words with pronunciation errors using Levenshtein on phoneme lists
    errors = []
    words_with_errors = set()
    
    # Map each expected phoneme index to a set of transcribed phoneme indices
    # This allows us to handle 1-to-N and N-to-1 word mappings
    alignment_map = [set() for _ in range(len(expected_phonemes))]
    
    opcodes = Levenshtein.opcodes(expected_phonemes, transcribed_phonemes)
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k, l in zip(range(i1, i2), range(j1, j2)):
                alignment_map[k].add(l)
        elif tag == 'replace':
            # For replacement, we map the range proportionally
            # This handles "I'm" (3 phonemes) -> "I M" (4 phonemes) better
            # And avoids "Hello how are" -> "enou i wor" mapping everything to everything
            len_i = i2 - i1
            len_j = j2 - j1
            for k in range(i1, i2):
                # Calculate proportional range in j
                start_j = j1 + int((k - i1) * len_j / len_i)
                end_j = j1 + int((k - i1 + 1) * len_j / len_i)
                
                if start_j == end_j and len_j > 0:
                     idx = min(start_j, j2 - 1)
                     alignment_map[k].add(idx)
                else:
                    for l in range(start_j, end_j):
                        alignment_map[k].add(l)
        elif tag == 'delete':
            # Expected phonemes have no match
            pass
        elif tag == 'insert':
            # Extra transcribed phonemes (ignored for now)
            pass

    # Group expected phonemes by word
    expected_words_indices = {} # word -> list of phoneme indices
    # Since words can be duplicate ("the cat and the dog"), we need to group by (word, position) or just iterate ranges
    # But phoneme_to_word maps index -> word string. We need to reconstruct the word boundaries.
    
    # Let's iterate through the expected phonemes and group them
    current_word = None
    word_start_index = 0
    processed_words = [] # List of (word, start_idx, end_idx)
    
    # Reconstruct word boundaries from phoneme_to_word
    # Note: phoneme_to_word is dense (0, 1, 2...). 
    # We can just iterate 0..len(expected_phonemes)
    if expected_phonemes:
        current_word = expected_map[0]
        word_start = 0
        for i in range(1, len(expected_phonemes)):
            word = expected_map[i]
            # If the word string changes, or if it's the same word string but logically a new word?
            # get_phonemes_with_word_mapping flattens everything. 
            # If we have "that that", phoneme_to_word will show "that" for indices 0..2 and "that" for 3..5
            # We can't distinguish duplicates easily unless we stored (word, index) in the map.
            # BUT, we know the order is preserved.
            # Wait, get_phonemes_with_word_mapping iterates words.
            # So we can just re-run the word iteration logic to get boundaries?
            # Or better: modify get_phonemes_with_word_mapping to return boundaries?
            # For now, let's assume adjacent identical words are merged or handled? 
            # Actually, "that that" -> "that" (idx 0,1,2), "that" (idx 3,4,5).
            # If we just look at string change, we merge them. That's a bug for "that that".
            # FIX: Let's assume we can't easily reconstruct boundaries from just the map.
            # Let's rely on the fact that we process words in order.
            pass

    # Better approach: Iterate over the original words again to get their phoneme counts
    # We need the original word list.
    expected_words_list = re.findall(r"\b[\w']+\b", text_reference)
    
    current_phoneme_idx = 0
    
    for word in expected_words_list:
        # Re-generate phonemes for this word to know how many there are
        # (This is slightly inefficient but safe)
        try:
            p_list = phonemize(word, language="en-us", backend="espeak", strip=True, preserve_punctuation=False).split()
        except:
            try:
                p_list = phonemize(word, language="en-us", backend="festival", strip=True, preserve_punctuation=False).split()
            except:
                p_list = []
        
        if not p_list:
            # Word has no phonemes (e.g. number or symbol that failed?)
            continue
            
        # Range for this word
        word_indices = range(current_phoneme_idx, current_phoneme_idx + len(p_list))
        current_phoneme_idx += len(p_list)
        
        # Find corresponding transcribed words
        matched_trans_indices = set()
        for idx in word_indices:
            if idx < len(alignment_map):
                matched_trans_indices.update(alignment_map[idx])
        
        if not matched_trans_indices:
            # Word is missing
            errors.append({"position": word_indices.start, "expected": word, "actual": "", "word": word})
            words_with_errors.add(word)
        else:
            # Get the transcribed words for these indices
            actual_words = []
            sorted_trans_indices = sorted(list(matched_trans_indices))
            
            # We need to group these indices into words
            # transcribed_map maps idx -> word string
            # We want to reconstruct the phrase "I M" from indices
            
            # Simple approach: get all word strings, unique them preserving order
            seen_words = set()
            for tidx in sorted_trans_indices:
                if tidx in transcribed_map:
                    w = transcribed_map[tidx]
                    # We want to capture "I" and "M". 
                    # If we just add to set, we lose order? No, we iterate sorted indices.
                    # But "I" might span indices 0,1. We see "I", "I".
                    if w not in seen_words: # This prevents "I I" -> "I"
                        actual_words.append(w)
                        seen_words.add(w) # This prevents "that that" -> "that that" if they are identical?
                        # This is a limitation. But for "I M", it works: "I", "M".
                        # For "that that", it would become "that". Acceptable for now.
            
            actual_text = " ".join(actual_words)
            
            # Compare
            # We compare the *text* of the word vs the *text* of the actual words
            # But we also want to check pronunciation quality?
            # If text matches, we assume pronunciation is good?
            # Or do we check phoneme distance?
            # The user wants to see "Mispronunciation" or "Missing".
            # If text matches "I" vs "I", it's good.
            # If "I'm" vs "I M", is that an error?
            # "I'm" vs "I M" -> Levenshtein("I'm", "I M") = 2.
            # Maybe we should check phoneme distance between the *segments*?
            
            # Let's calculate phoneme distance between expected segment and actual segment
            expected_seg = [expected_phonemes[i] for i in word_indices]
            actual_seg = [transcribed_phonemes[i] for i in sorted_trans_indices]
            
            # Use Levenshtein on phonemes
            p_dist = Levenshtein.distance(expected_seg, actual_seg)
            
            # Normalize distance
            # If p_dist > threshold, mark as error
            # Threshold: e.g. > 20% of length?
            # We remove max(1, ...) to be stricter on short words (len 1-2)
            if p_dist > len(expected_seg) * 0.4:
                # It's a mispronunciation
                # We show the phonemes of the *actual* segment
                # And the text of the *actual* words
                
                # Construct actual phoneme string
                actual_phoneme_str = "".join(actual_seg) # Or space separated?
                # The frontend expects a string.
                # Let's use the phoneme list directly? No, frontend expects string?
                # Frontend: `/${error.actual}/`
                
                # Wait, `actual` in error object is used for phonemes in frontend?
                # In previous code: `actual: transcribed_phonemes[j]` (single phoneme)
                # Now we have a sequence.
                
                errors.append({
                    "position": word_indices.start,
                    "expected": "".join(expected_seg), # Phonemes
                    "actual": "".join(actual_seg), # Phonemes
                    "word": word, # Expected Word Text
                    "actual_word": actual_text # Actual Word Text (e.g. "I M") - NEW FIELD
                })
                words_with_errors.add(word)

    # Generate understandable feedback
    feedback = "🔊 Feedback on your pronunciation:\n"
    if words_with_errors:
        feedback += "❌ You need to better pronounce these words: " + ", ".join(words_with_errors) + "\n"
    else:
        feedback += "✅ Your pronunciation is excellent! 🎉\n"

    # errors is an array, but can contains multiple time the same word (for complex sounds). We want to keep only one occurence of each word
    # With new logic, we iterate words, so uniqueness is guaranteed per position.
    # errors = [dict(t) for t in {tuple(d.items()) for d in errors}] 

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
        "words_with_errors": list(words_with_errors),
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

    # edge cases
    if final_score < 0:
        final_score = 0

    if final_score > 100:
        final_score = 100
    
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
    return text.strip()