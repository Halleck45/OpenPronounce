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


# Charger le modèle Wav2Vec2
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
    Extrait les embeddings bruts de Wav2Vec2 pour une entrée audio donnée.
    """
    # Assurer que l'audio est en float32 et squeeze les dimensions inutiles
    #audio_waveform = audio_waveform.squeeze().float()

    # Transformer l'audio en entrée pour Wav2Vec2
    inputs = processor(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Vérifier la forme avant d'envoyer au modèle
    input_values = inputs.input_values
    if len(input_values.shape) > 2:  # Supprimer les dimensions inutiles
        input_values = input_values.squeeze(0)

    with torch.no_grad():
        features = model(input_values).last_hidden_state  # (batch, time, features)

    return features.squeeze(0).numpy()


def get_phonemes_with_word_mapping(text):
    """ Retourne la liste des phonèmes et leur mot associé """
    words = text.split()  # Liste des mots
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True, preserve_punctuation=False).split()
    
    # Associer chaque phonème à un mot (naïvement basé sur le split)
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
    Convertit un texte en séquence de phonèmes avec phonemizer.
    """
    # Nettoyer le texte pour éviter des erreurs
    text = text.strip().lower()
    
    # Tester d'abord avec `espeak`, puis fallback vers `espeak-ng` si erreur
    try:
        phonemes = phonemize(
            text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=False  # Désactiver la ponctuation qui peut poser problème
        )
    except Exception as e:
        print(f"⚠️ Erreur avec espeak, passage à espeak-ng: {e}")
        phonemes = phonemize(
            text,
            language="en-us",
            backend="espeak-ng",
            strip=True,
            preserve_punctuation=False
        )

    return phonemes.split(" ")



def get_phoneme_embeddings(phoneme_seq):
    """ Convertit une séquence phonémique en séquence numérique """
    return np.array([ord(p) for p in phoneme_seq]).reshape(-1, 1)

def compare_pronunciation(expected, actual):
    """ Compare la prononciation avec DTW et retourne un score """
    expected_seq = get_phoneme_embeddings(expected)
    actual_seq = get_phoneme_embeddings(actual)

    distance, _ = fastdtw(expected_seq, actual_seq, dist=euclidean)
    
    return distance

def compare_transcriptions(transcription, text_reference):
    """
    Compare la transcription automatique avec le texte attendu.
    """

    transcription_clean = transcription.lower().strip()
    reference_clean = text_reference.lower().strip()

    # Vérifier la distance d'édition entre la transcription et le texte attendu
    word_distance = Levenshtein.distance(transcription_clean, reference_clean)

    # Extraire les phonèmes des deux versions
    expected_phonemes, phoneme_to_word = get_phonemes_with_word_mapping(text_reference)
    transcribed_phonemes, _ = get_phonemes_with_word_mapping(transcription_clean)

    # convertir les phonèmes en séquences numériques
    expected_seq = get_phoneme_embeddings(" ".join(expected_phonemes))
    transcribed_seq = get_phoneme_embeddings(" ".join(transcribed_phonemes))

    # Appliquer DTW pour aligner les phonèmes
    distance, path = fastdtw(expected_seq, transcribed_seq, dist=euclidean)

    # Identifier les mots avec erreurs de prononciation
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

    # Étape 6 : Générer un feedback compréhensible
    feedback = "🔊 Feedback sur votre prononciation :\n"
    if words_with_errors:
        feedback += "❌ Vous devez mieux prononcer ces mots : " + ", ".join(words_with_errors) + "\n"
    else:
        feedback += "✅ Votre prononciation est excellente ! 🎉\n"

    # errors is an array, but can contains multiple time the same word (for complex sounds). We want to keep only one occurence of each word
    errors = [dict(t) for t in {tuple(d.items()) for d in errors}]

    # convertir les vecteurs en json, pour afficher plus tard le tracé attendu et le tracé obtenu
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
    Aligne deux séquences de valeurs numériques en utilisant Dynamic Time Warping (DTW).
    Retourne les séquences interpolées pour avoir la même longueur.
    Ca permet de comparer les deux séquences plus facilement, car par exemple l'une peut être plus rapide que l'autre, 
    ou plus courte
    """
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    
    aligned_seq1 = []
    aligned_seq2 = []

    for i, j in path:
        aligned_seq1.append(seq1[i][0])  # Garder la première dimension
        aligned_seq2.append(seq2[j][0])

    # on amplifie artificiellement la différence, sinon souvent les deux courbes se superposent
    #aligned_seq2 = aligned_seq2 + (aligned_seq2 - aligned_seq1) * 2  # Amplifier la différence


    return np.array(aligned_seq1), np.array(aligned_seq2)

def compute_pronunciation_score(distance_dtw, phoneme_distance, word_distance, max_dtw=500, max_lev=30):
    """
    Calcule un score sur 100 en normalisant les distances.
    """
    # Normalisation des distances
    dtw_score = max(0, 100 - (distance_dtw / max_dtw) * 100)
    phoneme_score = max(0, 100 - (phoneme_distance / max_dtw) * 100)
    word_score = max(0, 100 - (word_distance / max_lev) * 100)
    
    # Poids des différentes composantes
    final_score = 0.4 * dtw_score + 0.3 * phoneme_score + 0.3 * word_score
    
    return round(final_score, 2)

def compare_audio_with_text(audio_1, text_reference, sampling_rate=16000):
    """
    Compare la prononciation d'un utilisateur avec une référence textuelle.
    """

    # Extraire les embeddings Wav2Vec2 de l'audio utilisateur
    emb_1 = extract_embeddings(audio_1, sampling_rate)

    # Generate a reference audio (via TTS) and extract its embeddings
    reference_file = audio.text2speech(text_reference)
    
    # Générer l'audio de référence (via TTS) et extraire ses embeddings
    # Supposons ici que vous avez déjà un fichier `reference.wav` généré à partir du texte.
    audio_2, sr = torchaudio.load(reference_file)
    emb_2 = extract_embeddings(audio_2, sr)
    
    # Appliquer DTW pour aligner les embeddings
    distance, path = fastdtw(emb_1, emb_2, dist=euclidean)
    distance = int(distance)  # Convertir en entier pour la réponse JSON
    
    # Convertir le texte de référence en phonèmes et récupérer le mapping mots-phonèmes
    expected_phonemes, phoneme_to_word = get_phonemes_with_word_mapping(text_reference)
    transcription = transcribe(audio_1)

    # Identifier les divergences
    differences = compare_transcriptions(transcription, text_reference)

    score = compute_pronunciation_score(distance, differences["phoneme_distance"], differences["word_distance"])

    # prosodie
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
    """ Extrait la fréquence fondamentale F0 de l'audio """
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_waveform, fmin=50, fmax=300)
    f0 = np.nan_to_num(f0)  # Remplace les NaN par 0
    return f0

def extract_energy(audio_waveform):
    """ Extrait et normalise l'énergie de l'audio """
    energy = librosa.feature.rms(y=audio_waveform)
    scaler = MinMaxScaler(feature_range=(0, 250))  # Mettre à l'échelle entre 0 et 250 pour matcher F0
    energy_scaled = scaler.fit_transform(energy.T).flatten()
    return energy_scaled

def interpolate_f0(f0):
    """ Interpole les valeurs manquantes de F0 pour éviter les coupures dans le graphe """
    f0 = np.array(f0)
    mask = f0 > 0  # On garde uniquement les valeurs valides
    f0_interp = np.interp(np.arange(len(f0)), np.where(mask)[0], f0[mask])
    return f0_interp


def transcribe(audio):
    """ Transcrit l'audio en texte avec Wav2Vec2 """
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = modelCTC(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

def clean_transcription(text):
    """ Corrige et nettoie la transcription brute """
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z' ]+", "", text)  # Supprime les caractères spéciaux
    text = text.replace("  ", " ")  # Évite les espaces multiples
    return text