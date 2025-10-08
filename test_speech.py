import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import warnings

warnings.filterwarnings("ignore")

import speech


class TestPhonemeFunctions(unittest.TestCase):
    """Tests pour les fonctions de phonémisation"""
    
    def setUp(self):
        self.sample_text = "hello world"
        self.expected_phonemes_hello = list("həloʊ")
        self.expected_phonemes_world = list("wɜrld")


    def test_get_phonemes_basic(self):
        phonemes = speech.get_phonemes(self.sample_text)
        
        self.assertIsInstance(phonemes, list)
        self.assertGreater(len(phonemes), 0)
        
        for phoneme in phonemes:
            self.assertIsInstance(phoneme, str)
            self.assertGreater(len(phoneme), 0)


    def test_get_phonemes_empty_text(self):
        phonemes = speech.get_phonemes("")
        self.assertEqual(phonemes, [])


    def test_get_phonemes_whitespace(self):
        phonemes = speech.get_phonemes("   ")
        self.assertEqual(phonemes, [])
    

    def test_get_phonemes_with_word_mapping(self):
        phonemes, phoneme_to_word = speech.get_phonemes_with_word_mapping(self.sample_text)
        
        self.assertIsInstance(phonemes, list)
        self.assertIsInstance(phoneme_to_word, dict)
        self.assertEqual(len(phonemes), len(phoneme_to_word))
        
        words = self.sample_text.split()
        for i, phoneme in enumerate(phonemes):
            self.assertIn(i, phoneme_to_word)
            self.assertIn(phoneme_to_word[i], words)
    
    def test_get_phonemes_special_characters(self):
        """Test avec des caractères spéciaux"""
        text_with_special = "hello, world!"
        phonemes = speech.get_phonemes(text_with_special)
        
        self.assertIsInstance(phonemes, list)
        self.assertGreater(len(phonemes), 0)


class TestEmbeddingFunctions(unittest.TestCase):
    """Tests for the get_phoneme_embeddings function"""
    
    def setUp(self):
        self.sample_phonemes = "həloʊ"
        self.expected_embeddings_shape = (5, 1)
    

    def test_get_phoneme_embeddings(self):
        embeddings = speech.get_phoneme_embeddings(self.sample_phonemes)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, self.expected_embeddings_shape)
        
        for i, phoneme in enumerate(self.sample_phonemes):
            expected_value = sum(ord(c) for c in phoneme)
            self.assertEqual(embeddings[i, 0], expected_value)
  
  
    def test_get_phoneme_embeddings_empty(self):
        embeddings = speech.get_phoneme_embeddings([])
        self.assertEqual(embeddings.shape, (0, 1))
    

    def test_compare_pronunciation(self):
        expected_phonemes = list("həloʊ")
        actual_phonemes = list("həloʊ")
        
        distance = speech.compare_pronunciation(expected_phonemes, actual_phonemes)
        
        self.assertEqual(distance, 0.0)
    

    def test_compare_pronunciation_different(self):
        expected_phonemes = list("həloʊ")
        actual_phonemes = list("həlo")
        
        distance = speech.compare_pronunciation(expected_phonemes, actual_phonemes)
        
        self.assertGreater(distance, 0)


class TestTranscriptionComparison(unittest.TestCase):
    """Tests for the compare_transcriptions function"""
    
    def setUp(self):
        self.reference_text = "hello world"
        self.perfect_transcription = "hello world"
        self.imperfect_transcription = "helo world"
    

    def test_compare_transcriptions_perfect_match(self):
        """Test with a perfect transcription"""
        result = speech.compare_transcriptions(
            self.perfect_transcription, 
            self.reference_text
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("word_distance", result)
        self.assertIn("phoneme_distance", result)
        self.assertIn("errors", result)
        self.assertIn("feedback", result)
        
        self.assertEqual(result["word_distance"], 0)
        self.assertEqual(result["phoneme_distance"], 0.0)
        self.assertEqual(len(result["errors"]), 0)


    def test_compare_transcriptions_imperfect_match(self):
        """Test avec une transcription imparfaite"""
        result = speech.compare_transcriptions(
            self.imperfect_transcription, 
            self.reference_text
        )
        
        self.assertIsInstance(result, dict)
        self.assertGreater(result["word_distance"], 0)
        self.assertGreaterEqual(result["phoneme_distance"], 0)
        self.assertIsInstance(result["errors"], list)
        self.assertIsInstance(result["feedback"], str)
    

    def test_compare_transcriptions_case_insensitive(self):
        """Test que la comparaison est insensible à la casse"""
        result1 = speech.compare_transcriptions("HELLO WORLD", self.reference_text)
        result2 = speech.compare_transcriptions("hello world", self.reference_text)
        
        self.assertEqual(result1["word_distance"], result2["word_distance"])
        self.assertEqual(result1["phoneme_distance"], result2["phoneme_distance"])


class TestAudioProcessingFunctions(unittest.TestCase):
    """Tests for the audio processing functions"""
    
    def setUp(self):
        # Create a test audio signal (1 second at 440Hz)
        self.sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        self.test_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)


    def test_extract_energy(self):
        energy = speech.extract_energy(self.test_audio)
        
        self.assertIsInstance(energy, np.ndarray)
        self.assertGreater(len(energy), 0)
        
        self.assertTrue(np.all(energy >= 0))
    

    def test_interpolate_f0(self):
        """Test of the F0 interpolation"""
        # Create an F0 with missing values
        f0_with_gaps = np.array([100, 0, 0, 120, 0, 130])
        f0_interp = speech.interpolate_f0(f0_with_gaps)

        self.assertIsInstance(f0_interp, np.ndarray)
        self.assertEqual(len(f0_interp), len(f0_with_gaps))
        self.assertFalse(np.any(np.isnan(f0_interp)))
    
    @patch('speech.processor')
    @patch('speech.modelCTC')
    def test_transcribe(self, mock_model, mock_processor):
        # Mocks
        mock_processor.return_value = MagicMock()
        mock_processor.return_value.input_values = torch.tensor([[1, 2, 3]])
        mock_model.return_value.logits = torch.tensor([[[0.1, 0.9, 0.2]]])
        mock_processor.batch_decode.return_value = ["hello world"]
        
        result = speech.transcribe(self.test_audio)

        self.assertIsInstance(result, str)
        self.assertEqual(result, "hello world")
        mock_processor.assert_called_once()
        mock_model.assert_called_once()


    def test_clean_transcription(self):
        dirty_text = "  Hello, World! 123  "
        clean_text = speech.clean_transcription(dirty_text)
        
        self.assertEqual(clean_text, "hello world")
        self.assertNotIn(",", clean_text)
        self.assertNotIn("!", clean_text)
        self.assertNotIn("123", clean_text)


class TestScoringFunctions(unittest.TestCase):
    """Tests for the scoring functions"""
    
    def test_compute_pronunciation_score_perfect(self):
        score = speech.compute_pronunciation_score(0, 0, 0)
        
        self.assertEqual(score, 100.0)
    

    def test_compute_pronunciation_score_imperfect(self):
        score = speech.compute_pronunciation_score(100, 50, 5)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    

    def test_compute_pronunciation_score_edge_cases(self):
        bad_score = speech.compute_pronunciation_score(1000, 1000, 100)
        self.assertGreaterEqual(bad_score, 0)
        
        negative_score = speech.compute_pronunciation_score(-10, -5, -2)
        self.assertEqual(negative_score, 100.0)


class TestAlignmentFunctions(unittest.TestCase):
    """Tests for the alignment functions"""
    
    def test_align_sequences_dtw_identical(self):
        seq1 = [[1], [2], [3], [4]]
        seq2 = [[1], [2], [3], [4]]
        
        aligned_seq1, aligned_seq2 = speech.align_sequences_dtw(seq1, seq2)
        
        self.assertIsInstance(aligned_seq1, np.ndarray)
        self.assertIsInstance(aligned_seq2, np.ndarray)
        self.assertEqual(len(aligned_seq1), len(aligned_seq2))
        np.testing.assert_array_equal(aligned_seq1, aligned_seq2)
    

    def test_align_sequences_dtw_different_lengths(self):
        seq1 = [[1], [2], [3]]
        seq2 = [[1], [2], [3], [4], [5]]
        
        aligned_seq1, aligned_seq2 = speech.align_sequences_dtw(seq1, seq2)
        
        self.assertIsInstance(aligned_seq1, np.ndarray)
        self.assertIsInstance(aligned_seq2, np.ndarray)
        self.assertEqual(len(aligned_seq1), len(aligned_seq2))
        self.assertGreater(len(aligned_seq1), 0)


class TestIntegration(unittest.TestCase):
    """Tests for the integration of the functions"""
    
    def setUp(self):
        self.sample_text = "hello"
        self.sample_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
    
    @patch('speech.extract_embeddings')
    @patch('speech.audio.text2speech')
    @patch('torchaudio.load')
    @patch('speech.transcribe')
    @patch('speech.compare_transcriptions')
    @patch('speech.compute_pronunciation_score')
    @patch('speech.extract_energy')
    @patch('speech.extract_f0')
    @patch('speech.interpolate_f0')
    def test_compare_audio_with_text_mocked(self, mock_interp_f0, mock_extract_f0, 
                                          mock_extract_energy, mock_compute_score,
                                          mock_compare_trans, mock_transcribe,
                                          mock_torchaudio_load, mock_text2speech,
                                          mock_extract_emb):
                
        # Mocks
        mock_extract_emb.return_value = np.random.randn(100, 768)
        mock_text2speech.return_value = "temp_reference.wav"
        mock_torchaudio_load.return_value = (torch.tensor([[1, 2, 3]]), 16000)
        mock_transcribe.return_value = "hello"
        mock_compare_trans.return_value = {
            "phoneme_distance": 10,
            "word_distance": 2,
            "feedback": "Good pronunciation!",
            "transcribe": "hello"
        }
        mock_compute_score.return_value = 85.5
        mock_extract_energy.return_value = np.array([1, 2, 3])
        mock_extract_f0.return_value = np.array([100, 110, 120])
        mock_interp_f0.return_value = np.array([100, 110, 120])
        
        result = speech.compare_audio_with_text(self.sample_audio, self.sample_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("distance", result)
        self.assertIn("differences", result)
        self.assertIn("feedback", result)
        self.assertIn("transcribe", result)
        self.assertIn("prosody", result)
        
        self.assertIsInstance(result["score"], (int, float))
        self.assertIsInstance(result["distance"], (int, float))
        self.assertIsInstance(result["differences"], dict)
        self.assertIsInstance(result["prosody"], dict)
        
        mock_extract_emb.assert_called()
        mock_text2speech.assert_called_with(self.sample_text)
        mock_transcribe.assert_called_with(self.sample_audio)


class TestEdgeCases(unittest.TestCase):
    """Tests for the edge cases and robustness"""
    
    def test_get_phonemes_unicode(self):
        unicode_text = "café naïve"
        phonemes = speech.get_phonemes(unicode_text)
        
        self.assertIsInstance(phonemes, list)
    

    def test_get_phoneme_embeddings_unicode(self):
        unicode_phonemes = ["c", "a", "f", "é"]
        embeddings = speech.get_phoneme_embeddings(unicode_phonemes)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (4, 1))
    

    def test_compare_pronunciation_empty_sequences(self):
        distance = speech.compare_pronunciation([], [])
        self.assertEqual(distance, 0.0)
    

    def test_align_sequences_dtw_empty(self):
        aligned_seq1, aligned_seq2 = speech.align_sequences_dtw([], [])
        
        self.assertIsInstance(aligned_seq1, np.ndarray)
        self.assertIsInstance(aligned_seq2, np.ndarray)
        self.assertEqual(len(aligned_seq1), 0)
        self.assertEqual(len(aligned_seq2), 0)
