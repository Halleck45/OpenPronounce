import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from openpronounce import speech


class TestPhonemeFunctions(unittest.TestCase):

    def test_get_phonemes_basic(self):
        phonemes = speech.get_phonemes("hello world")
        self.assertIsInstance(phonemes, list)
        self.assertGreater(len(phonemes), 0)
        for phoneme in phonemes:
            self.assertIsInstance(phoneme, str)
            self.assertGreater(len(phoneme), 0)

    def test_get_phonemes_empty_text(self):
        self.assertEqual(speech.get_phonemes(""), [])

    def test_get_phonemes_whitespace(self):
        self.assertEqual(speech.get_phonemes("   "), [])

    def test_get_phonemes_with_word_mapping(self):
        text = "hello world"
        phonemes, phoneme_to_word = speech.get_phonemes_with_word_mapping(text)
        self.assertEqual(len(phonemes), len(phoneme_to_word))
        for i in range(len(phonemes)):
            self.assertIn(phoneme_to_word[i], text.split())

    def test_get_phonemes_special_characters(self):
        self.assertGreater(len(speech.get_phonemes("hello, world!")), 0)

    def test_get_phonemes_is_case_insensitive(self):
        # espeak spells upper-case words letter by letter; we must not.
        self.assertEqual(speech.get_phonemes("IT TAKES HEAT"), speech.get_phonemes("it takes heat"))

    def test_get_phonemes_unicode(self):
        self.assertIsInstance(speech.get_phonemes("café naïve"), list)


class TestEmbeddingFunctions(unittest.TestCase):

    def test_get_phoneme_embeddings(self):
        embeddings = speech.get_phoneme_embeddings("həloʊ")
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (5, 1))
        for i, phoneme in enumerate("həloʊ"):
            self.assertEqual(embeddings[i, 0], ord(phoneme))

    def test_get_phoneme_embeddings_empty(self):
        self.assertEqual(speech.get_phoneme_embeddings([]).shape, (0, 1))

    def test_get_phoneme_embeddings_unicode(self):
        self.assertEqual(speech.get_phoneme_embeddings(["c", "a", "f", "é"]).shape, (4, 1))

    def test_compare_pronunciation(self):
        self.assertEqual(speech.compare_pronunciation(list("həloʊ"), list("həloʊ")), 0.0)

    def test_compare_pronunciation_different(self):
        self.assertGreater(speech.compare_pronunciation(list("həloʊ"), list("həlo")), 0)

    def test_compare_pronunciation_empty_sequences(self):
        self.assertEqual(speech.compare_pronunciation([], []), 0.0)


class TestTranscriptionComparison(unittest.TestCase):

    def test_perfect_match(self):
        result = speech.compare_transcriptions("hello world", "hello world")
        self.assertEqual(result["word_distance"], 0)
        self.assertEqual(result["phoneme_distance"], 0.0)
        self.assertEqual(result["word_error_rate"], 0.0)
        self.assertEqual(result["phoneme_error_rate"], 0.0)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["words_with_errors"], [])
        self.assertIn("excellent", result["feedback"])

    def test_imperfect_match(self):
        result = speech.compare_transcriptions("helo world", "hello world")
        self.assertGreater(result["word_distance"], 0)
        self.assertGreaterEqual(result["phoneme_distance"], 0)
        self.assertIsInstance(result["errors"], list)
        self.assertIsInstance(result["feedback"], str)

    def test_case_insensitive(self):
        result1 = speech.compare_transcriptions("HELLO WORLD", "hello world")
        result2 = speech.compare_transcriptions("hello world", "hello world")
        self.assertEqual(result1["word_distance"], result2["word_distance"])
        self.assertEqual(result1["phoneme_distance"], result2["phoneme_distance"])
        self.assertEqual(result1["errors"], [])

    def test_upper_case_reference_is_not_penalised(self):
        result = speech.compare_transcriptions("IT TAKES HEAT", "IT TAKES HEAT")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["phoneme_error_rate"], 0.0)

    def test_missing_word_is_reported(self):
        result = speech.compare_transcriptions("hello", "hello world")
        self.assertEqual(result["words_with_errors"], ["world"])
        error = result["errors"][0]
        self.assertEqual(error["word"], "world")
        self.assertEqual(error["actual"], "")
        self.assertGreater(len(error["expected"]), 0)

    def test_mispronounced_word_carries_phonemes(self):
        result = speech.compare_transcriptions("hello wild", "hello world")
        self.assertEqual(result["words_with_errors"], ["world"])
        error = result["errors"][0]
        self.assertEqual(error["actual_word"], "wild")
        self.assertNotEqual(error["expected"], error["actual"])

    def test_inserted_words_are_attached_to_the_previous_word(self):
        result = speech.compare_transcriptions("hell no who are you", "hello how are you")
        by_word = {e["word"]: e for e in result["errors"]}
        self.assertEqual(by_word["hello"]["actual_word"], "hell no")
        self.assertEqual(by_word["how"]["actual_word"], "who")

    def test_error_rates_are_normalised(self):
        result = speech.compare_transcriptions("", "hello world")
        self.assertEqual(result["word_error_rate"], 1.0)
        self.assertEqual(result["phoneme_error_rate"], 1.0)

    def test_result_is_json_serialisable(self):
        import json
        json.dumps(speech.compare_transcriptions("helo world", "hello world"))


class TestAudioProcessingFunctions(unittest.TestCase):

    def setUp(self):
        sr, duration, frequency = 16000, 1.0, 440.0
        t = np.linspace(0, duration, int(sr * duration), False)
        self.test_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    def test_extract_energy(self):
        energy = speech.extract_energy(self.test_audio)
        self.assertIsInstance(energy, np.ndarray)
        self.assertGreater(len(energy), 0)
        self.assertTrue(np.all(energy >= 0))

    def test_interpolate_f0(self):
        f0_interp = speech.interpolate_f0(np.array([100, 0, 0, 120, 0, 130]))
        self.assertEqual(len(f0_interp), 6)
        self.assertFalse(np.any(np.isnan(f0_interp)))
        self.assertTrue(np.all(f0_interp > 0))

    def test_interpolate_f0_all_unvoiced(self):
        np.testing.assert_array_equal(speech.interpolate_f0(np.zeros(4)), np.zeros(4))

    @patch("openpronounce.speech._get_model_ctc")
    @patch("openpronounce.speech._get_processor")
    def test_transcribe(self, mock_get_processor, mock_get_model):
        processor = MagicMock()
        processor.return_value.input_values = torch.tensor([[1.0, 2.0, 3.0]])
        processor.batch_decode.return_value = ["hello world"]
        mock_get_processor.return_value = processor
        model = MagicMock()
        model.return_value.logits = torch.tensor([[[0.1, 0.9, 0.2]]])
        mock_get_model.return_value = model

        self.assertEqual(speech.transcribe(self.test_audio), "hello world")
        processor.assert_called_once()
        model.assert_called_once()

    def test_clean_transcription(self):
        self.assertEqual(speech.clean_transcription("  Hello,   World! 123  "), "hello world")


class TestScoringFunctions(unittest.TestCase):

    def test_perfect(self):
        self.assertEqual(speech.compute_pronunciation_score(0, 0, 0), 100.0)

    def test_range(self):
        for args in [(7, 0.2, 0.2), (100, 50, 5), (12, 0.5, 0.5)]:
            score = speech.compute_pronunciation_score(*args)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    def test_edge_cases(self):
        self.assertEqual(speech.compute_pronunciation_score(1000, 10, 10), 0.0)
        self.assertEqual(speech.compute_pronunciation_score(-10, -5, -2), 100.0)

    def test_monotonic(self):
        good = speech.compute_pronunciation_score(6, 0.0, 0.0)
        medium = speech.compute_pronunciation_score(8, 0.2, 0.2)
        bad = speech.compute_pronunciation_score(12, 1.0, 1.0)
        self.assertGreater(good, medium)
        self.assertGreater(medium, bad)


class TestAlignmentFunctions(unittest.TestCase):

    def test_identical(self):
        a, b = speech.align_sequences_dtw([[1], [2], [3], [4]], [[1], [2], [3], [4]])
        np.testing.assert_array_equal(a, b)

    def test_different_lengths(self):
        a, b = speech.align_sequences_dtw([[1], [2], [3]], [[1], [2], [3], [4], [5]])
        self.assertEqual(len(a), len(b))
        self.assertGreater(len(a), 0)

    def test_empty(self):
        a, b = speech.align_sequences_dtw([], [])
        self.assertEqual(len(a), 0)
        self.assertEqual(len(b), 0)


class TestIntegration(unittest.TestCase):

    @patch("openpronounce.speech.phones.recognize_phones")
    @patch("openpronounce.speech.interpolate_f0")
    @patch("openpronounce.speech.extract_f0")
    @patch("openpronounce.speech.extract_energy")
    @patch("openpronounce.speech.transcribe")
    @patch("openpronounce.speech.audio.load")
    @patch("openpronounce.speech.audio.text2speech")
    @patch("openpronounce.speech.extract_embeddings")
    def test_compare_audio_with_text_mocked(self, mock_extract_emb, mock_text2speech, mock_load,
                                            mock_transcribe, mock_extract_energy, mock_extract_f0,
                                            mock_interp_f0, mock_recognize_phones):
        mock_recognize_phones.return_value = ["h", "ə", "l", "oʊ"]
        sample_audio = np.random.randn(16000).astype(np.float32)
        mock_extract_emb.return_value = np.random.randn(20, 8)
        mock_text2speech.return_value = "temp_reference.wav"
        mock_load.return_value = np.zeros(16000, dtype=np.float32)
        mock_transcribe.return_value = "hello"
        mock_extract_energy.return_value = np.array([1, 2, 3])
        mock_extract_f0.return_value = np.array([100, 110, 120])
        mock_interp_f0.return_value = np.array([100, 110, 120])

        result = speech.compare_audio_with_text(sample_audio, "hello")

        for key in ("score", "distance", "acoustic_distance", "differences", "feedback", "transcribe", "prosody"):
            self.assertIn(key, result)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)
        self.assertEqual(result["differences"]["errors"], [])
        self.assertEqual(result["differences"]["heard_phones"], ["h", "ə", "l", "oʊ"])
        self.assertEqual(result["prosody"]["f0"], [100, 110, 120])
        mock_text2speech.assert_called_with("hello", lang="en")
        mock_transcribe.assert_called_with(sample_audio, "en")

    @patch("openpronounce.speech.interpolate_f0", return_value=np.array([100.0]))
    @patch("openpronounce.speech.extract_f0", return_value=np.array([100.0]))
    @patch("openpronounce.speech.extract_energy", return_value=np.array([1.0]))
    @patch("openpronounce.speech.transcribe", return_value="HELLO WORLD")
    @patch("openpronounce.speech.audio.load", return_value=np.zeros(16000, dtype=np.float32))
    @patch("openpronounce.speech.audio.text2speech", return_value="ref.wav")
    @patch("openpronounce.speech.extract_embeddings", return_value=np.zeros((10, 4)))
    def test_text_fallback_when_phone_model_disabled(self, *_):
        result = speech.compare_audio_with_text(np.zeros(16000, dtype=np.float32), "hello world", use_phone_model=False)
        self.assertEqual(result["differences"]["errors"], [])
        self.assertNotIn("heard_phones", result["differences"])


if __name__ == "__main__":
    unittest.main()
