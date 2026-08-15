import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

import server
from openpronounce import LANGUAGES, get_language, phones, speech
from openpronounce.languages import DEFAULT_LANGUAGE


class TestRegistry(unittest.TestCase):

    def test_registry(self):
        self.assertEqual(DEFAULT_LANGUAGE, "en")
        self.assertEqual(LANGUAGES["en"].asr_model, "facebook/wav2vec2-large-960h")
        self.assertEqual(LANGUAGES["en"].espeak, "en-us")
        for code in ("en", "fr", "es", "de", "it", "pt", "nl"):
            language = LANGUAGES[code]
            self.assertEqual(language.code, code)
            self.assertTrue(language.espeak and language.asr_model and language.name)

    def test_get_language(self):
        self.assertIs(get_language("fr"), LANGUAGES["fr"])
        with self.assertRaises(ValueError) as ctx:
            get_language("xx")
        self.assertIn("'xx'", str(ctx.exception))
        self.assertIn("en", str(ctx.exception))
        self.assertIn("fr", str(ctx.exception))


class TestFrenchPhones(unittest.TestCase):

    def test_expected_phones_are_french(self):
        words, groups = phones.get_expected_phones("bonjour le monde", lang="fr")
        self.assertEqual(words, ["bonjour", "le", "monde"])
        flat = [p for g in groups for p in g]
        self.assertTrue("ʒ" in flat or "ɔ̃" in flat, flat)
        self.assertNotEqual(groups, phones.get_expected_phones("bonjour le monde")[1])

    def test_normalization_is_language_aware(self):
        # French merges mid vowels (ɔ/o), English merges cot/caught (ɔ/ɑ)
        self.assertEqual(phones.normalize_phones(["ɔ"], lang="fr"), ["o"])
        self.assertEqual(phones.normalize_phones(["ɔ", "ɛ"], lang="de"), ["ɔ", "ɛ"])
        self.assertEqual(phones.normalize_phones(["ɔ"], lang="en"), ["ɑ"])
        self.assertEqual(phones.normalize_phones(["ɔ"]), ["ɑ"])
        self.assertEqual(phones.normalize_phones(["ɾ"], lang="es"), ["ɾ"])
        # Length marks and repetitions are dropped whatever the language.
        self.assertEqual(phones.normalize_phones(["aː", "a", "b", "b"], lang="fr"), ["a", "b"])

    def test_compare_phones_french_perfect(self):
        _, groups = phones.get_expected_phones("bonjour, je suis développeur", lang="fr")
        heard = [p for g in groups for p in g]
        result = phones.compare_phones(heard, "Bonjour, je suis développeur", lang="fr")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["phone_error_rate"], 0.0)

    def test_french_schwa_elision_is_accepted(self):
        _, groups = phones.get_expected_phones("je suis", lang="fr")
        heard = [g for g in groups[0] if g != "ə"] + list(groups[1])
        self.assertEqual(phones.compare_phones(heard, "je suis", lang="fr")["errors"], [])

    def test_speech_phonemes_french(self):
        self.assertNotEqual(speech.get_phonemes("bonjour", lang="fr"), speech.get_phonemes("bonjour"))


class TestFrenchPipeline(unittest.TestCase):

    @patch("openpronounce.speech.phones.recognize_phones")
    @patch("openpronounce.speech.interpolate_f0", return_value=np.array([100.0]))
    @patch("openpronounce.speech.extract_f0", return_value=np.array([100.0]))
    @patch("openpronounce.speech.extract_energy", return_value=np.array([1.0]))
    @patch("openpronounce.speech.transcribe", return_value="bonjour le monde")
    @patch("openpronounce.speech.audio.load", return_value=np.zeros(16000, dtype=np.float32))
    @patch("openpronounce.speech.audio.text2speech", return_value="ref.wav")
    @patch("openpronounce.speech.extract_embeddings", return_value=np.zeros((10, 4)))
    def test_compare_audio_with_text_french(self, _emb, mock_text2speech, _load, mock_transcribe,
                                            _energy, _f0, _interp, mock_recognize_phones):
        _, groups = phones.get_expected_phones("bonjour le monde", lang="fr")
        mock_recognize_phones.return_value = [p for g in groups for p in g]
        sound = np.zeros(16000, dtype=np.float32)

        result = speech.compare_audio_with_text(sound, "Bonjour le monde", lang="fr")

        self.assertEqual(result["language"], "fr")
        self.assertEqual(result["differences"]["errors"], [])
        self.assertEqual(result["differences"]["phoneme_error_rate"], 0.0)
        mock_text2speech.assert_called_with("Bonjour le monde", lang="fr")
        mock_transcribe.assert_called_with(sound, "fr")
        mock_recognize_phones.assert_called_with(sound, 16000, lang="fr")

    def test_unknown_language(self):
        with self.assertRaises(ValueError):
            speech.compare_audio_with_text(np.zeros(16000, dtype=np.float32), "hello", lang="xx")

    @patch("openpronounce.speech._load_models")
    def test_transcribe_uses_the_language_model(self, mock_load_models):
        import torch
        processor = MagicMock()
        processor.return_value.input_values = torch.tensor([[1.0, 2.0, 3.0]])
        processor.batch_decode.return_value = ["bonjour"]
        model = MagicMock()
        model.return_value.logits = torch.tensor([[[0.1, 0.9, 0.2]]])
        mock_load_models.return_value = (processor, model)

        self.assertEqual(speech.transcribe(np.zeros(16000, dtype=np.float32), lang="fr"), "bonjour")
        mock_load_models.assert_called_with(LANGUAGES["fr"].asr_model)


class TestServerLanguages(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(server.app)

    def test_languages_endpoint(self):
        body = self.client.get("/languages").json()
        self.assertEqual(body["default"], "en")
        self.assertIn({"code": "fr", "name": "French"}, body["languages"])
        self.assertEqual(len(body["languages"]), len(LANGUAGES))

    def test_phonemes_french(self):
        response = self.client.post("/phonemes", data={"text": "bonjour", "lang": "fr"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["phonemes"], speech.get_phonemes("bonjour", lang="fr"))

    def test_unknown_language_is_rejected(self):
        response = self.client.post("/phonemes", data={"text": "bonjour", "lang": "xx"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("fr", response.json()["detail"])


class TestAcousticCalibration(unittest.TestCase):

    def test_every_language_has_a_baseline_at_least_english(self):
        from openpronounce.languages import LANGUAGES
        for language in LANGUAGES.values():
            self.assertGreaterEqual(language.acoustic_good, LANGUAGES["en"].acoustic_good)

    def test_score_uses_the_language_baseline(self):
        from openpronounce import speech
        # Two native French voices sit at ~9: full acoustic marks in French, not in English
        self.assertEqual(speech.compute_pronunciation_score(9.0, 0, 0, lang="fr"), 100.0)
        self.assertLess(speech.compute_pronunciation_score(9.0, 0, 0, lang="en"), 100.0)
