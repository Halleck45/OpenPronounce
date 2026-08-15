import json
import unittest

import numpy as np

from openpronounce import phones


class TestNormalization(unittest.TestCase):

    def test_length_marks_and_reduced_vowels(self):
        self.assertEqual(phones.normalize_phones(["h", "iː", "t", "ᵻ", "ɐ"]), ["h", "i", "t", "ɪ", "ə"])

    def test_repetitions_collapse(self):
        self.assertEqual(phones.normalize_phones(["d", "d", "ɚ", "ɹ", "z"]), ["d", "ɚ", "z"])


class TestExpectedPhones(unittest.TestCase):

    def test_words_and_groups_are_aligned(self):
        words, groups = phones.get_expected_phones("Hello, how are you?")
        self.assertEqual(words, ["hello", "how", "are", "you"])
        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0][0], "h")
        self.assertTrue(all(len(g) > 0 for g in groups))

    def test_case_insensitive(self):
        self.assertEqual(phones.get_expected_phones("IT TAKES HEAT"), phones.get_expected_phones("it takes heat"))

    def test_empty(self):
        self.assertEqual(phones.get_expected_phones(""), ([], []))


class TestComparePhones(unittest.TestCase):

    def heard(self, text):
        _, groups = phones.get_expected_phones(text)
        return [p for g in groups for p in g]

    def test_perfect(self):
        result = phones.compare_phones(self.heard("hello how are you"), "Hello, how are you?")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["phone_error_rate"], 0.0)
        self.assertEqual(result["words_with_errors"], [])

    def test_substituted_vowel_is_reported_on_the_right_word(self):
        # "hell no who are you" for "hello how are you"
        result = phones.compare_phones(["h", "ɛ", "l", "n", "oʊ", "h", "u", "ɑɹ", "j", "u"], "Hello, how are you?")
        self.assertEqual(result["words_with_errors"], ["hello", "how"])
        hello = result["errors"][0]
        self.assertEqual(hello["expected"], "həloʊ")
        self.assertEqual(hello["actual"], "hɛlnoʊ")
        self.assertEqual(hello["position"], 0)

    def test_missing_word(self):
        result = phones.compare_phones(self.heard("hello are you"), "hello how are you")
        self.assertEqual(result["words_with_errors"], ["how"])
        self.assertEqual(result["errors"][0]["actual"], "")

    def test_alternate_pronunciations_are_accepted(self):
        heard = self.heard("hello") + ["eɪ"] + self.heard("developer")
        result = phones.compare_phones(heard, "hello a developer")
        self.assertEqual(result["errors"], [])

    def test_merged_boundary_phone_is_accepted(self):
        # "heat to" said as "hea-to": the second t is dropped
        heard = self.heard("heat") + self.heard("to")[1:]
        result = phones.compare_phones(heard, "heat to")
        self.assertEqual(result["errors"], [])

    def test_single_wrong_phone_in_long_word_is_not_reported(self):
        heard = self.heard("developer")
        heard[1] = "i"
        result = phones.compare_phones(heard, "developer")
        self.assertEqual(result["errors"], [])

    def test_serialisable(self):
        json.dumps(phones.compare_phones(self.heard("hello"), "hello world"))


class TestPhonemizerFallback(unittest.TestCase):

    def test_words_merged_by_espeak_still_get_phones(self):
        # espeak phonemizes "would have to" as a single token; the per-word fallback must kick in.
        words, groups = phones.get_expected_phones("would have to")
        self.assertEqual(words, ["would", "have", "to"])
        self.assertTrue(all(len(g) > 0 for g in groups), groups)

    def test_phone_error_rate_without_expected_phones(self):
        self.assertEqual(phones.compare_phones(["a"], "")["phone_error_rate"], 1.0)
        self.assertEqual(phones.compare_phones([], "")["phone_error_rate"], 0.0)


class TestDecodeWithConfidence(unittest.TestCase):

    VOCAB = ("<pad>", "<s>", "h", "ə", "l", "oʊ", "ɜː", "ɹ", "iː", "d", "i5")

    def log_posteriors(self, frames):
        """``frames`` is a list of ``{token: probability}``; the rest of the mass goes to the blank."""
        out = np.full((len(frames), len(self.VOCAB)), 1e-6)
        for t, probs in enumerate(frames):
            for token, p in probs.items():
                out[t, self.VOCAB.index(token)] = p
            out[t, 0] = max(1e-6, 1.0 - sum(probs.values()))
        return np.log(out / out.sum(axis=1, keepdims=True))

    def test_blanks_dropped_and_repeats_collapsed(self):
        lp = self.log_posteriors([{"h": 0.9}, {"h": 0.6}, {}, {"ə": 0.7}, {}, {}, {"l": 0.8}, {"l": 0.95}, {"oʊ": 0.6}])
        rec = phones.decode_ctc(lp, self.VOCAB)
        self.assertEqual(rec.phones, ["h", "ə", "l", "oʊ"])
        self.assertEqual(rec.spans, [(0, 2), (3, 4), (6, 8), (8, 9)])
        self.assertAlmostEqual(rec.confidences[0], 0.9, places=3)  # peak over the two h frames
        self.assertAlmostEqual(rec.confidences[2], 0.95, places=3)
        self.assertAlmostEqual(rec.confidences[3], 0.6, places=3)
        self.assertEqual(len(rec.phones), len(rec.confidences))

    def test_repeated_phone_separated_by_blank_is_kept(self):
        lp = self.log_posteriors([{"d": 0.9}, {}, {"d": 0.8}])
        self.assertEqual(phones.decode_ctc(lp, self.VOCAB, normalize=False).phones, ["d", "d"])

    def test_special_tokens_dropped(self):
        lp = self.log_posteriors([{"<s>": 0.9}, {"h": 0.9}])
        self.assertEqual(phones.decode_ctc(lp, self.VOCAB).phones, ["h"])

    def test_normalization_keeps_confidences_aligned(self):
        # "ɜː ɹ" merges into ɚ (keeps the highest confidence, spans both tokens), "iː" loses its length
        # mark, "d d" collapses, the Mandarin tone number of "i5" is dropped.
        lp = self.log_posteriors([{"ɜː": 0.6}, {"ɹ": 0.9}, {}, {"iː": 0.7}, {"d": 0.8}, {}, {"d": 0.7}, {"i5": 0.6}])
        rec = phones.decode_ctc(lp, self.VOCAB)
        self.assertEqual(rec.phones, ["ɚ", "i", "d", "i"])
        self.assertEqual([round(c, 3) for c in rec.confidences], [0.9, 0.7, 0.8, 0.6])
        self.assertEqual(rec.spans, [(0, 2), (3, 4), (4, 7), (7, 8)])
        raw = phones.decode_ctc(lp, self.VOCAB, normalize=False)
        self.assertEqual(raw.phones, ["ɜː", "ɹ", "iː", "d", "d", "i5"])

    def test_normalize_phones_aliases(self):
        self.assertEqual(phones.normalize_phones(["th", "ai5", "ɔːɹ", "ɜː"]), ["t", "aɪ", "oɹ", "ɚ"])


class TestConfidenceRule(unittest.TestCase):
    """The same wrong phones flag a word or not depending on how sure the recognizer was."""

    VOCAB = ("<pad>", "h", "aʊ", "u", "j", "s", "z", "t")

    def recognition(self, phone_list, expected_posterior=None):
        """A recognition of ``phone_list`` (one frame each, posterior 0.9) where the phones given in
        ``expected_posterior`` (``{frame: {phone: p}}``) also get some posterior mass."""
        lp = np.full((len(phone_list), len(self.VOCAB)), 1e-6)
        for t, phone in enumerate(phone_list):
            lp[t, self.VOCAB.index(phone)] = 0.9
            for other, p in (expected_posterior or {}).get(t, {}).items():
                lp[t, self.VOCAB.index(other)] = p
            lp[t, 0] = max(1e-6, 1 - lp[t, 1:].sum())
        return phones.decode_ctc(np.log(lp / lp.sum(axis=1, keepdims=True)), self.VOCAB)

    def test_confident_substitution_is_flagged(self):
        result = phones.compare_phones(self.recognition(["h", "u"]), "how")
        self.assertEqual(result["words_with_errors"], ["how"])
        error = result["errors"][0]
        self.assertGreaterEqual(error["confidence"], 0.4)
        self.assertEqual([p["expected"] for p in error["phones"]], ["h", "aʊ"])
        self.assertEqual(error["phones"][1]["heard"], "u")
        self.assertEqual(error["phones"][0]["confidence"], 0.0)
        self.assertGreater(error["phones"][1]["confidence"], 0.9)
        self.assertEqual(result["heard_phones_confidence"], [0.9, 0.9])

    def test_plausible_expected_phone_is_not_flagged(self):
        # Same heard phones, but the recognizer also gave the expected aʊ a fair posterior on that frame.
        result = phones.compare_phones(self.recognition(["h", "u"], {1: {"aʊ": 0.08}}), "how")
        self.assertEqual(result["words_with_errors"], [])

    def test_near_phones_count_half(self):
        # s for z (voicing) in "zoo": half an error over 2 phones, below the threshold
        self.assertEqual(phones.compare_phones(self.recognition(["s", "u"]), "zoo")["words_with_errors"], [])
        # t for z is a full substitution
        self.assertEqual(phones.compare_phones(self.recognition(["t", "u"]), "zoo")["words_with_errors"], ["zoo"])

    def test_plain_list_keeps_full_confidence(self):
        result = phones.compare_phones(["h", "u"], "how")
        self.assertEqual(result["words_with_errors"], ["how"])
        self.assertEqual(result["heard_phones_confidence"], [1.0, 1.0])
        self.assertEqual(result["errors"][0]["confidence"], 0.5)

    def test_serialisable(self):
        result = phones.compare_phones(self.recognition(["h", "u"]), "how")
        json.dumps(result)
        self.assertIsInstance(result["errors"][0]["phones"][1]["confidence"], float)
