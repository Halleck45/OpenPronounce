import json
import unittest

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
