"""
Unit tests for AI Email Spam Detector.
Tests text preprocessing, model training, prediction, and edge cases.
"""

import unittest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import clean_text, tokenize_and_remove_stopwords, preprocess_text
from src.model import SpamClassifier
from src.utils import compute_text_stats, generate_explanation


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_lowercasing_and_punctuation(self):
        raw = "HELLO WORLD!!! WIN $1,000 NOW..."
        cleaned = clean_text(raw)
        self.assertNotIn("!", cleaned)
        self.assertNotIn(".", cleaned)
        self.assertIn("hello", cleaned)
        self.assertIn("moneymarker", cleaned)

    def test_clean_text_url_and_email_marker(self):
        raw = "Click here: https://example-scam.com/win or email support@bank.com"
        cleaned = clean_text(raw)
        self.assertIn("urlmarker", cleaned)
        self.assertIn("emailmarker", cleaned)
        self.assertNotIn("https://", cleaned)

    def test_empty_preprocessing(self):
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text("   "), "")
        self.assertEqual(preprocess_text(None), "")

    def test_stopwords_removal(self):
        raw = "this is a wonderful and awesome project for college"
        processed = preprocess_text(raw, apply_stemming=False)
        self.assertNotIn("this", processed.split())
        self.assertNotIn("is", processed.split())
        self.assertIn("wonder", processed)


class TestSpamModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = SpamClassifier(algorithm="naive_bayes")
        cls.metrics = cls.classifier.train("data/spam_dataset.csv")

    def test_model_training_metrics(self):
        self.assertTrue(self.classifier.is_trained)
        self.assertIn("accuracy", self.metrics)
        self.assertIn("precision", self.metrics)
        self.assertIn("recall", self.metrics)
        self.assertIn("f1_score", self.metrics)
        self.assertGreaterEqual(self.metrics["accuracy"], 0.85)

    def test_spam_prediction(self):
        spam_sample = "CONGRATULATIONS! You won $500,000 cash in the international lottery. Click here to claim your prize now: http://lottery.top"
        result = self.classifier.predict(spam_sample)
        self.assertEqual(result["prediction"], "SPAM")
        self.assertGreater(result["confidence"], 50.0)
        self.assertGreater(result["spam_prob"], result["ham_prob"])

    def test_ham_prediction(self):
        ham_sample = "Hi Sarah, can you please review the project presentation slides and send your feedback before tomorrow morning standup meeting?"
        result = self.classifier.predict(ham_sample)
        self.assertEqual(result["prediction"], "NOT SPAM")
        self.assertGreater(result["confidence"], 50.0)
        self.assertGreater(result["ham_prob"], result["spam_prob"])

    def test_empty_prediction(self):
        result = self.classifier.predict("")
        self.assertEqual(result["prediction"], "UNKNOWN")
        self.assertEqual(result["confidence"], 0.0)


class TestUtils(unittest.TestCase):
    def test_text_stats(self):
        text = "URGENT: Click http://win.com to claim $500 today! Call 123-456-7890."
        stats = compute_text_stats(text)
        self.assertGreater(stats["total_chars"], 0)
        self.assertGreater(stats["total_words"], 0)
        self.assertGreater(stats["uppercase_percentage"], 0.0)
        self.assertEqual(stats["urls_count"], 1)
        self.assertEqual(stats["currency_count"], 1)

    def test_generate_explanation(self):
        stats = compute_text_stats("URGENT: WIN $1000 NOW at http://claim.com")
        reasons = generate_explanation("SPAM", 95.0, stats, ["win", "urgent", "claim"])
        self.assertTrue(len(reasons) > 0)
        self.assertTrue(any("confidence" in r.lower() for r in reasons))


if __name__ == "__main__":
    unittest.main()
