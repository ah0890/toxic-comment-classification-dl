"""Unit tests for src/preprocessing.py.

Run from the project root:  python -m unittest discover tests
"""
import unittest

from src.preprocessing import clean_text


class CleanTextTests(unittest.TestCase):
    def test_lowercase_and_html(self):
        self.assertEqual(clean_text("<p>Hello &amp; WELCOME!</p>"), "hello welcome")

    def test_removes_urls(self):
        self.assertEqual(
            clean_text("Visit https://x.com/a?b=1 or www.site.org now"), "visit or now"
        )

    def test_removes_emails(self):
        self.assertEqual(clean_text("mail me@test.com pls"), "mail pls")

    def test_removes_digits_punctuation_and_emojis(self):
        self.assertEqual(clean_text("You're SO dumb!!! 😡😡 123"), "youre so dumb")

    def test_collapses_whitespace(self):
        self.assertEqual(clean_text("   lots    of\tspace\n"), "lots of space")

    def test_missing_values_become_empty_string(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(float("nan")), "")


if __name__ == "__main__":
    unittest.main()
