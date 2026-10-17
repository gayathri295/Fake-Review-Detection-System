import unittest

import pandas as pd

from data_loader import clean_dataset
from preprocessing import normalize_review


class DatasetCleaningTests(unittest.TestCase):
    def test_uses_only_one_hot_documented_reviewer_labels(self):
        source = pd.DataFrame(
            {
                "review_text": [
                    "Likely fake text",
                    "Likely genuine text",
                    "Second fake example",
                    "Second genuine example",
                    "Unlabeled text",
                    "Contradictory flags",
                ],
                "reviewer_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
                "reviewer_labeled_fake": [1, 0, 1, 0, 0, 1],
                "reviewer_labeled_honest": [0, 1, 0, 1, 0, 1],
            }
        )
        cleaned, summary = clean_dataset(source)
        self.assertEqual(
            cleaned["label"].tolist(), ["Fake", "Genuine", "Fake", "Genuine"]
        )
        self.assertEqual(summary.unlabeled_rows_removed, 2)
        self.assertEqual(summary.class_counts, {"Fake": 2, "Genuine": 2})

    def test_removes_duplicate_text_and_conflicting_duplicate_text(self):
        source = pd.DataFrame(
            {
                "review_text": [
                    "Repeated text",
                    "Repeated text",
                    "Conflicting text",
                    "Conflicting text",
                    "Another fake",
                    "Another genuine",
                    "One more genuine",
                ],
                "reviewer_id": ["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
                "reviewer_labeled_fake": [1, 1, 1, 0, 1, 0, 0],
                "reviewer_labeled_honest": [0, 0, 0, 1, 0, 1, 1],
            }
        )
        cleaned, summary = clean_dataset(source)
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(summary.duplicate_texts_removed, 1)
        self.assertEqual(summary.conflicting_texts_removed, 2)

    def test_rejects_missing_evidence_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required documented columns"):
            clean_dataset(
                pd.DataFrame({"review_text": ["Review"], "label": ["Fake"]})
            )


class ReviewPreprocessingTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace_without_joining_words(self):
        self.assertEqual(
            normalize_review("  VERY  Nice,\nProduct!  "), "very nice, product!"
        )


if __name__ == "__main__":
    unittest.main()
