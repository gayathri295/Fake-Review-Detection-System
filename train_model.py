"""Train and evaluate the TF-IDF + Logistic Regression baseline in memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from data_loader import DatasetSummary, load_dataset
from preprocessing import normalize_review


RANDOM_STATE = 42
CLASS_NAMES = ["Fake", "Genuine"]
VECTORIZER_PARAMETERS = {
    "max_features": 10_000,
    "ngram_range": (1, 2),
    "min_df": 2,
}
C_VALUES = (0.5, 1.0, 2.0)


@dataclass
class TrainingResult:
    model: LogisticRegression
    vectorizer: TfidfVectorizer
    summary: DatasetSummary
    metrics: dict
    selected_c: float
    validation_macro_f1: float
    training_rows: int
    validation_rows: int
    test_rows: int
    vocabulary_size: int


def _macro_f1(model: LogisticRegression, features, labels: pd.Series) -> float:
    return float(
        f1_score(labels, model.predict(features), average="macro", zero_division=0)
    )


def train_model(dataset_path: Path | str | None = None) -> TrainingResult:
    """Group-split by reviewer, tune C on validation, and evaluate once on test."""
    reviews, summary = load_dataset(dataset_path)
    texts = reviews["text"].map(normalize_review)
    labels = reviews["label"]
    groups = reviews["reviewer_id"]

    outer_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_validation_indices, test_indices = next(
        outer_splitter.split(np.zeros(len(labels)), labels, groups)
    )

    inner_splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=RANDOM_STATE
    )
    training_relative_indices, validation_relative_indices = next(
        inner_splitter.split(
            np.zeros(len(train_validation_indices)),
            labels.iloc[train_validation_indices],
            groups.iloc[train_validation_indices],
        )
    )
    training_indices = train_validation_indices[training_relative_indices]
    validation_indices = train_validation_indices[validation_relative_indices]

    x_train = texts.iloc[training_indices]
    y_train = labels.iloc[training_indices]
    x_validation = texts.iloc[validation_indices]
    y_validation = labels.iloc[validation_indices]
    x_test = texts.iloc[test_indices]
    y_test = labels.iloc[test_indices]

    if any(set(values) != set(CLASS_NAMES) for values in (y_train, y_validation, y_test)):
        raise ValueError(
            "The grouped train/validation/test split did not contain both labels. "
            "Check that the dataset has enough labeled reviewers in each class."
        )

    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMETERS)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_validation_tfidf = vectorizer.transform(x_validation)
    x_test_tfidf = vectorizer.transform(x_test)

    validation_scores: dict[float, float] = {}
    validation_models: dict[float, LogisticRegression] = {}
    for c_value in C_VALUES:
        candidate = LogisticRegression(
            C=c_value, max_iter=1000, random_state=RANDOM_STATE
        )
        candidate.fit(x_train_tfidf, y_train)
        validation_models[c_value] = candidate
        validation_scores[c_value] = _macro_f1(
            candidate, x_validation_tfidf, y_validation
        )

    selected_c = max(C_VALUES, key=lambda value: validation_scores[value])
    model = validation_models[selected_c]
    predictions = model.predict(x_test_tfidf)
    fake_probability_index = list(model.classes_).index("Fake")
    fake_probabilities = model.predict_proba(x_test_tfidf)[:, fake_probability_index]
    y_test_fake = y_test.eq("Fake").astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=CLASS_NAMES
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "roc_auc": float(roc_auc_score(y_test_fake, fake_probabilities)),
    }

    return TrainingResult(
        model=model,
        vectorizer=vectorizer,
        summary=summary,
        metrics=metrics,
        selected_c=selected_c,
        validation_macro_f1=validation_scores[selected_c],
        training_rows=len(training_indices),
        validation_rows=len(validation_indices),
        test_rows=len(test_indices),
        vocabulary_size=len(vectorizer.vocabulary_),
    )


if __name__ == "__main__":
    try:
        result = train_model()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Training failed: {exc}") from exc

    print(f"Dataset rows: {result.summary.rows_before_cleaning:,}")
    print(f"Dataset columns: {result.summary.column_count}")
    print(f"Labeled, usable reviews: {result.summary.rows_after_cleaning:,}")
    print(f"Fake: {result.summary.class_counts['Fake']:,}")
    print(f"Genuine: {result.summary.class_counts['Genuine']:,}")
    print(
        "Train / validation / test: "
        f"{result.training_rows:,} / {result.validation_rows:,} / "
        f"{result.test_rows:,} (reviewers kept separate)"
    )
    print(f"Selected C: {result.selected_c}")
    print(f"Validation macro F1: {result.validation_macro_f1:.4f}")
    print(f"Vocabulary size: {result.vocabulary_size:,}")
    print("Held-out test metrics:")
    for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"):
        print(f"  {name}: {result.metrics[name]:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; Fake, Genuine):")
    for row in result.metrics["confusion_matrix"]:
        print(f"  {row}")
    print("The model is held in memory only; no model file was saved.")
