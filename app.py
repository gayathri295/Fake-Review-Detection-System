"""Minimal Streamlit interface for fake review detection."""

from __future__ import annotations

import streamlit as st

from data_loader import ensure_dataset_available
from preprocessing import normalize_review
from train_model import TrainingResult, train_model


st.set_page_config(page_title="Fake Review Detection System", layout="centered")
st.title("Fake Review Detection System")


@st.cache_resource(show_spinner="Preparing the review checker...")
def get_trained_model(
    dataset_path: str, dataset_modified_ns: int, dataset_size: int
) -> TrainingResult:
    del dataset_modified_ns, dataset_size
    return train_model(dataset_path)


review = st.text_area(
    "Enter product review",
    height=150,
    max_chars=20_000,
    placeholder="Type or paste a product review here...",
)

if st.button("Analyze", type="primary"):
    cleaned_review = normalize_review(review)
    if not cleaned_review:
        st.warning("Please enter a review.")
    elif len(cleaned_review) > 20_000:
        st.warning("Review must be 20,000 characters or fewer.")
    else:
        try:
            dataset_path = ensure_dataset_available()
            dataset_stat = dataset_path.stat()
            result = get_trained_model(
                str(dataset_path), dataset_stat.st_mtime_ns, dataset_stat.st_size
            )
            features = result.vectorizer.transform([cleaned_review])
            prediction = str(result.model.predict(features)[0])
            if prediction == "Fake":
                st.error("Likely Fake")
            else:
                st.success("Likely Genuine")
        except (
            AttributeError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            st.error(f"Could not analyze this review: {error}")
