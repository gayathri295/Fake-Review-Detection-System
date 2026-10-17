"""Download, inspect, and validate the preferred evidence-based review dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copyfileobj
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd


DATASET_SOURCE = "https://bretthollenbeck.com/fake-reviews-data/"
DATASET_ZIP_URL = (
    "https://bretthollenbeckcom.wordpress.com/wp-content/uploads/2026/03/"
    "public_reviews_dataset_cleaned.csv_.zip"
)
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DATASET_FILENAME = "public_reviews_dataset_cleaned.csv"
DATASET_ARCHIVE = DATASET_DIR / "public_reviews_dataset_cleaned.csv_.zip"
DEFAULT_DATASET_PATH = DATASET_DIR / DATASET_FILENAME


@dataclass(frozen=True)
class DatasetSummary:
    rows_before_cleaning: int
    rows_after_cleaning: int
    column_count: int
    source_columns: list[str]
    missing_values: dict[str, int]
    duplicate_rows: int
    unlabeled_rows_removed: int
    unusable_rows_removed: int
    conflicting_texts_removed: int
    duplicate_texts_removed: int
    class_counts: dict[str, int]
    first_five_records: list[dict[str, str]]

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_dataset(archive_path: Path, destination: Path) -> Path:
    """Extract only the expected CSV, never arbitrary paths from the ZIP."""
    try:
        with ZipFile(archive_path) as archive:
            try:
                entry = archive.getinfo(DATASET_FILENAME)
            except KeyError as exc:
                raise ValueError(
                    f"The dataset archive does not contain {DATASET_FILENAME}."
                ) from exc
            if entry.file_size > 1_000_000_000:
                raise ValueError("The dataset CSV in the archive is unexpectedly large.")
            with archive.open(entry) as source, destination.open("wb") as output:
                copyfileobj(source, output)
    except (BadZipFile, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Could not safely extract the dataset archive: {exc}") from exc
    return destination


def ensure_dataset_available() -> Path:
    """Use the preferred dataset, downloading it only when it is not present."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if DEFAULT_DATASET_PATH.is_file():
        return DEFAULT_DATASET_PATH
    if not DATASET_ARCHIVE.is_file():
        temporary_path = DATASET_ARCHIVE.with_suffix(
            DATASET_ARCHIVE.suffix + ".download"
        )
        try:
            with urlopen(DATASET_ZIP_URL, timeout=90) as response:
                if response.status != 200:
                    raise OSError(
                        f"Dataset server returned HTTP {response.status}."
                    )
                with temporary_path.open("wb") as output:
                    copyfileobj(response, output)
            temporary_path.replace(DATASET_ARCHIVE)
        except (OSError, URLError) as exc:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the preferred evidence-based dataset. "
                f"Download the CSV ZIP from {DATASET_SOURCE} and place it in "
                f"'{DATASET_DIR}', then restart the app."
            ) from exc
    return _extract_dataset(DATASET_ARCHIVE, DEFAULT_DATASET_PATH)


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, DatasetSummary]:
    """Construct labels only from the documented reviewer evidence flags."""
    if frame.empty:
        raise ValueError("The preferred dataset contains no rows.")

    required_columns = {
        "review_text",
        "reviewer_id",
        "reviewer_labeled_fake",
        "reviewer_labeled_honest",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "The preferred dataset is missing required documented columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    text = frame["review_text"].astype("string").str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    reviewer_id = frame["reviewer_id"].astype("string").str.strip()
    fake_flag = pd.to_numeric(frame["reviewer_labeled_fake"], errors="coerce")
    honest_flag = pd.to_numeric(frame["reviewer_labeled_honest"], errors="coerce")

    fake_mask = fake_flag.eq(1) & honest_flag.eq(0)
    genuine_mask = honest_flag.eq(1) & fake_flag.eq(0)
    label = pd.Series(pd.NA, index=frame.index, dtype="string")
    label.loc[fake_mask] = "Fake"
    label.loc[genuine_mask] = "Genuine"

    unusable_mask = text.isna() | text.eq("") | reviewer_id.isna() | reviewer_id.eq("")
    unlabelled_mask = label.isna() & ~unusable_mask
    labelled = pd.DataFrame(
        {"text": text, "label": label, "reviewer_id": reviewer_id}
    ).loc[~unusable_mask & label.notna()].copy()

    conflicting_texts = labelled.groupby("text", sort=False)["label"].nunique()
    conflicting_text_values = conflicting_texts[conflicting_texts > 1].index
    conflicting_mask = labelled["text"].isin(conflicting_text_values)
    conflicting_texts_removed = int(conflicting_mask.sum())
    labelled = labelled.loc[~conflicting_mask].copy()

    before_deduplication = len(labelled)
    labelled = labelled.drop_duplicates(subset=["text"], keep="first").reset_index(
        drop=True
    )
    duplicate_texts_removed = before_deduplication - len(labelled)

    class_counts = {
        name: int(labelled["label"].eq(name).sum()) for name in ("Fake", "Genuine")
    }
    if min(class_counts.values()) < 2:
        raise ValueError(
            "The documented reviewer-label fields must provide at least two "
            "usable examples for both Fake and Genuine."
        )

    preview_columns = [
        "review_text",
        "reviewer_labeled_fake",
        "reviewer_labeled_honest",
    ]
    first_five = frame.loc[:, preview_columns].head().fillna("").astype(str)
    summary = DatasetSummary(
        rows_before_cleaning=len(frame),
        rows_after_cleaning=len(labelled),
        column_count=len(frame.columns),
        source_columns=[str(column) for column in frame.columns],
        missing_values={
            str(column): int(count) for column, count in frame.isna().sum().items()
        },
        duplicate_rows=int(frame.duplicated().sum()),
        unlabeled_rows_removed=int(unlabelled_mask.sum()),
        unusable_rows_removed=int(unusable_mask.sum()),
        conflicting_texts_removed=conflicting_texts_removed,
        duplicate_texts_removed=duplicate_texts_removed,
        class_counts=class_counts,
        first_five_records=first_five.to_dict(orient="records"),
    )
    return labelled, summary


def load_dataset(
    path: Path | str | None = None,
) -> tuple[pd.DataFrame, DatasetSummary]:
    """Load and validate the preferred CSV (or a caller-supplied CSV for tests)."""
    dataset_path = Path(path) if path is not None else ensure_dataset_available()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {dataset_path}")
    try:
        frame = pd.read_csv(dataset_path, low_memory=False)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Could not read dataset file '{dataset_path}': {exc}") from exc
    return clean_dataset(frame)
