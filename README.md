# Fake Review Detection System

A beginner-friendly Streamlit project that uses review text, TF-IDF, and
Logistic Regression to estimate whether a product review resembles research-
labeled fake or genuine examples. It is a learning and evaluation project, not
a tool for proving fraud or truth.

## Features

- Automatically downloads the preferred evidence-based Amazon review dataset.
- Checks the source schema, missing values, duplicates, label distribution, and
  the first five records before training.
- Constructs labels only from the documented reviewer-label fields. It does
  not infer fake labels from star ratings or positive sentiment.
- Uses normalized review text as the only model feature; reviewer, product,
  rating, campaign, and label-derived fields are excluded from the vectorizer.
- Uses a reproducible, reviewer-grouped train/validation/test split to reduce
  leakage from the same reviewer appearing in multiple sets.
- Selects Logistic Regression regularization using validation macro F1 and
  reports held-out test metrics, including ROC-AUC and a confusion matrix.
- Provides a compact Streamlit dashboard, a review text box, an Analyze button,
  and results directly below the button.
- Keeps the trained model in Streamlit memory with `st.cache_resource`.
  No model or vectorizer is saved to disk. Restarting the app process requires
  training again.

## Dataset and label definitions

**Preferred dataset:** [Amazon Fake Reviews Dataset — evidence-based review
labels](https://bretthollenbeck.com/fake-reviews-data/). The documentation
describes labels based on evidence and research rules related to paid review
campaigns. The published CSV used by this project is downloaded from the CSV
ZIP link on that page and extracted into `dataset/`.

This CSV provides the `reviewer_labeled_fake` and
`reviewer_labeled_honest` flags. The project maps an explicitly one-hot label
pair to a usable review label:

- `reviewer_labeled_fake == 1` and `reviewer_labeled_honest == 0` → **Fake**
- `reviewer_labeled_honest == 1` and `reviewer_labeled_fake == 0` → **Genuine**
- Pairs that are both zero, missing, or contradictory are not used as target
  labels.

The target is therefore a documented reviewer-evidence label inherited by a
review, not a universal guarantee about that review. A reviewer label does not
prove an individual review was paid for, and a genuine label does not prove a
review is accurate. The app’s predictions and probability scores are model
estimates only.

The ZIP is extracted by reading only the expected CSV member. The existing
unrelated `data/fake_reviews.csv` is deliberately not used as a substitute.
Manual setup: download the CSV ZIP linked from the source page and put the ZIP
or the extracted `public_reviews_dataset_cleaned.csv` in the project's
`dataset/` folder.

## Technology

- Python 3.10+
- Pandas and NumPy
- Scikit-learn: `TfidfVectorizer` and `LogisticRegression`
- Streamlit
- Matplotlib and Seaborn for the confusion-matrix visualization

There is no Flask, NLTK, deep-learning framework, Pickle, or Joblib-based model
storage in this project.

## Project structure

```text
fakenewsdetection/
├── dataset/
│   ├── public_reviews_dataset_cleaned.csv
│   └── public_reviews_dataset_cleaned.csv_.zip
├── app.py
├── data_loader.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
├── README.md
└── tests/
    └── test_pipeline.py
```

The downloaded dataset is local and ignored by Git. No model artifacts are
created. Streamlit caches the trained objects in memory until its cache is
cleared, the dataset changes, or the app process restarts.

## Install and run in Windows PowerShell

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The first run downloads and extracts the preferred CSV if it is not already
present, then prints model evaluation results. The Streamlit app trains once
per process and caches the in-memory model for reruns. If PowerShell blocks
virtual-environment activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window,
then activate again. Alternatively, invoke
`.\.venv\Scripts\python.exe` and
`.\.venv\Scripts\streamlit.exe` directly.

## Evaluation protocol and results

Training uses TF-IDF with a maximum of 10,000 features and unigram/bigram
features. Whitespace and case are normalized; punctuation and word boundaries
are preserved, and stop words are not removed. The vectorizer is fitted only
on the training partition. Candidate Logistic Regression `C` values are
compared on the validation partition. The held-out test set is grouped by
reviewer and is not used for model selection.

Actual execution using the downloaded CSV:

| Dataset / evaluation item | Result |
|---|---:|
| Source records / columns | 381,734 / 23 |
| Usable labeled reviews after exclusions | 32,396 |
| Fake / Genuine class counts | 22,068 / 10,328 |
| Train / validation / test reviews | 19,437 / 6,479 / 6,480 |
| Validation-selected Logistic Regression C | 2.0 |
| Test accuracy | 76.30% |
| Test macro precision | 73.16% |
| Test macro recall | 69.35% |
| Test macro F1 | 70.51% |
| Test ROC-AUC | 0.7909 |

Test confusion matrix (rows = actual; columns = predicted; order = Fake,
Genuine):

|  | Predicted Fake | Predicted Genuine |
|---|---:|---:|
| Actual Fake | 3,907 | 507 |
| Actual Genuine | 1,029 | 1,037 |

Metrics include accuracy, macro precision, macro recall, macro F1, ROC-AUC,
and a confusion matrix. Scores below 95% are possible and expected for a
text-only model of this challenging task; do not treat any particular
accuracy as guaranteed.

Run the unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## Limitations and possible improvements

- The model learns language correlations for a research-labeled dataset, not
  direct evidence that a particular review is fake.
- Reviewer-level labels are imperfect proxies and reviews without a clear
  one-hot reviewer label are excluded rather than assigned an invented class.
- Language patterns can vary across product categories, marketplaces, time
  periods, and languages. A separate external dataset would be needed to
  measure cross-domain generalization.
- Estimated Logistic Regression probabilities are not necessarily calibrated.
- Future work could test probability calibration, evaluate a separate
  marketplace or time period, compare other simple linear baselines, and
  examine error patterns without using test data for tuning.
