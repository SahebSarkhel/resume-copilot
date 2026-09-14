"""
Loads the trained resume-category classifier (Logistic Regression on
sentence embeddings, C=1.0 — see notebooks/01_eda.ipynb for the full
training, tuning, and evaluation process) and uses it to predict a
resume's category.
"""
import joblib

from src.scoring.embedder import embed_text

_classifier = None

# Path to the saved model, relative to the project root
MODEL_PATH = "data/resume_classifier.joblib"


def get_classifier():
    """
    Returns a cached classifier instance, loading it from disk on first call.
    """
    global _classifier
    if _classifier is None:
        _classifier = joblib.load(MODEL_PATH)
    return _classifier


def predict_category(resume_text: str) -> str:
    """
    Predicts the job category (e.g. 'INFORMATION-TECHNOLOGY', 'FINANCE')
    for a given resume's text.
    """
    clf = get_classifier()
    resume_vec = embed_text(resume_text)

    # Classifier expects a 2D array (a batch), so wrap in a list
    prediction = clf.predict([resume_vec])[0]
    return prediction