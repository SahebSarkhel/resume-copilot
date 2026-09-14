"""
Computes similarity between a resume and a job description using
their embeddings. This is the core "match score" for the product.
"""
from sklearn.metrics.pairwise import cosine_similarity

from src.scoring.embedder import embed_text


def compute_match_score(resume_text: str, job_description: str) -> float:
    """
    Returns a match score from 0-100 representing how semantically
    similar the resume is to the job description.
    """
    resume_vec = embed_text(resume_text)
    job_vec = embed_text(job_description)

    similarity = cosine_similarity([resume_vec], [job_vec])[0][0]
    return round(float(similarity) * 100, 2)