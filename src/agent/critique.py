"""
Generates a structured resume critique using Mistral: strengths, gaps,
and actionable suggestions, given a resume, a job description, and the
match score / predicted category already computed by the scoring layer.

This is intentionally a single-pass call, not an agentic loop (score ->
critique -> rewrite -> re-score) -- see README for the scoping decision.
"""
import json
import os

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

_client = None


def get_client() -> Mistral:
    """
    Returns a cached Mistral client instance, creating it on first call.
    """
    global _client
    if _client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY not found. Make sure it's set in your .env file."
            )
        _client = Mistral(api_key=api_key)
    return _client


def get_resume_critique(
    resume_text: str,
    job_description: str,
    match_score: float,
    predicted_category: str,
) -> dict:
    """
    Sends the resume + job description to Mistral and returns a
    structured critique as a dict with keys: strengths, gaps, suggestions.
    """
    prompt = f"""You are a resume reviewer. Compare this resume against the job description and give structured feedback.

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

CONTEXT: This resume has a computed match score of {match_score:.1f}% and was classified into the category "{predicted_category}".

Respond ONLY with valid JSON in exactly this format, no other text:
{{
  "strengths": ["point 1", "point 2", "point 3"],
  "gaps": ["point 1", "point 2", "point 3"],
  "suggestions": ["specific actionable suggestion 1", "specific actionable suggestion 2", "specific actionable suggestion 3"]
}}"""

    client = get_client()
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)