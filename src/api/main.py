"""
FastAPI entrypoint for Resume Copilot.

Run with: uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent.critique import get_resume_critique
from src.parsing.resume_parser import extract_text_from_pdf
from src.scoring.classifier import predict_category
from src.scoring.matcher import compute_match_score

app = FastAPI(title="Resume Copilot API", version="0.1.0")

# Allows the standalone frontend/index.html file (opened directly in a
# browser, or served from any local port) to call this API.
# Restricted to a wildcard here since this is a local portfolio project,
# not a production deployment -- would be locked down to a specific
# origin in a real production setting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    match_score: float
    predicted_category: str
    strengths: list[str]
    gaps: list[str]
    suggestions: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume_pdf: UploadFile = File(..., description="Resume as a PDF file"),
    job_description: str = Form(..., description="Target job description text"),
):
    """
    Accepts a resume PDF and a job description, and returns:
    - a semantic match score (0-100)
    - the resume's predicted job category
    - a structured critique (strengths, gaps, suggestions)
    """
    # 1. Parse the uploaded PDF into plain text
    pdf_bytes = await resume_pdf.read()
    try:
        resume_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Compute the semantic match score between resume and job description
    match_score = compute_match_score(resume_text, job_description)

    # 3. Predict the resume's job category
    category = predict_category(resume_text)

    # 4. Generate the structured critique via Mistral
    critique = get_resume_critique(
        resume_text=resume_text,
        job_description=job_description,
        match_score=match_score,
        predicted_category=category,
    )

    return AnalyzeResponse(
        match_score=match_score,
        predicted_category=category,
        strengths=critique["strengths"],
        gaps=critique["gaps"],
        suggestions=critique["suggestions"],
    )