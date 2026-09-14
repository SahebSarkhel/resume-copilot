"""
Extracts plain text from an uploaded resume PDF.

Known limitation: this uses simple text extraction, not layout-aware
parsing. Multi-column resumes or heavily designed templates may come
out with jumbled word order. This is an accepted tradeoff for a
portfolio project — noted here and in the README rather than hidden.
"""
import pdfplumber


def extract_text_from_pdf(pdf_path_or_file) -> str:
    """
    Extract all text from a PDF resume.

    Args:
        pdf_path_or_file: a file path (str) or a file-like object
                           (e.g. from FastAPI's UploadFile.file)

    Returns:
        Extracted text as a single string, with pages joined by newlines.
    """
    text_parts = []

    with pdfplumber.open(pdf_path_or_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # some pages may have no extractable text
                text_parts.append(page_text)

    full_text = "\n".join(text_parts)

    if not full_text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned/image-only document."
        )

    return full_text


if __name__ == "__main__":
    # Quick manual test — run with: python -m src.parsing.resume_parser
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not test_path:
        print("Usage: python -m src.parsing.resume_parser <path_to_pdf>")
    else:
        result = extract_text_from_pdf(test_path)
        print(f"Extracted {len(result)} characters")
        print("---")
        print(result[:500])
        print("---")