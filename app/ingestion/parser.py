import fitz  # pymupdf
import pdfplumber
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ParsedPage:
    text: str
    page_number: int
    source_file: str
    total_pages: int

def parse_pdf(file_path: str) -> list[ParsedPage]:
    """
    Parse a PDF file and return a list of ParsedPage objects.
    Uses PyMuPDF for speed, falls back to pdfplumber for complex PDFs.
    """
    path = Path(file_path)
    pages = []

    try:
        # Primary parser — PyMuPDF (fast)
        doc = fitz.open(file_path)
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text").strip()

            # If page has very little text, try pdfplumber (handles tables better)
            if len(text) < 50:
                text = _extract_with_pdfplumber(file_path, page_num)

            if text:  # skip empty pages
                pages.append(ParsedPage(
                    text=text,
                    page_number=page_num + 1,
                    source_file=path.name,
                    total_pages=total_pages
                ))

        doc.close()

    except Exception as e:
        print(f"PyMuPDF failed: {e}, trying pdfplumber...")
        pages = _parse_with_pdfplumber(file_path)

    return pages


def _extract_with_pdfplumber(file_path: str, page_num: int) -> str:
    """Extract text from a single page using pdfplumber."""
    try:
        with pdfplumber.open(file_path) as pdf:
            page = pdf.pages[page_num]
            text = page.extract_text() or ""
            return text.strip()
    except:
        return ""


def _parse_with_pdfplumber(file_path: str) -> list[ParsedPage]:
    """Full fallback parser using pdfplumber."""
    path = Path(file_path)
    pages = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(ParsedPage(
                    text=text,
                    page_number=page_num + 1,
                    source_file=path.name,
                    total_pages=total_pages
                ))

    return pages


def parse_txt(file_path: str) -> list[ParsedPage]:
    """Parse a plain text file as a single page."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [ParsedPage(
        text=text,
        page_number=1,
        source_file=path.name,
        total_pages=1
    )]


def parse_document(file_path: str) -> list[ParsedPage]:
    """
    Auto-detect file type and parse accordingly.
    Supports: .pdf, .txt
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")