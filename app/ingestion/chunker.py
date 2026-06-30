from dataclasses import dataclass, field
from app.ingestion.parser import ParsedPage
from app.api.core.config import settings
import re

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    total_chunks_in_page: int
    word_count: int
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Remove excessive whitespace and fix common PDF parsing artifacts."""
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove multiple newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page numbers that appear as lone numbers
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    # Strip
    text = text.strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences — respects abbreviations better than naive split."""
    # Split on period/exclamation/question followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(page: ParsedPage, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Split a single page into overlapping chunks.
    Splits at sentence boundaries — not mid-sentence.
    """
    text = clean_text(page.text)
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())

        # If adding this sentence exceeds chunk_size, save current chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Overlap — keep last N words for context continuity
            overlap_text = " ".join(current_chunk[-3:])  # last 3 sentences
            current_chunk = [overlap_text]
            current_length = len(overlap_text.split())
            chunk_index += 1

        current_chunk.append(sentence)
        current_length += sentence_len

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Convert to Chunk objects with metadata
    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{page.source_file}_p{page.page_number}_c{i}"
        result.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            source_file=page.source_file,
            page_number=page.page_number,
            chunk_index=i,
            total_chunks_in_page=len(chunks),
            word_count=len(chunk_text.split()),
            metadata={
                "source_file": page.source_file,
                "page_number": page.page_number,
                "chunk_index": i,
                "total_pages": page.total_pages,
                "chunk_id": chunk_id,
            }
        ))

    return result


def chunk_document(pages: list[ParsedPage]) -> list[Chunk]:
    """
    Chunk all pages of a document.
    Returns a flat list of all chunks with metadata.
    """
    all_chunks = []

    for page in pages:
        if not page.text.strip():
            continue
        chunks = chunk_page(
            page,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        all_chunks.extend(chunks)

    print(f"Document chunked: {len(pages)} pages → {len(all_chunks)} chunks")
    return all_chunks