from app.retrieval.semantic_search import SearchResult

MAX_CONTEXT_CHARS = 6000  # keeps prompt size reasonable for token budget


def build_context(results: list[SearchResult]) -> tuple[str, list[dict]]:
    """
    Build a context string from search results, formatted with source labels.
    Also returns a citation map so the LLM's [Source N] references can be resolved.

    Returns: (context_string, citation_map)
    """
    context_parts = []
    citation_map = []
    total_chars = 0

    for i, result in enumerate(results):
        source_label = f"[Source {i + 1}]"
        source_file = result.metadata.get("source_file", "unknown")
        page_number = result.metadata.get("page_number", "?")

        chunk_text = result.text.strip()
        entry = f"{source_label} (from {source_file}, page {page_number}):\n{chunk_text}"

        # Stop adding chunks once we hit the character budget
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(entry)
        total_chars += len(entry)

        citation_map.append({
            "source_number": i + 1,
            "chunk_id": result.chunk_id,
            "source_file": source_file,
            "page_number": page_number,
            "score": round(result.score, 4)
        })

    context_string = "\n\n".join(context_parts)
    return context_string, citation_map


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """
    Remove near-duplicate chunks (same source_file + page_number + similar text start).
    Helps reduce wasted tokens on redundant context.
    """
    seen = set()
    deduped = []

    for result in results:
        key = (
            result.metadata.get("source_file"),
            result.metadata.get("page_number"),
            result.text[:50]  # first 50 chars as a fingerprint
        )
        if key not in seen:
            seen.add(key)
            deduped.append(result)

    return deduped