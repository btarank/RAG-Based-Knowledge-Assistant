import json
import time
from pathlib import Path
from app.api.core.llm import get_llm_response
from app.generation.rag_pipeline import answer_query
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank as rerank_fn
import numpy as np

TEST_SET_PATH = Path("data/eval/test_questions.json")
RESULTS_PATH = Path("data/eval/eval_results.json")

CALL_DELAY = 3.0  # seconds between every Groq call — safely under 30 RPM


def _safe_llm_call(prompt: str) -> str:
    time.sleep(CALL_DELAY)
    try:
        return get_llm_response([{"role": "user", "content": prompt}], temperature=0.0)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""


def _parse_score(text: str) -> float:
    import re
    match = re.search(r"(\d*\.?\d+)", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.0


def _score_faithfulness(answer: str, context: str) -> float:
    prompt = f"""Context:
{context}

Answer:
{answer}

Question: Is every claim in the Answer directly supported by the Context above?
Reply with ONLY a number between 0 and 1, where 1 means fully supported and 0 means not supported at all. No explanation, just the number."""
    return _parse_score(_safe_llm_call(prompt))


def _score_relevance(question: str, answer: str) -> float:
    prompt = f"""Question: {question}

Answer: {answer}

Does the Answer directly and completely address the Question?
Reply with ONLY a number between 0 and 1. No explanation, just the number."""
    return _parse_score(_safe_llm_call(prompt))


def _score_context_relevance(question: str, context: str) -> float:
    prompt = f"""Question: {question}

Retrieved context:
{context}

How relevant is this context to answering the question?
Reply with ONLY a number between 0 and 1. No explanation, just the number."""
    return _parse_score(_safe_llm_call(prompt))


def load_test_questions() -> list[dict]:
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_pipeline(use_rerank: bool = True, use_rewrite: bool = True) -> dict:
    test_questions = load_test_questions()

    faithfulness_scores = []
    relevance_scores = []
    context_scores = []

    print(f"Running evaluation (rerank={use_rerank}, rewrite={use_rewrite}) on {len(test_questions)} questions...")

    for i, item in enumerate(test_questions):
        question = item["question"]
        print(f"[{i+1}/{len(test_questions)}] {question[:60]}...")

        result = answer_query(query=question, use_rewrite=use_rewrite, use_rerank=use_rerank)
        answer = result["answer"]

        retrieve_k = 15 if use_rerank else 5
        raw_results = hybrid_search(question, top_k=retrieve_k)
        if use_rerank:
            raw_results = rerank_fn(question, raw_results, top_k=5)
        else:
            raw_results = raw_results[:5]

        context = "\n\n".join([r.text for r in raw_results]) or "No context retrieved."

        f_score = _score_faithfulness(answer, context)
        r_score = _score_relevance(question, answer)
        c_score = _score_context_relevance(question, context)

        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)
        context_scores.append(c_score)

        print(f"  faithfulness={f_score:.2f}  relevance={r_score:.2f}  context={c_score:.2f}")

    summary = {
        "config": {"use_rerank": use_rerank, "use_rewrite": use_rewrite},
        "num_questions": len(test_questions),
        "scores": {
            "faithfulness": round(float(np.mean(faithfulness_scores)), 4),
            "answer_relevancy": round(float(np.mean(relevance_scores)), 4),
            "context_relevancy": round(float(np.mean(context_scores)), 4),
        }
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation complete:", summary["scores"])
    return summary