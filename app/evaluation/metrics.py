import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
import nest_asyncio
nest_asyncio.apply()


from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from app.api.core.config import settings
from app.generation.rag_pipeline import answer_query

TEST_SET_PATH = Path("data/eval/test_questions.json")
RESULTS_PATH = Path("data/eval/eval_results.json")

# Use Groq as the judge LLM instead of OpenAI
judge_llm = ChatGroq(
    api_key=settings.groq_api_key,
    model=settings.llm_model,
    temperature=0
)

judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from app.generation.rag_pipeline import answer_query

TEST_SET_PATH = Path("data/eval/test_questions.json")
RESULTS_PATH = Path("data/eval/eval_results.json")


def load_test_questions() -> list[dict]:
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_test_set(use_rerank: bool = True, use_rewrite: bool = True) -> list[dict]:
    """
    Run the full RAG pipeline on every test question.
    Collects question, answer, retrieved contexts, and ground truth
    in the format RAGAS expects.
    """
    test_questions = load_test_questions()
    eval_rows = []

    for item in test_questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        result = answer_query(
            query=question,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank
        )

        contexts = [
            f"(from {c['source_file']}, page {c['page_number']})"
            for c in result["citations"]
        ]
        # RAGAS needs actual chunk text, not just citation labels —
        # re-fetch the text from citations for accurate context scoring
        from app.retrieval.hybrid_search import hybrid_search
        from app.retrieval.reranker import rerank as rerank_fn

        retrieve_k = 15 if use_rerank else 5
        raw_results = hybrid_search(question, top_k=retrieve_k)
        if use_rerank:
            raw_results = rerank_fn(question, raw_results, top_k=5)
        else:
            raw_results = raw_results[:5]

        context_texts = [r.text for r in raw_results]

        eval_rows.append({
            "question": question,
            "answer": result["answer"],
            "contexts": context_texts,
            "ground_truth": ground_truth
        })

        print(f"Processed: {question[:60]}...")

    return eval_rows


def run_ragas_evaluation(eval_rows: list[dict]) -> dict:
    """
    Run RAGAS metrics on the collected eval rows.
    Returns a dict of average scores.
    """
    dataset = Dataset.from_list(eval_rows)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,        # does the answer stick to retrieved context?
            answer_relevancy,    # does the answer actually address the question?
            context_precision,   # are retrieved chunks relevant?
            context_recall,      # did retrieval find what's needed for ground truth?
        ],
        llm=judge_llm,
        embeddings=judge_embeddings  
    )

    return result.to_pandas().mean(numeric_only=True).to_dict()


def evaluate_pipeline(use_rerank: bool = True, use_rewrite: bool = True) -> dict:
    """
    Full evaluation run: pipeline execution + RAGAS scoring.
    Saves results to disk and returns the summary.
    """
    print(f"Running evaluation (rerank={use_rerank}, rewrite={use_rewrite})...")
    eval_rows = run_pipeline_on_test_set(use_rerank=use_rerank, use_rewrite=use_rewrite)

    print("Scoring with RAGAS...")
    scores = run_ragas_evaluation(eval_rows)

    summary = {
        "config": {"use_rerank": use_rerank, "use_rewrite": use_rewrite},
        "num_questions": len(eval_rows),
        "scores": {k: round(v, 4) for k, v in scores.items()}
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary