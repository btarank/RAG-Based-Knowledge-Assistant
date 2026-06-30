import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.evaluation.metrics import evaluate_pipeline

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.post("/run")
async def run_evaluation(use_rerank: bool = True, use_rewrite: bool = True):
    try:
        result = await run_in_threadpool(
            evaluate_pipeline, use_rerank, use_rewrite
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_configurations():
    try:
        with_rerank = await run_in_threadpool(
            evaluate_pipeline, True, True
        )
        await asyncio.sleep(30) 
        without_rerank = await run_in_threadpool(
            evaluate_pipeline, False, False
        )

        improvement = {}
        for metric in with_rerank["scores"]:
            before = without_rerank["scores"].get(metric, 0)
            after = with_rerank["scores"].get(metric, 0)
            if before > 0:
                pct_change = round(((after - before) / before) * 100, 2)
            else:
                pct_change = None
            improvement[metric] = {
                "before": before,
                "after": after,
                "improvement_pct": pct_change
            }

        return {
            "with_hybrid_and_rerank": with_rerank["scores"],
            "baseline_no_optimization": without_rerank["scores"],
            "improvement": improvement
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))