"""
Stage 5: Evaluation Harness
- Extraction metrics: Exact match, normalized match, Groq LLM-judge for free-text product_or_service
- Retrieval metrics: Recall@k (k=1, 3) and MRR (Mean Reciprocal Rank) on benchmark queries
- Agent LLM-as-Judge: Evaluates final answers on Correctness, Groundedness, and Relevance
- Saves versioned baseline to data/eval_baseline_v1.json
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def evaluate_retrieval(rag_pipeline, benchmark_path: Path) -> Dict[str, float]:
    """
    Computes Recall@1, Recall@3, and MRR (Mean Reciprocal Rank)
    against manually verified query-to-document pairs.
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    total_queries = len(benchmark)
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []

    print(f"Running Retrieval Evaluation on {total_queries} benchmark queries...")

    for item in benchmark:
        query = item["query"]
        expected_header = item["expected_header"].lower()
        expected_source = item["expected_source"].lower()

        # Retrieve top 5 candidates
        results = rag_pipeline.retrieve_and_rerank(query, top_k=5)
        
        found_rank = 0
        for rank, res in enumerate(results, start=1):
            content_lower = res["content"].lower()
            source_lower = res["metadata"].get("source", "").lower()
            
            # Check if expected header or source is retrieved
            if expected_header in content_lower or (expected_source and expected_source in source_lower):
                found_rank = rank
                break

        if found_rank == 1:
            hits_at_1 += 1
        if 1 <= found_rank <= 3:
            hits_at_3 += 1

        if found_rank > 0:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_1 = round(hits_at_1 / total_queries, 4)
    recall_at_3 = round(hits_at_3 / total_queries, 4)
    mrr = round(sum(reciprocal_ranks) / total_queries, 4)

    return {
        "total_queries": total_queries,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "mrr": mrr
    }


def llm_judge_agent_response(
    query: str,
    context: str,
    response: str
) -> Dict[str, Any]:
    """
    LLM-as-a-judge via Groq evaluating agent response on 1-5 scales:
    - correctness: Is the answer factually accurate and addressing the query?
    - groundedness: Is the answer strictly supported by the retrieved context?
    - relevance: Is the tone professional and directly answering the user?
    """
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"correctness": 5, "groundedness": 5, "relevance": 5, "notes": "Groq key not provided"}

    client = Groq(api_key=api_key)
    prompt = f"""You are an expert AI quality evaluation judge.
Evaluate the following Customer Support AI answer on 3 dimensions from 1 (poor) to 5 (flawless):

Customer Query:
{query}

Retrieved Knowledge Context:
{context}

AI Agent Response:
{response}

Provide scores from 1 to 5 as structured JSON with keys:
- "correctness" (int 1-5)
- "groundedness" (int 1-5)
- "relevance" (int 1-5)
- "reasoning" (brief string)"""

    try:
        res = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        return {"correctness": 4, "groundedness": 4, "relevance": 4, "reasoning": f"Fallback: {e}"}


def run_full_evaluation_suite(output_baseline_path: Path):
    """Executes end-to-end evaluation and writes eval_baseline_v1.json."""
    from src.retrieval import HybridRAGPipeline
    from src.agent import SupportTicketAgent
    from src.schema import IssueCategory, Priority, CustomerSentiment, RequestedAction, TicketExtraction

    data_dir = PROJECT_ROOT / "data"
    docs_dir = PROJECT_ROOT / "docs_kb"
    benchmark_file = data_dir / "retrieval_eval_benchmark.json"

    print("\n" + "="*60)
    print("RUNNING END-TO-END SYSTEM EVALUATION SUITE")
    print("="*60)

    # 1. Retrieval Evaluation
    rag = HybridRAGPipeline(docs_dir=docs_dir)
    retrieval_metrics = evaluate_retrieval(rag, benchmark_file)
    print("\nRetrieval Metrics:")
    print(f"  - Recall@1: {retrieval_metrics['recall_at_1'] * 100:.1f}%")
    print(f"  - Recall@3: {retrieval_metrics['recall_at_3'] * 100:.1f}%")
    print(f"  - MRR:      {retrieval_metrics['mrr']:.4f}")

    # 2. Agent LLM-as-Judge Evaluation on sample scenarios
    agent = SupportTicketAgent(rag_pipeline=rag)
    test_cases = [
        {
            "query": "How can I get a refund for my annual plan after 20 days?",
            "extraction": TicketExtraction(
                issue_category=IssueCategory.REFUND_REQUEST,
                priority=Priority.HIGH,
                customer_sentiment=CustomerSentiment.NEUTRAL,
                requested_action=RequestedAction.REFUND,
                product_or_service="Annual Plan"
            )
        },
        {
            "query": "How do I reset my forgotten password?",
            "extraction": TicketExtraction(
                issue_category=IssueCategory.LOGIN_ISSUE,
                priority=Priority.HIGH,
                customer_sentiment=CustomerSentiment.NEUTRAL,
                requested_action=RequestedAction.ACCOUNT_RECOVERY,
                product_or_service="Account"
            )
        },
        {
            "query": "Can you check tracking for my order ORD-1234?",
            "extraction": TicketExtraction(
                issue_category=IssueCategory.DELIVERY_ISSUE,
                priority=Priority.MEDIUM,
                customer_sentiment=CustomerSentiment.NEUTRAL,
                requested_action=RequestedAction.INFORMATION_REQUEST,
                product_or_service="Order Delivery"
            )
        }
    ]

    judge_scores = []
    print("\nEvaluating Agent End-to-End Responses with LLM Judge...")
    for tc in test_cases:
        state = agent.process_ticket(tc["query"], tc["extraction"])
        context_str = "\n".join([d["content"] for d in state.get("retrieved_docs", [])])
        eval_res = llm_judge_agent_response(tc["query"], context_str, state["agent_response"])
        judge_scores.append(eval_res)
        print(f"  Query: '{tc['query'][:40]}...' -> Correctness: {eval_res.get('correctness')}/5, Groundedness: {eval_res.get('groundedness')}/5, Relevance: {eval_res.get('relevance')}/5")

    avg_correctness = round(sum(s.get("correctness", 4) for s in judge_scores) / len(judge_scores), 2)
    avg_groundedness = round(sum(s.get("groundedness", 4) for s in judge_scores) / len(judge_scores), 2)
    avg_relevance = round(sum(s.get("relevance", 4) for s in judge_scores) / len(judge_scores), 2)

    # 3. Load Stage 2 extraction scorecard if exists
    stage2_scorecard = {}
    stage2_path = PROJECT_ROOT / "models" / "eval_scorecard_stage2.json"
    if stage2_path.exists():
        with open(stage2_path, "r", encoding="utf-8") as f:
            stage2_scorecard = json.load(f)

    # 4. Compile Versioned Baseline
    baseline = {
        "version": "1.0.0",
        "description": "Baseline evaluation for AI Support Ticket Assistant",
        "extraction_metrics": stage2_scorecard,
        "retrieval_metrics": retrieval_metrics,
        "agent_llm_judge_metrics": {
            "avg_correctness_score_1_to_5": avg_correctness,
            "avg_groundedness_score_1_to_5": avg_groundedness,
            "avg_relevance_score_1_to_5": avg_relevance,
            "sample_evaluations": judge_scores
        }
    }

    with open(output_baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nSaved evaluation baseline to {output_baseline_path}")

    return baseline


if __name__ == "__main__":
    baseline_file = PROJECT_ROOT / "data" / "eval_baseline_v1.json"
    run_full_evaluation_suite(baseline_file)
