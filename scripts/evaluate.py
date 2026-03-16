"""
Evaluation harness for the USCIS Policy RAG system.

Runs a set of known question/answer pairs through the pipeline and
reports retrieval quality metrics: keyword recall, similarity scores,
and confidence distribution.

Usage:
    python -m scripts.evaluate
"""

import json
import logging
import sys
from dataclasses import dataclass

from app.rag.answer_generator import AnswerGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]
    description: str = ""


# Known policy questions with expected keywords that should appear
# in a correct answer. These are drawn from the USCIS Policy Manual.
EVAL_CASES = [
    EvalCase(
        question="What are the eligibility requirements for naturalization?",
        expected_keywords=[
            "18",
            "lawful permanent resident",
            "continuous residence",
            "physical presence",
            "good moral character",
            "english",
            "civics",
            "oath",
        ],
        description="Core naturalization requirements",
    ),
    EvalCase(
        question="How long must someone be a permanent resident before applying for citizenship?",
        expected_keywords=[
            "5 years",
            "3 years",
            "continuous residence",
            "lawful permanent resident",
        ],
        description="Residence duration requirement",
    ),
    EvalCase(
        question="What is the physical presence requirement for naturalization?",
        expected_keywords=[
            "physical presence",
            "30 months",
            "months",
        ],
        description="Physical presence requirement",
    ),
    EvalCase(
        question="Who is exempt from the English language requirement?",
        expected_keywords=[
            "age",
            "years",
            "disability",
            "exempt",
        ],
        description="English test exemptions",
    ),
    EvalCase(
        question="What disqualifies someone from good moral character?",
        expected_keywords=[
            "criminal",
            "moral character",
            "conviction",
        ],
        description="Good moral character bars",
    ),
    EvalCase(
        question="Can military service members apply for naturalization differently?",
        expected_keywords=[
            "military",
            "service",
            "armed forces",
        ],
        description="Military naturalization path",
    ),
    EvalCase(
        question="What is adjustment of status?",
        expected_keywords=[
            "adjustment of status",
            "permanent resident",
            "immigrant visa",
        ],
        description="Adjustment of status definition",
    ),
    EvalCase(
        question="What are grounds of inadmissibility?",
        expected_keywords=[
            "inadmissibility",
            "health",
            "criminal",
            "security",
        ],
        description="Inadmissibility overview",
    ),
]


def evaluate_case(generator: AnswerGenerator, case: EvalCase) -> dict:
    """Run a single evaluation case and compute metrics."""
    response = generator.answer(case.question)

    answer_lower = response.answer.lower()
    found = [kw for kw in case.expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in case.expected_keywords if kw.lower() not in answer_lower]
    recall = len(found) / len(case.expected_keywords) if case.expected_keywords else 0

    return {
        "question": case.question,
        "description": case.description,
        "expected_keywords": case.expected_keywords,
        "answer": response.answer,
        "confidence": response.confidence.value,
        "top_similarity": response.retrieval.top_similarity,
        "expanded_queries": response.retrieval.expanded_queries,
        "chunks_used": response.retrieval.chunks_sent_to_model,
        "keywords_found": found,
        "keywords_missing": missing,
        "keyword_recall": recall,
    }


def main():
    logger.info("Starting evaluation harness with %d cases", len(EVAL_CASES))
    generator = AnswerGenerator()

    results = []
    for i, case in enumerate(EVAL_CASES, start=1):
        logger.info("[%d/%d] %s", i, len(EVAL_CASES), case.description)
        try:
            result = evaluate_case(generator, case)
            results.append(result)

            logger.info(
                "  confidence=%s  top_sim=%.4f  recall=%.0f%%  (%d/%d keywords)",
                result["confidence"],
                result["top_similarity"] or 0,
                result["keyword_recall"] * 100,
                len(result["keywords_found"]),
                len(case.expected_keywords),
            )

            if result["keywords_missing"]:
                logger.info("  missing: %s", result["keywords_missing"])

        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            results.append({
                "question": case.question,
                "description": case.description,
                "error": str(exc),
                "keyword_recall": 0,
                "confidence": "error",
                "top_similarity": None,
            })

    # Aggregate metrics
    recalls = [r["keyword_recall"] for r in results if "error" not in r]
    similarities = [
        r["top_similarity"] for r in results
        if r.get("top_similarity") is not None
    ]

    confidence_dist: dict[str, int] = {}
    for r in results:
        level = r.get("confidence", "error")
        confidence_dist[level] = confidence_dist.get(level, 0) + 1

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions:         {len(EVAL_CASES)}")
    print(f"Avg keyword recall:      {sum(recalls) / len(recalls):.1%}" if recalls else "N/A")
    print(f"Avg top similarity:      {sum(similarities) / len(similarities):.4f}" if similarities else "N/A")
    print(f"Confidence distribution: {json.dumps(confidence_dist)}")
    print("=" * 70)

    for r in results:
        status = "PASS" if r.get("keyword_recall", 0) >= 0.5 else "FAIL"
        print(
            f"  [{status}] {r['description']}: "
            f"recall={r.get('keyword_recall', 0):.0%}  "
            f"confidence={r.get('confidence', 'error')}"
        )

    print("=" * 70)

    # Write detailed results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Detailed results written to eval_results.json")

    # Exit with failure if average recall is too low
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    if avg_recall < 0.5:
        logger.error("Average keyword recall %.1f%% is below 50%% threshold.", avg_recall * 100)
        sys.exit(1)


if __name__ == "__main__":
    main()
