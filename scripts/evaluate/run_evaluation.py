"""
GraphRAG Evaluation Runner

Runs the evaluation dataset against the GraphRAG system and measures:
1. Retrieval Quality: Entity matching precision/recall
2. Answer Quality: Keyword coverage, LLM-as-judge scoring
3. Performance: Query latency

Usage:
    python scripts/evaluate/run_evaluation.py
    python scripts/evaluate/run_evaluation.py --subset 10  # Run on first 10 questions
    python scripts/evaluate/run_evaluation.py --category cpf  # Run only CPF questions
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

from eval_dataset import (
    EvalQuestion,
    Category,
    QuestionType,
    Difficulty,
    load_dataset,
    get_all_questions,
)
from llm_judge import LLMJudge, JudgeResult


# Configuration
BACKEND_URL = "http://localhost:8001"
RESULTS_DIR = Path(__file__).parent / "eval_data" / "results"


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    expected_entities: list[str]
    matched_entities: list[str]
    matched_nodes: list[str]
    precision: float  # matched_expected / total_matched
    recall: float     # matched_expected / total_expected
    f1: float
    source_chunks_loaded: int


@dataclass
class AnswerMetrics:
    """Metrics for answer quality."""
    answer: str
    answer_keywords: list[str]
    keywords_found: list[str]
    keywords_missing: list[str]
    keyword_coverage: float  # % of expected keywords found
    answer_length: int


@dataclass
class PerformanceMetrics:
    """Metrics for query performance."""
    latency_ms: float
    search_mode: str
    resolved_mode: str


@dataclass
class EvalResult:
    """Complete evaluation result for a single question."""
    question_id: str
    question: str
    category: str
    question_type: str
    difficulty: str
    retrieval: RetrievalMetrics
    answer: AnswerMetrics
    performance: PerformanceMetrics
    success: bool
    error: str | None = None
    judge: JudgeResult | None = None

    def to_dict(self) -> dict:
        d = {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "retrieval": asdict(self.retrieval),
            "answer": asdict(self.answer),
            "performance": asdict(self.performance),
            "success": self.success,
            "error": self.error,
        }
        if self.judge is not None:
            d["judge"] = self.judge.to_dict()
        return d


def check_backend_health() -> bool:
    """Check if backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def query_graphrag(question: str, session_id: str | None = None) -> dict:
    """Query the GraphRAG backend."""
    response = requests.post(
        f"{BACKEND_URL}/query",
        json={
            "question": question,
            "session_id": session_id,
            "temperature": 0.3,  # Lower temperature for consistency
            "max_tokens": 1024,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def calculate_entity_overlap(expected: list[str], retrieved: list[str]) -> tuple[list[str], float, float]:
    """
    Calculate overlap between expected and retrieved entities.
    Uses fuzzy matching (substring/word overlap).

    Returns:
        (matched_entities, precision, recall)
    """
    # Normalize for comparison
    expected_normalized = set(e.lower().replace("_", " ").replace("-", " ") for e in expected)
    retrieved_normalized = {r.lower().replace("_", " ").replace("-", " ") for r in retrieved}

    matched = []
    for exp in expected_normalized:
        exp_words = set(exp.split())
        for ret in retrieved_normalized:
            ret_words = set(ret.split())
            # Match if significant word overlap or substring match
            overlap = exp_words & ret_words
            if len(overlap) >= min(2, len(exp_words)) or exp in ret or ret in exp:
                matched.append(exp)
                break

    # Calculate metrics
    precision = len(matched) / len(retrieved) if retrieved else 0.0
    recall = len(matched) / len(expected) if expected else 0.0

    return matched, precision, recall


def calculate_keyword_coverage(answer: str, keywords: list[str]) -> tuple[list[str], list[str], float]:
    """
    Check which expected keywords appear in the answer.

    Returns:
        (keywords_found, keywords_missing, coverage_ratio)
    """
    answer_lower = answer.lower()
    found = []
    missing = []

    for keyword in keywords:
        keyword_lower = keyword.lower()
        # Check exact match or partial match for multi-word keywords
        if keyword_lower in answer_lower:
            found.append(keyword)
        elif all(word in answer_lower for word in keyword_lower.split()):
            found.append(keyword)
        else:
            missing.append(keyword)

    coverage = len(found) / len(keywords) if keywords else 0.0
    return found, missing, coverage


def evaluate_question(question: EvalQuestion, judge: LLMJudge | None = None) -> EvalResult:
    """Evaluate a single question."""
    start_time = time.perf_counter()
    error = None
    success = True

    try:
        # Query the system
        result = query_graphrag(question.question)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract retrieval info
        context_summary = result.get("context_summary", {})
        matched_entities = context_summary.get("matched_entities", [])
        matched_nodes = context_summary.get("matched_nodes", [])
        source_chunks = context_summary.get("source_chunks_loaded", 0)

        # Calculate retrieval metrics
        overlap, precision, recall = calculate_entity_overlap(
            question.expected_entities, matched_nodes
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        retrieval_metrics = RetrievalMetrics(
            expected_entities=question.expected_entities,
            matched_entities=matched_entities,
            matched_nodes=matched_nodes,
            precision=precision,
            recall=recall,
            f1=f1,
            source_chunks_loaded=source_chunks,
        )

        # Calculate answer metrics
        answer = result.get("answer", "")
        keywords_found, keywords_missing, coverage = calculate_keyword_coverage(
            answer, question.answer_keywords
        )

        answer_metrics = AnswerMetrics(
            answer=answer,
            answer_keywords=question.answer_keywords,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            keyword_coverage=coverage,
            answer_length=len(answer),
        )

        # Performance metrics
        performance_metrics = PerformanceMetrics(
            latency_ms=latency_ms,
            search_mode="auto",
            resolved_mode=result.get("resolved_mode", "unknown"),
        )

    except Exception as e:
        logger.error(f"Failed to evaluate {question.id}: {e}")
        error = str(e)
        success = False
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Create empty metrics for failed query
        retrieval_metrics = RetrievalMetrics(
            expected_entities=question.expected_entities,
            matched_entities=[],
            matched_nodes=[],
            precision=0.0,
            recall=0.0,
            f1=0.0,
            source_chunks_loaded=0,
        )

        answer_metrics = AnswerMetrics(
            answer="",
            answer_keywords=question.answer_keywords,
            keywords_found=[],
            keywords_missing=question.answer_keywords,
            keyword_coverage=0.0,
            answer_length=0,
        )

        performance_metrics = PerformanceMetrics(
            latency_ms=latency_ms,
            search_mode="auto",
            resolved_mode="error",
        )

    # Run LLM judge if provided
    judge_result = None
    if judge is not None and success:
        judge_result = judge.judge(
            question=question.question,
            answer=answer_metrics.answer,
            expected_entities=question.expected_entities,
            expected_keywords=question.answer_keywords,
        )

    return EvalResult(
        question_id=question.id,
        question=question.question,
        category=question.category.value,
        question_type=question.question_type.value,
        difficulty=question.difficulty.value,
        retrieval=retrieval_metrics,
        answer=answer_metrics,
        performance=performance_metrics,
        success=success,
        error=error,
        judge=judge_result,
    )


def run_evaluation(
    questions: list[EvalQuestion],
    show_progress: bool = True,
    judge: LLMJudge | None = None,
) -> list[EvalResult]:
    """Run evaluation on a list of questions."""
    results = []

    for i, question in enumerate(questions):
        if show_progress:
            print(f"[{i+1}/{len(questions)}] Evaluating: {question.id} - {question.question[:50]}...")

        result = evaluate_question(question, judge=judge)
        results.append(result)

        if show_progress:
            status = "OK" if result.success else "FAIL"
            judge_str = ""
            if result.judge is not None and result.judge.overall_score is not None:
                judge_str = f" | Judge: {result.judge.overall_score:.1f}/5"
            print(f"  {status} | Recall: {result.retrieval.recall:.2f} | Keywords: {result.answer.keyword_coverage:.2f} | Latency: {result.performance.latency_ms:.0f}ms{judge_str}")

    return results


def compute_aggregate_metrics(results: list[EvalResult]) -> dict:
    """Compute aggregate metrics across all results."""
    successful = [r for r in results if r.success]

    if not successful:
        return {"error": "No successful evaluations"}

    # Retrieval metrics
    precisions = [r.retrieval.precision for r in successful]
    recalls = [r.retrieval.recall for r in successful]
    f1s = [r.retrieval.f1 for r in successful]

    # Answer metrics
    keyword_coverages = [r.answer.keyword_coverage for r in successful]
    answer_lengths = [r.answer.answer_length for r in successful]

    # Performance metrics
    latencies = [r.performance.latency_ms for r in successful]

    metrics = {
        "total_questions": len(results),
        "successful_queries": len(successful),
        "success_rate": len(successful) / len(results),

        "retrieval": {
            "mean_precision": statistics.mean(precisions),
            "mean_recall": statistics.mean(recalls),
            "mean_f1": statistics.mean(f1s),
            "median_recall": statistics.median(recalls),
            "recall_std": statistics.stdev(recalls) if len(recalls) > 1 else 0,
        },

        "answer_quality": {
            "mean_keyword_coverage": statistics.mean(keyword_coverages),
            "median_keyword_coverage": statistics.median(keyword_coverages),
            "keyword_coverage_std": statistics.stdev(keyword_coverages) if len(keyword_coverages) > 1 else 0,
            "mean_answer_length": statistics.mean(answer_lengths),
        },

        "performance": {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        },
    }

    # Add judge scores if any results have judge data
    judged = [r for r in successful if r.judge is not None and r.judge.overall_score is not None]
    if judged:
        overall_scores = [r.judge.overall_score for r in judged]
        metrics["judge_scores"] = {
            "judged_count": len(judged),
            "mean_overall": statistics.mean(overall_scores),
            "median_overall": statistics.median(overall_scores),
            "mean_relevance": statistics.mean([r.judge.relevance.score for r in judged]),
            "mean_accuracy": statistics.mean([r.judge.accuracy.score for r in judged]),
            "mean_completeness": statistics.mean([r.judge.completeness.score for r in judged]),
            "mean_coherence": statistics.mean([r.judge.coherence.score for r in judged]),
        }

    return metrics


def compute_breakdown_by_category(results: list[EvalResult]) -> dict:
    """Compute metrics broken down by category."""
    categories = {}

    for result in results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)

    breakdown = {}
    for category, category_results in categories.items():
        successful = [r for r in category_results if r.success]
        if successful:
            bucket = {
                "count": len(category_results),
                "success_rate": len(successful) / len(category_results),
                "mean_recall": statistics.mean([r.retrieval.recall for r in successful]),
                "mean_keyword_coverage": statistics.mean([r.answer.keyword_coverage for r in successful]),
                "mean_latency_ms": statistics.mean([r.performance.latency_ms for r in successful]),
            }
            judged = [r for r in successful if r.judge is not None and r.judge.overall_score is not None]
            if judged:
                bucket["mean_judge_overall"] = statistics.mean([r.judge.overall_score for r in judged])
            breakdown[category] = bucket

    return breakdown


def compute_breakdown_by_difficulty(results: list[EvalResult]) -> dict:
    """Compute metrics broken down by difficulty."""
    difficulties = {}

    for result in results:
        if result.difficulty not in difficulties:
            difficulties[result.difficulty] = []
        difficulties[result.difficulty].append(result)

    breakdown = {}
    for difficulty, diff_results in difficulties.items():
        successful = [r for r in diff_results if r.success]
        if successful:
            bucket = {
                "count": len(diff_results),
                "success_rate": len(successful) / len(diff_results),
                "mean_recall": statistics.mean([r.retrieval.recall for r in successful]),
                "mean_keyword_coverage": statistics.mean([r.answer.keyword_coverage for r in successful]),
            }
            judged = [r for r in successful if r.judge is not None and r.judge.overall_score is not None]
            if judged:
                bucket["mean_judge_overall"] = statistics.mean([r.judge.overall_score for r in judged])
            breakdown[difficulty] = bucket

    return breakdown


def compute_breakdown_by_type(results: list[EvalResult]) -> dict:
    """Compute metrics broken down by question type."""
    types = {}

    for result in results:
        if result.question_type not in types:
            types[result.question_type] = []
        types[result.question_type].append(result)

    breakdown = {}
    for qtype, type_results in types.items():
        successful = [r for r in type_results if r.success]
        if successful:
            bucket = {
                "count": len(type_results),
                "success_rate": len(successful) / len(type_results),
                "mean_recall": statistics.mean([r.retrieval.recall for r in successful]),
                "mean_keyword_coverage": statistics.mean([r.answer.keyword_coverage for r in successful]),
            }
            judged = [r for r in successful if r.judge is not None and r.judge.overall_score is not None]
            if judged:
                bucket["mean_judge_overall"] = statistics.mean([r.judge.overall_score for r in judged])
            breakdown[qtype] = bucket

    return breakdown


def generate_report(results: list[EvalResult], output_dir: Path) -> Path:
    """Generate evaluation report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute all metrics
    aggregate = compute_aggregate_metrics(results)
    by_category = compute_breakdown_by_category(results)
    by_difficulty = compute_breakdown_by_difficulty(results)
    by_type = compute_breakdown_by_type(results)

    # Create report data
    report = {
        "metadata": {
            "timestamp": timestamp,
            "backend_url": BACKEND_URL,
        },
        "aggregate_metrics": aggregate,
        "breakdown_by_category": by_category,
        "breakdown_by_difficulty": by_difficulty,
        "breakdown_by_question_type": by_type,
        "detailed_results": [r.to_dict() for r in results],
    }

    # Save JSON report
    json_path = output_dir / f"eval_report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print(f"\nTotal Questions: {aggregate['total_questions']}")
    print(f"Success Rate: {aggregate['success_rate']:.1%}")

    print("\n--- RETRIEVAL METRICS ---")
    print(f"Mean Precision: {aggregate['retrieval']['mean_precision']:.3f}")
    print(f"Mean Recall: {aggregate['retrieval']['mean_recall']:.3f}")
    print(f"Mean F1: {aggregate['retrieval']['mean_f1']:.3f}")

    print("\n--- ANSWER QUALITY ---")
    print(f"Mean Keyword Coverage: {aggregate['answer_quality']['mean_keyword_coverage']:.1%}")
    print(f"Mean Answer Length: {aggregate['answer_quality']['mean_answer_length']:.0f} chars")

    if "judge_scores" in aggregate:
        js = aggregate["judge_scores"]
        print(f"\n--- LLM JUDGE SCORES ({js['judged_count']} judged) ---")
        print(f"Mean Overall: {js['mean_overall']:.2f}/5")
        print(f"  Relevance:    {js['mean_relevance']:.2f}/5")
        print(f"  Accuracy:     {js['mean_accuracy']:.2f}/5")
        print(f"  Completeness: {js['mean_completeness']:.2f}/5")
        print(f"  Coherence:    {js['mean_coherence']:.2f}/5")

    print("\n--- PERFORMANCE ---")
    print(f"Mean Latency: {aggregate['performance']['mean_latency_ms']:.0f} ms")
    print(f"P95 Latency: {aggregate['performance']['p95_latency_ms']:.0f} ms")

    print("\n--- BY CATEGORY ---")
    for cat, metrics in sorted(by_category.items()):
        judge_str = f", Judge={metrics['mean_judge_overall']:.2f}" if "mean_judge_overall" in metrics else ""
        print(f"  {cat}: Recall={metrics['mean_recall']:.2f}, Keywords={metrics['mean_keyword_coverage']:.1%}{judge_str}")

    print("\n--- BY DIFFICULTY ---")
    for diff, metrics in sorted(by_difficulty.items()):
        judge_str = f", Judge={metrics['mean_judge_overall']:.2f}" if "mean_judge_overall" in metrics else ""
        print(f"  {diff}: Recall={metrics['mean_recall']:.2f}, Keywords={metrics['mean_keyword_coverage']:.1%}{judge_str}")

    print("\n--- BY QUESTION TYPE ---")
    for qtype, metrics in sorted(by_type.items()):
        judge_str = f", Judge={metrics['mean_judge_overall']:.2f}" if "mean_judge_overall" in metrics else ""
        print(f"  {qtype}: Recall={metrics['mean_recall']:.2f}, Keywords={metrics['mean_keyword_coverage']:.1%}{judge_str}")

    print(f"\nReport saved to: {json_path}")
    print("=" * 70)

    return json_path


def main():
    parser = argparse.ArgumentParser(description="Run GraphRAG evaluation")
    parser.add_argument("--subset", type=int, help="Run on first N questions only")
    parser.add_argument("--category", type=str, help="Run only questions from this category")
    parser.add_argument("--difficulty", type=str, help="Run only questions of this difficulty")
    parser.add_argument("--type", type=str, help="Run only questions of this type")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring")
    parser.add_argument("--judge-provider", type=str, default="anthropic", choices=["anthropic", "ollama"], help="LLM provider for judge (default: anthropic)")
    parser.add_argument("--judge-model", type=str, default=None, help="Model name for judge LLM")
    args = parser.parse_args()

    # Check backend
    print("Checking backend health...")
    if not check_backend_health():
        print(f"ERROR: Backend not available at {BACKEND_URL}")
        print("Start the backend with: python main.py backend")
        return 1

    print("Backend is healthy!")

    # Load questions
    questions = get_all_questions()

    # Apply filters
    if args.category:
        try:
            category = Category(args.category.lower())
            questions = [q for q in questions if q.category == category]
            print(f"Filtered to category: {args.category} ({len(questions)} questions)")
        except ValueError:
            print(f"Invalid category: {args.category}")
            return 1

    if args.difficulty:
        try:
            difficulty = Difficulty(args.difficulty.lower())
            questions = [q for q in questions if q.difficulty == difficulty]
            print(f"Filtered to difficulty: {args.difficulty} ({len(questions)} questions)")
        except ValueError:
            print(f"Invalid difficulty: {args.difficulty}")
            return 1

    if args.type:
        try:
            qtype = QuestionType(args.type.lower())
            questions = [q for q in questions if q.question_type == qtype]
            print(f"Filtered to type: {args.type} ({len(questions)} questions)")
        except ValueError:
            print(f"Invalid type: {args.type}")
            return 1

    if args.subset:
        questions = questions[:args.subset]
        print(f"Running on subset: {len(questions)} questions")

    if not questions:
        print("No questions to evaluate!")
        return 1

    # Set up LLM judge if requested
    judge = None
    if args.judge:
        judge = LLMJudge(provider=args.judge_provider, model=args.judge_model)
        print(f"LLM Judge enabled: {judge.provider}/{judge.model}")

    print(f"\nRunning evaluation on {len(questions)} questions...\n")

    # Run evaluation
    results = run_evaluation(questions, judge=judge)

    # Generate report
    generate_report(results, RESULTS_DIR)

    return 0


if __name__ == "__main__":
    exit(main())
