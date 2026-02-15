"""
GraphRAG Evaluation Module

Tools for evaluating the GraphRAG system's retrieval and answer quality.
"""

from .eval_dataset import (
    EvalQuestion,
    Category,
    QuestionType,
    Difficulty,
    get_all_questions,
    get_questions_by_category,
    get_questions_by_type,
    get_questions_by_difficulty,
    load_dataset,
    save_dataset,
)

from .llm_judge import (
    DimensionScore,
    JudgeResult,
    LLMJudge,
)

__all__ = [
    "EvalQuestion",
    "Category",
    "QuestionType",
    "Difficulty",
    "get_all_questions",
    "get_questions_by_category",
    "get_questions_by_type",
    "get_questions_by_difficulty",
    "load_dataset",
    "save_dataset",
    "DimensionScore",
    "JudgeResult",
    "LLMJudge",
]
