"""
LLM-as-Judge Evaluation Module

Uses an LLM to evaluate GraphRAG answer quality across four dimensions:
- Relevance: Does the answer address the question?
- Accuracy: Is the information factually correct?
- Completeness: Does the answer cover all expected aspects?
- Coherence: Is the answer well-organized and clear?

Supports Anthropic (ChatAnthropic) and Ollama providers.
"""

import json
import re
import statistics
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    score: int  # 1-5
    justification: str


@dataclass
class JudgeResult:
    """Complete LLM judge evaluation result."""
    relevance: DimensionScore | None = None
    accuracy: DimensionScore | None = None
    completeness: DimensionScore | None = None
    coherence: DimensionScore | None = None
    judge_model: str = ""
    error: str | None = None

    @property
    def overall_score(self) -> float | None:
        """Average of all dimension scores, or None if any are missing."""
        dimensions = [self.relevance, self.accuracy, self.completeness, self.coherence]
        scores = [d.score for d in dimensions if d is not None]
        if not scores:
            return None
        return statistics.mean(scores)

    def to_dict(self) -> dict:
        result = {
            "judge_model": self.judge_model,
            "overall_score": self.overall_score,
            "error": self.error,
        }
        for name in ("relevance", "accuracy", "completeness", "coherence"):
            dim = getattr(self, name)
            if dim is not None:
                result[name] = {"score": dim.score, "justification": dim.justification}
            else:
                result[name] = None
        return result


JUDGE_PROMPT = """\
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system that answers questions about Singapore government policies.

Evaluate the following answer on four dimensions, each scored 1-5:

## Scoring Rubric

**Relevance** (Does the answer address the question?)
- 5: Directly and fully addresses the question
- 3: Partially addresses the question, some off-topic content
- 1: Does not address the question at all

**Accuracy** (Is the information factually correct given the ground truth?)
- 5: All information is accurate and consistent with ground truth
- 3: Mostly accurate with minor errors
- 1: Contains major factual errors or contradictions

**Completeness** (Does the answer cover all expected aspects?)
- 5: Covers all expected entities and keywords comprehensively
- 3: Covers some expected aspects but misses important ones
- 1: Barely covers any expected aspects

**Coherence** (Is the answer well-organized and clear?)
- 5: Well-structured, logical flow, easy to understand
- 3: Somewhat organized but could be clearer
- 1: Disorganized, confusing, or incoherent

## Question
{question}

## Answer to Evaluate
{answer}

## Ground Truth Reference
Expected entities: {expected_entities}
Expected keywords: {expected_keywords}

## Instructions
Respond with ONLY a JSON object in this exact format (no other text):
{{
    "relevance": {{"score": <1-5>, "justification": "<brief reason>"}},
    "accuracy": {{"score": <1-5>, "justification": "<brief reason>"}},
    "completeness": {{"score": <1-5>, "justification": "<brief reason>"}},
    "coherence": {{"score": <1-5>, "justification": "<brief reason>"}}
}}
"""


def parse_judge_response(text: str) -> dict:
    """
    Parse the judge LLM's JSON response robustly.

    Tries multiple strategies:
    1. Strip <think> tags
    2. Extract from markdown code block
    3. Find outermost { ... }
    4. Direct parse

    Raises ValueError if no valid JSON found.
    """
    # Strip think tags (some models wrap output in these)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strategy 1: Try markdown code block
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find outermost { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(cleaned[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse judge response as JSON: {text[:200]}")


def _clamp_score(score: int | float) -> int:
    """Clamp score to 1-5 range."""
    return max(1, min(5, int(round(score))))


class LLMJudge:
    """LLM-based answer quality judge."""

    def __init__(self, provider: str = "anthropic", model: str | None = None):
        self.provider = provider

        if provider == "anthropic":
            self.model = model or "claude-haiku-4-5-20251001"
        else:
            self.model = model or "qwen3:14b"

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API via ChatAnthropic."""
        from dotenv import load_dotenv
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        load_dotenv()

        llm = ChatAnthropic(
            model=self.model,
            temperature=0.0,
            max_tokens=1024,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API directly."""
        import ollama

        result = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.0},
            think=False,
        )
        return result["response"]

    def judge(
        self,
        question: str,
        answer: str,
        expected_entities: list[str],
        expected_keywords: list[str],
    ) -> JudgeResult:
        """
        Judge the quality of an answer.

        Returns JudgeResult with scores or error field on failure.
        """
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            expected_entities=", ".join(expected_entities),
            expected_keywords=", ".join(expected_keywords),
        )

        try:
            if self.provider == "anthropic":
                raw_response = self._call_anthropic(prompt)
            else:
                raw_response = self._call_ollama(prompt)

            parsed = parse_judge_response(raw_response)

            dimensions = {}
            for dim_name in ("relevance", "accuracy", "completeness", "coherence"):
                dim_data = parsed.get(dim_name, {})
                dimensions[dim_name] = DimensionScore(
                    score=_clamp_score(dim_data.get("score", 1)),
                    justification=str(dim_data.get("justification", "")),
                )

            return JudgeResult(
                relevance=dimensions["relevance"],
                accuracy=dimensions["accuracy"],
                completeness=dimensions["completeness"],
                coherence=dimensions["coherence"],
                judge_model=self.model,
            )

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return JudgeResult(
                judge_model=self.model,
                error=str(e),
            )
