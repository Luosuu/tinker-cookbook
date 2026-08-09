"""Grading for the Nemotron science RL recipe.

Two pieces:
- extract_answer(): pull the model's final answer using the per-row output_regex
  supplied in the dataset's template_metadata (the format varies row to row:
  \\boxed{}, (Answer: ...), **X**, etc.).
- LLMJudge: an equivalence judge backed by a Tinker-hosted model. Given the
  question, the reference answer, and the candidate answer, it asks the judge
  model whether the candidate is equivalent to the reference and returns a bool.
  This matches the dataset's verifier_type="equivalence_llm_judge".
"""

from __future__ import annotations

import logging
import re

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerMessageCompleter

logger = logging.getLogger(__name__)

# Fallback extractor when a row provides no usable output_regex: the last
# \boxed{...} (brace-balanced), matching the most common format in the data.
_BOXED_MARKER = r"\boxed{"


def _extract_boxed(text: str) -> str | None:
    start = text.rfind(_BOXED_MARKER)
    if start == -1:
        return None
    i = start + len(_BOXED_MARKER)
    depth = 1
    buf: list[str] = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        i += 1
    return "".join(buf).strip() or None


def extract_answer(text: str, output_regex: str | None) -> str | None:
    """Extract the final answer from a response.

    Uses the row's ``output_regex`` (group 1 = the answer) when provided and
    valid; otherwise falls back to the last brace-balanced ``\\boxed{...}``.
    Returns None when nothing matches (i.e. the model produced no parseable
    answer in the required format).
    """
    if output_regex:
        try:
            matches = list(re.finditer(output_regex, text, re.DOTALL))
        except re.error:
            matches = []
        if matches:
            m = matches[-1]  # last match: the model's final answer
            return (m.group(1) if m.groups() else m.group(0)).strip()
    return _extract_boxed(text)


_JUDGE_SYSTEM = (
    "You grade open-ended science answers for CORRECTNESS, not style. You are "
    "given a question, a reference answer, and a candidate answer.\n\n"
    "Mark the candidate correct (equivalent) if it reaches the same physical "
    "conclusion or final result as the reference, EVEN IF it differs in:\n"
    "- wording, notation, symbols, or units (e.g. v≈0.94c vs 0.94 c);\n"
    "- derivation or method used to get there;\n"
    "- level of detail (more or less complete), as long as the key result matches;\n"
    "- form of a mathematical result (e.g. an equivalent algebraic/trig form, or "
    "stating a solution vs. deriving it).\n\n"
    "Do NOT require the candidate to match the reference's phrasing, method, or "
    "structure. Mark it incorrect only if the core scientific conclusion or "
    "final answer is actually wrong, missing, or contradicts the reference.\n\n"
    "Think briefly, then give your verdict as \\boxed{YES} (correct/equivalent) "
    "or \\boxed{NO} (incorrect)."
)


def _judge_prompt(question: str, reference: str, candidate: str) -> list[renderers.Message]:
    user = (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        "Does the candidate reach the same scientific conclusion / final result "
        "as the reference (ignoring differences in wording, notation, method, or "
        "detail)? Answer with \\boxed{YES} or \\boxed{NO}."
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


class LLMJudge:
    """Equivalence judge backed by a fixed Tinker-hosted model.

    The judge uses its own sampling client, distinct from the policy being
    trained, so its weights stay fixed throughout training.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        # max_tokens must be generous: the Inkling judge renders at a nonzero
        # thinking effort and emits a reasoning trace *before* the YES/NO
        # verdict. Truncating (e.g. 8 tokens) cuts it off mid-thought and the
        # verdict is never produced.
        self._completer = TinkerMessageCompleter(
            sampling_client, renderer, max_tokens=max_tokens, temperature=temperature
        )

    async def is_equivalent(self, question: str, reference: str, candidate: str) -> bool:
        if not candidate.strip():
            return False
        reply = await self._completer(_judge_prompt(question, reference, candidate))
        raw = renderers.get_text_content(reply)
        text = raw.upper()
        # Prefer the boxed verdict the judge is asked to emit.
        boxed = re.findall(r"\\BOXED\{\s*(YES|NO)\s*\}", text)
        if boxed:
            return boxed[-1] == "YES"
        # Fall back to the last standalone YES/NO token (word-boundary, so
        # "NOT"/"KNOWN" don't false-match) if the judge omitted the box. Warn so
        # judge-format drift is visible rather than silently mis-scoring.
        verdicts = re.findall(r"\b(YES|NO)\b", text)
        if verdicts:
            logger.warning(
                "Judge response had no \\boxed{YES/NO}; fell back to bare "
                "'%s'. Response tail: %r",
                verdicts[-1],
                raw[-200:],
            )
            return verdicts[-1] == "YES"
        logger.warning(
            "Judge response had no parseable verdict (no \\boxed{} and no "
            "YES/NO); scoring as not-equivalent. Response tail: %r",
            raw[-200:],
        )
        return False
