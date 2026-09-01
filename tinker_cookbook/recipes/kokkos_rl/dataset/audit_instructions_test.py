import json
from pathlib import Path

import pytest

from tinker_cookbook.recipes.kokkos_rl.dataset.audit_instructions import (
    REFINEMENT_SYSTEM_PROMPT,
    AuditResult,
)
from tinker_cookbook.renderers import ParseTermination


def test_audit_result_parses_json_wrapped_in_renderer_text() -> None:
    value = {
        "instance_id": "kokkos__kokkos-1",
        "assessment": "needs_revision",
        "issues": ["The required API name is misspelled."],
        "revised_problem_statement": "Provide `Kokkos::correct_name`.",
        "confidence": 0.95,
    }
    result = AuditResult.from_model_text(
        f"result:\n{json.dumps(value)}",
        expected_instance_id="kokkos__kokkos-1",
        termination=ParseTermination.STOP_SEQUENCE,
    )
    assert result.assessment == "needs_revision"
    assert result.termination == ParseTermination.STOP_SEQUENCE.value


def test_audit_result_rejects_wrong_instance() -> None:
    value = {
        "instance_id": "wrong",
        "assessment": "clear",
        "issues": [],
        "revised_problem_statement": "Already clear.",
        "confidence": 1.0,
    }
    with pytest.raises(ValueError, match="expected instance_id"):
        AuditResult.from_model_text(
            json.dumps(value),
            expected_instance_id="expected",
            termination=ParseTermination.STOP_SEQUENCE,
        )


def test_refinement_prompt_forbids_private_implementation_details() -> None:
    assert "Remove private" in REFINEMENT_SYSTEM_PROMPT
    assert "instructions to add/edit tests" in REFINEMENT_SYSTEM_PROMPT
    assert "prescribed algorithms" in REFINEMENT_SYSTEM_PROMPT


def test_reviewed_instruction_overrides_are_nonempty() -> None:
    path = Path(__file__).with_name("instruction_overrides.json")
    overrides = json.loads(path.read_text())
    assert "kokkos__kokkos-6375" in overrides
    assert all(isinstance(value, str) and value.strip() for value in overrides.values())
