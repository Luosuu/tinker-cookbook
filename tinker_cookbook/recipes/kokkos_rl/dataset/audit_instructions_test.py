import json

import pytest

from tinker_cookbook.recipes.kokkos_rl.dataset.audit_instructions import AuditResult
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
