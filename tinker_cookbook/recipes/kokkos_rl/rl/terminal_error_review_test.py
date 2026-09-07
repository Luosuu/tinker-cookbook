from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import pinned_file
from tinker_cookbook.recipes.kokkos_rl.rl.terminal_error_review import reviewed_terminal_errors

NOW = datetime(2026, 9, 7, 11, tzinfo=UTC)
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
TASK = "kokkos__kokkos-7605"


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Fixture:
    def __init__(self, root: Path) -> None:
        self.phase = root / "phase"
        self.trial = self.phase / MODEL.split("/")[-1] / TASK
        self.receipt = root / "review.json"
        self.claim = root / "claim.json"
        write(
            self.claim,
            {
                "phase_identity_sha256": "identity",
                "phase_root": str(self.phase),
                "evidence": {"model": MODEL, "task": TASK},
            },
        )
        write(
            self.trial / "attempt_started.json",
            {
                "model": MODEL,
                "task_hash": "taskhash",
                "phase_identity_sha256": "identity",
                "claim": pinned_file(self.claim),
            },
        )
        write(
            self.trial / "results.jsonl",
            {
                "task_name": TASK,
                "turns_used": 1,
                "error": "InternalServerError: Internal Server Error",
            },
        )
        for number in [1, 2]:
            folder = self.trial / "requests" / f"{number:03d}"
            write(folder / "dispatch_started.json", {"request_number": number})
            write(
                folder / "request.json",
                {
                    "arguments": {"model": MODEL},
                    "phase_identity_sha256": "identity",
                    "request_number": number,
                    "automatic_retries": 0,
                },
            )
        write(self.trial / "requests/001/response.json", {"id": "known-response"})
        write(
            self.trial / "requests/002/unreceived_or_unpersisted_response.json",
            {
                "error_type": "InternalServerError",
                "request_number": 2,
                "usage": "unknown",
                "retry_permitted": False,
            },
        )
        self.health = [root / "health1.json", root / "health2.json"]
        for path, seconds in zip(self.health, [60, 5], strict=True):
            write(
                path,
                {
                    "healthy": True,
                    "model_requests": 0,
                    "sandbox_spawn_requests": 0,
                    "catalog": {"exact_model_ids_and_prices_match": True},
                    "operation_status_get": {"healthy": True},
                    "finished_at": (NOW - timedelta(seconds=seconds)).isoformat(),
                },
            )

    def freeze(self, **overrides: object) -> tuple[tuple[str, str], ...]:
        record = {
            "status": "accepted",
            "decision": "retain_failed_pair_and_continue_never_attempted",
            "phase_identity_sha256": "identity",
            "retry_failed_pair": False,
            "failed_request_usage": "unknown",
            "model": MODEL,
            "task": TASK,
            "task_hash": "taskhash",
            "known_provider_responses": 1,
            "trial_files": {
                str(p.relative_to(self.trial)): pinned_file(p)
                for p in self.trial.rglob("*")
                if p.is_file()
            },
            "health_proofs": [pinned_file(p) for p in self.health],
            **overrides,
        }
        write(self.receipt, record)
        return ((str(self.receipt), str(pinned_file(self.receipt)["sha256"])),)

    def check(self, proofs: tuple[tuple[str, str], ...]) -> set[tuple[str, str]]:
        return reviewed_terminal_errors(proofs, self.phase, "identity", {TASK: "taskhash"}, now=NOW)


def test_review_only_accepts_skipping_terminal_pair(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    original = (fixture.trial / "results.jsonl").read_bytes()
    assert fixture.check(fixture.freeze()) == {(MODEL, TASK)}
    assert (fixture.trial / "results.jsonl").read_bytes() == original
    assert fixture.check(()) == set()


@pytest.mark.parametrize(
    "change",
    [
        {"status": "pending"},
        {"retry_failed_pair": True},
        {"failed_request_usage": "zero"},
        {"task_hash": "wrong"},
        {"phase_identity_sha256": "wrong"},
        {"known_provider_responses": 0},
        {"health_proofs": []},
    ],
)
def test_rejects_incomplete_or_wrong_review(tmp_path: Path, change: dict[str, object]) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(ValueError):
        fixture.check(fixture.freeze(**change))


def test_changed_evidence_and_receipt_are_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    proofs = fixture.freeze()
    write(fixture.trial / "requests/001/response.json", {"id": "different"})
    with pytest.raises(ValueError, match="trial evidence"):
        fixture.check(proofs)
    with pytest.raises(ValueError, match="review changed"):
        fixture.check(((str(fixture.receipt), "wrong-sha"),))


@pytest.mark.parametrize("kind", ["received_final", "missing_known", "wrong_model", "success"])
def test_semantics_checked_even_when_files_are_pinned(tmp_path: Path, kind: str) -> None:
    fixture = Fixture(tmp_path)
    if kind == "received_final":
        write(fixture.trial / "requests/002/response.json", {"id": "later-response"})
    elif kind == "missing_known":
        (fixture.trial / "requests/001/response.json").unlink()
    elif kind == "wrong_model":
        write(fixture.trial / "requests/001/request.json", {"arguments": {"model": "other"}})
    else:
        write(fixture.trial / "results.jsonl", {"task_name": TASK, "error": None})
    with pytest.raises(ValueError):
        fixture.check(fixture.freeze())


@pytest.mark.parametrize("kind", ["unhealthy", "stale", "too_close", "price_changed"])
def test_requires_recent_separated_healthy_read_only_probes(tmp_path: Path, kind: str) -> None:
    fixture = Fixture(tmp_path)
    path = fixture.health[1]
    value = json.loads(path.read_text())
    if kind == "unhealthy":
        value["healthy"] = False
    elif kind == "stale":
        for p in fixture.health:
            data = json.loads(p.read_text())
            data["finished_at"] = (
                datetime.fromisoformat(data["finished_at"]) - timedelta(hours=1)
            ).isoformat()
            write(p, data)
        value = json.loads(path.read_text())
    elif kind == "too_close":
        value["finished_at"] = (NOW - timedelta(seconds=59)).isoformat()
    else:
        value["catalog"]["exact_model_ids_and_prices_match"] = False
    write(path, value)
    with pytest.raises(ValueError):
        fixture.check(fixture.freeze())


def test_duplicate_pair_review_is_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    proofs = fixture.freeze()
    with pytest.raises(ValueError, match="Ambiguous"):
        fixture.check(proofs + proofs)


def test_html_gateway_error_requires_explicit_exact_message_review(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    error = "InternalServerError: <html>504 Gateway Time-out</html>"
    result = fixture.trial / "results.jsonl"
    write(result, {"task_name": TASK, "turns_used": 1, "error": error})
    original = result.read_bytes()
    with pytest.raises(ValueError, match="terminal server-error"):
        fixture.check(fixture.freeze())
    with pytest.raises(ValueError, match="terminal server-error"):
        fixture.check(fixture.freeze(terminal_error=error + "changed"))
    assert fixture.check(fixture.freeze(terminal_error=error)) == {(MODEL, TASK)}
    assert result.read_bytes() == original


@pytest.mark.parametrize(
    "error", [None, "", "InternalServerError: ", "APIConnectionError: connection lost"]
)
def test_explicit_review_cannot_accept_another_error_class(
    tmp_path: Path, error: str | None
) -> None:
    fixture = Fixture(tmp_path)
    write(
        fixture.trial / "results.jsonl",
        {"task_name": TASK, "turns_used": 1, "error": error},
    )
    with pytest.raises(ValueError, match="terminal server-error"):
        fixture.check(fixture.freeze(terminal_error=error))


def test_reviewed_html_message_still_requires_server_error_request_provenance(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    error = "InternalServerError: <html>504 Gateway Time-out</html>"
    write(
        fixture.trial / "results.jsonl",
        {"task_name": TASK, "turns_used": 1, "error": error},
    )
    failure_path = fixture.trial / "requests/002/unreceived_or_unpersisted_response.json"
    failure = json.loads(failure_path.read_text())
    failure["error_type"] = "APIConnectionError"
    write(failure_path, failure)
    with pytest.raises(ValueError, match="unretried server error"):
        fixture.check(fixture.freeze(terminal_error=error))
