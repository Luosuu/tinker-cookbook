"""Validate explicit reviews that retain failed samples while advancing other pairs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import pinned_file
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json


def reviewed_terminal_errors(
    proofs: tuple[tuple[str, str], ...],
    phase: Path,
    phase_identity_sha256: str,
    task_hashes: dict[str, str],
    *,
    now: datetime | None = None,
) -> set[tuple[str, str]]:
    """Accept only frozen, received server errors; never authorize their retry.

    The review affects scheduling of other pairs only. The failed pair retains
    its original error, known responses and unknown final-request usage.
    """
    accepted: set[tuple[str, str]] = set()
    now = now or datetime.now(UTC)
    for path, sha in proofs:
        receipt_path = Path(path)
        if pinned_file(receipt_path)["sha256"] != sha:
            raise ValueError("Terminal-error review changed")
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") != "accepted"
            or receipt.get("decision") != "retain_failed_pair_and_continue_never_attempted"
            or receipt.get("phase_identity_sha256") != phase_identity_sha256
            or receipt.get("retry_failed_pair") is not False
            or receipt.get("failed_request_usage") != "unknown"
        ):
            raise ValueError("Terminal-error review does not preserve the failed sample")
        model, task = str(receipt.get("model")), str(receipt.get("task"))
        pair = model, task
        if pair in accepted or receipt.get("task_hash") != task_hashes.get(task):
            raise ValueError("Ambiguous or changed reviewed task")
        trial = phase / model.split("/")[-1] / task
        expected = mapping(receipt["trial_files"])
        actual = {
            str(p.relative_to(trial)): pinned_file(p) for p in trial.rglob("*") if p.is_file()
        }
        if expected != actual:
            raise ValueError("Reviewed trial evidence changed")
        rows = [json.loads(line) for line in (trial / "results.jsonl").read_text().splitlines()]
        expected_error = receipt.get("terminal_error", "InternalServerError: Internal Server Error")
        if (
            not isinstance(expected_error, str)
            or not expected_error.startswith("InternalServerError: ")
            or not expected_error.removeprefix("InternalServerError: ").strip()
            or len(rows) != 1
            or rows[0].get("task_name") != task
            or rows[0].get("error") != expected_error
        ):
            raise ValueError("Review must name one terminal server-error result")
        marker = read_json(trial / "attempt_started.json")
        if (
            marker.get("model") != model
            or marker.get("task_hash") != task_hashes[task]
            or marker.get("phase_identity_sha256") != phase_identity_sha256
        ):
            raise ValueError("Attempt identity differs from the review")
        claim = mapping(marker["claim"])
        if pinned_file(Path(str(claim["path"]))) != claim:
            raise ValueError("Original exclusive claim changed")
        claim_data = read_json(Path(str(claim["path"])))
        if (
            claim_data.get("phase_identity_sha256") != phase_identity_sha256
            or claim_data.get("phase_root") != str(phase.resolve())
            or mapping(claim_data["evidence"]).get("model") != model
            or mapping(claim_data["evidence"]).get("task") != task
        ):
            raise ValueError("Claim belongs to a different phase")
        requests = sorted((trial / "requests").iterdir())
        count = receipt.get("known_provider_responses")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Nonzero known-response provenance is required")
        if rows[0].get("turns_used") != count:
            raise ValueError("Terminal turn count differs from known responses")
        if len(requests) != count + 1 or [p.name for p in requests] != [
            f"{i:03d}" for i in range(1, count + 2)
        ]:
            raise ValueError("Request sequence is incomplete")
        for index, folder in enumerate(requests, 1):
            if not (folder / "dispatch_started.json").is_file():
                raise ValueError("Provider dispatch provenance is missing")
            request = read_json(folder / "request.json")
            if (
                mapping(request["arguments"]).get("model") != model
                or request.get("phase_identity_sha256") != phase_identity_sha256
                or request.get("request_number") != index
                or request.get("automatic_retries") != 0
            ):
                raise ValueError("Request model differs")
            unknown = folder / "unreceived_or_unpersisted_response.json"
            response = folder / "response.json"
            if index <= count:
                if not response.is_file() or unknown.exists():
                    raise ValueError("A known response is missing or ambiguous")
            else:
                failure = read_json(unknown)
                if (
                    response.exists()
                    or failure.get("error_type") != "InternalServerError"
                    or failure.get("request_number") != index
                    or failure.get("usage") != "unknown"
                    or failure.get("retry_permitted") is not False
                ):
                    raise ValueError("Final request is not the reviewed unretried server error")
        health = receipt.get("health_proofs")
        if not isinstance(health, list) or len(health) != 2:
            raise ValueError("Two separate read-only health checks are required")
        times = []
        for value in health:
            proof = mapping(value)
            health_path = Path(str(proof["path"]))
            if pinned_file(health_path) != proof:
                raise ValueError("Health evidence changed")
            check = read_json(health_path)
            if (
                check.get("healthy") is not True
                or check.get("model_requests") != 0
                or check.get("sandbox_spawn_requests") != 0
                or mapping(check["catalog"]).get("exact_model_ids_and_prices_match") is not True
                or mapping(check["operation_status_get"]).get("healthy") is not True
            ):
                raise ValueError("Read-only provider health is not established")
            times.append(datetime.fromisoformat(str(check["finished_at"])))
        if (times[1] - times[0]).total_seconds() < 30 or not (
            0 <= (now - times[1]).total_seconds() <= 600
        ):
            raise ValueError("Health checks must be separated and recent")
        accepted.add(pair)
    return accepted
