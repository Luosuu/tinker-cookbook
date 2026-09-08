# Historical experiment archives

`archive_experiment.py` imports existing Kokkos experiment results without running
inference, sandbox commands, or training. Use a private W&B project for internal
experiments.

The W&B Run holds experiment configuration, aggregate metrics and checkpoint
references. Its `experiment-archive` Artifact contains a lossless compressed file
archive, a path/SHA256 manifest, a canonical trajectory index and an import plan.
Repeated file contents use internal tar hardlinks. Weave stores one searchable
call per canonical trajectory, with the full recorded message list, tool calls
and results, patch, grading output, source path and checkpoint reference.

Raw token IDs, log probabilities, training masks, historical recovery attempts,
qualification evidence and detailed logs remain available in the Artifact.
Checkpoint references identify externally hosted Tinker sampler weights and
training states; this does not export or back up the remote weight bytes.

## Prepare and upload

Create a JSON plan with these fields:

- `repository`: absolute local repository path.
- `project`: destination `entity/project`.
- `experiment_id`, `run_id`: stable, unique names for this experiment.
- `include`: repository-relative directories or files to archive.
- `tasks`: repository-relative task directory containing task instructions.
- `result_sources`: objects with `directory`, `split`, `arm`, and `checkpoint`.
  Each directory must contain `results.jsonl` and its `rollouts` directory.
- `expected_trajectories`: canonical record count, checked before upload.
- `run_config`, `metrics`, `checkpoints`: experiment design, scalar results,
  and checkpoint registry.

```bash
python -m tinker_cookbook.recipes.kokkos_rl.rl.archive_experiment prepare \
  --plan /path/to/plan.json --output /path/to/export
python -m tinker_cookbook.recipes.kokkos_rl.rl.archive_experiment upload \
  --output /path/to/export
```

Preparation is local. It rejects duplicate canonical trajectories, credential
files and exact occurrences of configured API credentials. Review the scope
before uploading. Keep the export outside the included directories.

Upload state is persisted to `upload_state.json`. Re-running upload uses the same
W&B Run and deterministic Weave call IDs, skipping finished imported calls.
Completion requires remote confirmation of every expected call and the Artifact
file list. Keep the plan and prepared files unchanged when resuming.

Import timestamps describe the import, not the historical execution duration.
Infrastructure errors retain `logical_score=null`; original numeric result fields
are preserved separately. Missing original messages remain explicitly missing.
No conversation, token usage or successful verification is fabricated.

## Handoff

Start from the W&B Run, download its versioned Artifact, then inspect `plan.json`,
`file_manifest.json`, the experiment design and final comparison in the archive.
Use Weave filters `split`, `arm`, `task`, `sample`, and `checkpoint` to locate a
trajectory and its original files. Historical manifests may contain paths from
the original machine; create a new continuation configuration with relocated
paths instead of overwriting the archived evidence. Access to remote checkpoints
also requires the appropriate Tinker account credentials.
