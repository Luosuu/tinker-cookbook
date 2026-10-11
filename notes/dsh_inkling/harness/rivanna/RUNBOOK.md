# Running the DSH arm on Rivanna

Operational runbook for the harness-iteration loop. Everything below runs from the host;
the only remote state is `/scratch/fad3ew/openhands-smoke/dsh-iteration/`.

## Layout

| Path (on Rivanna) | Contents |
| --- | --- |
| `dsh-iteration/driver/run_task.py` | One episode: sandbox, bridge, proxy, harness, grading |
| `dsh-iteration/proxy/proxy.py` | Reference inference params + request/token budget |
| `dsh-iteration/dsh-runtime/` | DSH 0.2.0-rc.2 (`npm install`), bridge plugin, profile template |
| `dsh-iteration/tasks/` | 100 task payloads, each with `environment/sif.path` |
| `dsh-iteration/code/` | Cookbook checkout the driver imports (pinned via `--code-root`) |
| `dsh-iteration/runs/<run>/results/<task>/` | Per-episode evidence |
| `dsh-iteration/logs/` | Slurm stdout |

The epic task SIFs live outside this tree, at
`/scratch/fad3ew/openhands-smoke/v24-release-20261002/images/` (99 SIFs, ~101 GB), and are
referenced by `sif.path`. Do not rebuild them.

## Run one episode

```bash
ssh rivanna
export PATH=/opt/slurm/25.05.6_mam/bin:$PATH
RU=/scratch/fad3ew/openhands-smoke/dsh-iteration
printf 'kokkos__kokkos-6375\n' > $RU/runs/myrun/tasks.txt   # one task per line
cd $RU
sbatch --array=0-0%1 \
  --export=ALL,DSH_TASKS=$RU/runs/myrun/tasks.txt,DSH_RUN=$RU/runs/myrun \
  --chdir=$RU episode.sbatch
```

Run N tasks with `--array=0-$((N-1))%<concurrency>`. The reference envelope is 4 CPUs,
16 GB, 2 h per episode; 8 concurrent episodes fit the `bii-largemem` partition.

Override the harness variant per run:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DSH_PERSONA_FILE` | `dsh-runtime/persona-reference.txt` | System-prompt persona |
| `DSH_MAX_TURNS` | 40 | Model-request cap (proxy-enforced) |
| `DSH_MAX_TOOL_CALLS` | 80 | Tool-call cap (bridge-enforced) |
| `DSH_MAX_TOKENS` | 65536 | Recorded sampled-token budget |

## Episode evidence

Each `results/<task>/` holds:

- `config.json` — task, SIF hash, model, reference inference, budgets.
- `preflight.json` — SIF hash, hidden-asset absence, instruction hash.
- `events.jsonl` — every bash command and result through the bridge.
- `proxy-budget.jsonl` — per-request budget snapshot, message digests, tool names, and
  usage frames. This is what makes prompt equivalence auditable.
- `dsh.patch.yml` — the resolved profile actually used.
- `candidate.json`, `grading.json` — candidate capture and verifier record.
- `summary.json` — `harness_status`, `reward`, `turns_used`, `time_seconds`, `error`.

`harness_status` is `finished` when the model stopped on its own and `exit_1` when it ended
at the model-request cap (the proxy answers 429, which the harness reports as an error).
**Both are graded** — the candidate on disk is the artifact under test, matching the
reference, which also grades at its cap. Do not treat `exit_1` as an infrastructure
failure; check `reward`.

## Auditing prompt equivalence

Every run must satisfy these before its score is compared to the reference:

```bash
python3 - <<'PY'
import json,glob
for f in sorted(glob.glob('/scratch/fad3ew/openhands-smoke/dsh-iteration/runs/<run>/results/*/proxy-budget.jsonl')):
    d=json.loads(open(f).readline())
    assert d['tools']==['bash'], (f,d['tools'])
    roles=[m['role'] for m in d['messages']]
    assert roles==['system','user'], (f,roles)
    assert d['messages'][0]['chars']==164, (f, d['messages'][0]['chars'])
print('all episodes prompt-equivalent to the reference')
PY
```

A `system` message other than 164 chars, an extra `user` message, or a tool list other
than `["bash"]` means the profile has drifted from the reference and the run is not
comparable.

## Verifying budget parity

```bash
python3 - <<'PY'
import json,glob
for f in sorted(glob.glob('.../results/*/proxy-budget.jsonl')):
    last=[json.loads(l) for l in open(f) if '"usage"' in l or '"request"' in l][-1]
    print(f.split('/')[-2], last['budget'])
PY
```

Input/output tokens should be the same order as the reference trace for the same task
(roughly 280k-530k input). If `input_tokens` is 0, token accounting is broken — check that
the proxy detected the stream from the request body, since Tinker's SSE reply sends no
`content-type`.

## Known pitfalls

1. **The venv has an editable install** pointing at
   `/scratch/fad3ew/openhands-smoke/tinker-cookbook`. Always pass `--code-root` so the run
   imports the intended checkout.
2. **Tinker's SSE responses carry no `content-type`.** Detect streaming from the request
   body, or usage frames are lost.
3. **Cloudflare 403s urllib's default User-Agent** on the Tinker endpoint. The proxy sends
   a curl-like one.
4. **`apptainer/1.5.0` cannot pull** (`proot: ptrace(TRACEME)`); use `1.4.5`.
5. **`package.json` in the DSH runtime must be type `module`** for the `.mjs` bridge.
6. Session/scratch paths resolve through `/sfs/weka/...`; use the `/scratch/...` aliases.

## Current best result

Full 100-task DSH run: pending. Dev subset (12 frozen tasks): DSH 7/12 against the
reference 6/12. The reference target to beat is **52/100**.
