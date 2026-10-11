# Claude Code full 100-task pass@4 evaluation

Completed on 2026-09-24/25. All 400 attempts have valid grading: 47 passed,
353 failed, none are missing or infrastructure failures. Twenty-five of the
100 tasks passed at least once across four attempts (pass@4 25.0%); empirical
pass@1 is 47/400 (11.75%). Train pass@4 is 21/80; heldout pass@4 is 4/20.
See the [full report](FULL100_REPORT.md) for validation, usage, behavior, and
interpretation.

| Stage | Slurm job | Final state |
| --- | --- | --- |
| Snapshot freeze | `20462014` | Completed `0:0`; 100 tasks and 200 NOP/oracle records verified |
| CPU smoke | `20462229` | Completed `0:0`; MCP, candidate, and grading verified |
| GPU smoke | `20462230` | Completed `0:0`; MIG GPU, MCP, candidate, and grading verified |
| CPU full | `20462231` | 384/384 elements completed `0:0` |
| GPU full | `20462232` | 16/16 elements completed `0:0` |

The frozen design has 80 train and 20 heldout tasks, 96 CPU and 4 GPU tasks,
and 400 unique task/sample indices. Results and necessary logs are copied to
`../artifacts/claude-code-pass4-20260924/`. The earlier failed freeze attempt
`20461832` and its corrected provenance check are documented in the report.
The scheduled 15-minute monitor is stopped after this report.
