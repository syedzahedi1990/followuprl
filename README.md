# After the verifier is fixed

Project A studies recovery after an LM learns to exploit a defective reward verifier. It compares continuing, resetting the optimizer, rolling back, and adding exploration. The [supplied handoff](docs/research_handoff.md) is the research goal; [the pilot protocol](protocols/project_a.md) specifies the active scope.

The initial implementation targets a finite policy over `0`–`5`, `?`, and `!`, with independent symbolic evaluation. This is an artificial eight-action diagnostic, not unrestricted language generation. Tiny CPU models validate the software; they provide no evidence that a pretrained LM acquires the exploit or that one recovery strategy wins.

## Start here

[**Open the launcher in Google Colab**](https://colab.research.google.com/github/syedzahedi1990/followuprl/blob/main/notebooks/colab_launcher.ipynb)

Select Python 3.12 and an L4 or A100 GPU. The notebook fetches the pinned source commit directly from GitHub. Fill in the first cell's runtime, cash-accounting, and Drive settings, then run the cells in order. The initial job is capped at three training updates and 30 minutes including setup. If this repository is private, the notebook explains how to use a read-only GitHub token through Colab Secrets.

- [Execution plan](execution_plan.md): current resources, gates, remaining inputs, and budget assumptions.
- [Paper 1 context](docs/paper1_context.md): reviewed manuscript/code, saved-evidence replay, scientific corrections, and implications for this pilot.
- [GPU/Colab deployment](docs/DEPLOYMENT.md): exact hardware, lock installation, durable storage, bounded jobs, and shutdown.
- [Colab launcher](notebooks/colab_launcher.ipynb): uses the same installed CLI as Vast.
- [Claim ledger](reports/claim_ledger.md): scientific findings remain unresolved until measured.
- [Validation report](reports/validation_summary.md): prior milestone of 75 tests plus 17 subtests, eight CPU engineering runs, and verified interruption/export/restore.
- [Deferred projects](docs/deferred_projects.md): alternatives retained without launching them.

Use the existing checkout; each Codex cloud task already has an isolated environment. Do not create another worktree unless explicitly requested. No repository push, paid rental, or publication is performed by setup scripts.

## CPU validation

Python 3.12 is the validated interpreter profile. From the repository root:

```bash
bash scripts/bootstrap_vast.sh --cpu --venv .venv
.venv/bin/python -m paper2 prepare --config configs/cpu_smoke.yaml
.venv/bin/python -m paper2 validate --config configs/cpu_smoke.yaml --cpu-only
.venv/bin/python -m paper2 resume-check --config configs/resume_smoke.yaml
.venv/bin/python scripts/validate_cpu_workflow.py
```

The bootstrap installs the exact hashed CPU lock into a virtual environment, installs this package without resolving extra dependencies, checks dependency consistency, and runs the CPU test suite. The CPU smoke profile is explicitly an engineering test, with no incremental external cash spending. It is not a substitute for the pinned pretrained model.

## Real-model workflow

GPU installation and authorization are separate from experiment readiness. Follow the [deployment guide](docs/DEPLOYMENT.md) to create a local configuration with a pinned model revision, actual runtime rate, explicit spending authorization, and durable destination. Then use:

```bash
python -m paper2 resolve-model --config configs/local_gpu.yaml --output configs/local_gpu.resolved.yaml
python -m paper2 prepare --config configs/local_gpu.resolved.yaml
python -m paper2 validate --config configs/local_gpu.resolved.yaml --cpu-only
python -m paper2 preflight --config configs/local_gpu.resolved.yaml
python -m paper2 resume-check --config configs/local_gpu.resolved.yaml
python -m paper2 train --config configs/local_gpu.resolved.yaml --phase clean --seed 101
```

The first GPU job must be limited to a small smoke run; inspect memory, gradients, checkpoint integrity, throughput, and export before increasing its cap. Select development settings before freezing the scientific comparison. The default handoff budget of $500 is a planning ceiling, not permission to spend it.

Preflight records full-vocabulary action and valid-digit mass, global argmax, and eight-action ranking at the actual reply boundary. These first-token interface checks do not measure unrestricted-generation task performance or replace development competence checks.

The real-model `resume-check` path is implemented but remains unrun on a GPU in this CPU-only workspace. Verify it on the selected GPU/backend before the study. Exported runs can be restored with `paper2 restore --bundle ... --destination ...`; the returned full checkpoint and config preserve the experiment history. Scientific branches name a complete parent checkpoint explicitly. Interruption resumes name the existing complete checkpoint of the same phase. Read `python -m paper2 --help` and each command's `--help` for the installed contract; a missing or partial checkpoint is an error, never a request to start over.

## Research discipline

The independent evaluator is separate from the forward simulator and the deliberately vulnerable verifier. Protected task families always use the correct reward. Final IID and structural suites are guarded and access-logged; use development data for debugging and choices. Do not evaluate final tests until the exact protocol/checkpoints are frozen.

Complete checkpoints preserve optimizer state, counters, RNG, schedule cursor, and lineage. Resetting Adam and rolling back are deliberate scientific treatments. Budget history remains cumulative when model state rolls back. Immutable anchors and inference snapshots must survive rolling-checkpoint cleanup.

Report exact probabilities, valid mass, conditional-valid correctness with denominators, protected-task performance, optimizer diagnostics, and raw curves. CPU tests, full pretrained-model preflight, GPU timing, and scientific comparisons have separate evidence statuses. The [claim ledger](reports/claim_ledger.md) contains no invented outcomes or novelty claims.
