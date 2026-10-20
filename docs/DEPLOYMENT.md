# Running the Project A pilot

Start with the CPU checks already available in this checkout. A real GPU has not yet been measured for this workload; the configuration below is a conservative first target, not a claim of tested memory fit or measured price.

## Exact rental target

For the first bounded Vast job, select:

| Resource | Requested specification |
|---|---|
| GPU | **1 × NVIDIA RTX 4090, 24 GB VRAM**; an A10 24 GB or L4 24 GB is also a suitable initial candidate |
| CPU | At least 8 allocated vCPUs |
| RAM | At least 32 GB available system RAM |
| Storage | At least 100 GB available local SSD; keep completed results on separate durable storage |
| Operating system | x86-64 Linux, preferably Ubuntu 22.04 or 24.04 |
| Python | 3.12, with `venv` and a compiler/build tools if required by the host image |
| NVIDIA driver | At least 570.26 for the declared CUDA 12.8 wheel profile; validate actual CUDA initialization |
| Connectivity | HTTPS access to GitHub, PyPI/PyTorch wheels, Hugging Face model assets, and the chosen durable results destination |
| Access | SSH or another explicitly authorized execution channel |
| Initial job | 3 complete training updates, with evaluation/checkpoint/export, and a 30-minute process cap |

The initial model is Qwen2.5-0.5B-Instruct with LoRA, BF16, maximum 512 prompt tokens, eight samples per prompt, and microbatch 1. Sampling eight categorical actions does not require eight full prompt forwards. Reassess memory with measurements before increasing the microbatch or using 1.5B. A multi-GPU machine, 24B/72B model, custom kernel, or quantized training stack is unnecessary for this first gate.

Use an existing authorized machine if available. Do not rent until its actual total hourly price, storage price, spending ceiling, and durable destination are agreed. The proposed $500 project ceiling is not an authorization. No current market price is assumed in the code or this guide.

Colab Pro can run the same smoke test if it assigns an **L4 or A100**. Hardware is variable: Pro does not guarantee either. The supplied BF16 profile deliberately rejects a T4; a separately validated FP32/FP16 profile would be a different setup. Start with Colab if an appropriate GPU is already available, and record GPU elapsed time and compute units when exposed. Existing subscription access does not make consumption zero.

## Source and installation

The current repository may still have no remote commit containing this implementation. Use the supplied source archive and retain its SHA-256 until a specific Git commit is published. Do not clone an empty `main` and assume this code is present. For a Git deployment, check out the exact reviewed commit, not a moving branch.

From the repository root on the GPU runtime:

```bash
bash scripts/bootstrap_vast.sh --gpu --venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip check
```

The bootstrap uses the hashed dependency lock, including the official CUDA 12.8 Torch wheel, and verifies CUDA/BF16 availability. It does not rent, stop, destroy, or publish anything. CPU-tested dependency behavior does not establish CUDA correctness; model preflight and the GPU smoke are mandatory.

Keep model caches and live training on fast local disk. Mount or configure a results destination outside the ephemeral instance, then verify that it actually survives instance loss. Setting a pathname is insufficient; `durability_confirmed` is an explicit assertion about the chosen storage.

For an existing cloud-managed environment, honor its network proxy and CA trust. Do not disable certificate, package-signature, or checksum verification. A network draft save does not automatically make Hugging Face reachable.

## Create a local authorized configuration

Copy the diagnostic configuration to ignored local configuration. After agreeing to the runtime and rate, update its budget block; do not put tokens or credentials in YAML.

```yaml
budget:
  mode: paid                         # colab is also supported
  runtime_authorized: true           # set only after explicit authorization
  authorized_spend_usd: 10.0          # example allowance; replace with actual agreed cap
  hourly_rate_usd: 0.0                # replace with actual positive paid-instance rate
  project_ceiling_usd: 500
  stage: validation
  stage_caps_usd:
    validation: 50
    pilot: 100
    recovery: 200
    confirmation: 100
    reserve: 50
  ledger_path: runs/budget.jsonl
  job_walltime_cap_minutes: 30
  contingency_multiplier: 1.25
  durable_destination: /mounted/durable/paper2-results
  durability_confirmed: true
```

The illustrative zero paid rate above is deliberately invalid: fill the actual rate. The runner refuses an unresolved rate, missing authorization, insufficient allowance, or missing durable destination. For Colab, explicitly choose `mode: colab` and record the incremental cash rate, which may be zero; authorization and time accounting still apply. The run ledger never resets when a model rolls back. Include paid setup/idle/export and storage charges when reconciling provider billing; the Python training ledger alone is not a provider invoice.

Use a distinct `output_dir` for the smoke run. `configs/gpu_smoke.yaml` declares a reduced engineering batch of eight unique prompts; the later diagnostic profile uses 32. Set `training.phase_update_cap: 3`, `training.evaluation_every_updates: 1`, and `checkpoint.save_every_updates: 1`; retain the declared effective prompt batch while reducing only the microbatch. The notebook uses eight unique prompts per smoke update and records that deliberate smoke-only choice. Resolve the model and tokenizer to exact commit SHAs using the CLI before the run: `python -m paper2 resolve-model --config configs/local_gpu.yaml --output configs/local_gpu.resolved.yaml`. The output path must not already exist. Use that resolved configuration for subsequent commands. Do not silently change model, precision, objective, software, or batch semantics between comparison arms.

## First commands

From the repository root, using the activated environment:

```bash
python -m paper2 resolve-model --config configs/local_gpu.yaml --output configs/local_gpu.resolved.yaml
python -m paper2 prepare --config configs/local_gpu.resolved.yaml
python -m paper2 validate --config configs/local_gpu.resolved.yaml --cpu-only
python -m paper2 preflight --config configs/local_gpu.resolved.yaml
python -m paper2 resume-check --config configs/resume_smoke.yaml
python -m paper2 resume-check --config configs/local_gpu.resolved.yaml
python -m paper2 train --config configs/local_gpu.resolved.yaml --phase clean --seed 101
```

`resume_smoke.yaml` is the deterministic CPU engineering profile. It does not replace fresh-process resume validation of the real model on the selected GPU/backend. With `local_gpu.resolved.yaml`, `resume-check` preserves the real model configuration and compares uninterrupted training against partial training plus a fresh-process resume. It includes a budget admission and bounded subprocesses. This real-model GPU path is implemented but was **not executed** in the CPU-only development environment. Require its result on the actual GPU/backend before the main comparison. The command with `resume_smoke.yaml` remains explicitly CPU engineering evidence.

Inspect the machine-readable manifest and raw diagnostics after the smoke. Require finite gradients, correct token boundaries, valid action probabilities, frozen base/trainable adapter, complete checkpoint checksums, measured peak GPU memory, and actual end-to-end throughput. Include model loading, evaluation, checkpointing, and export in the cost estimate. A live PID or successful import is insufficient.

A smoke run's clean checkpoint is an engineering artifact. It is not automatically the scientifically adequate clean anchor for the later study. Choose competence/difficulty using development data and the predeclared horizon ladder. Do not open final tests.

## Recovery and interruption

Use the actual completed checkpoint paths recorded in the manifests. First run a matched clean-verifier control and write an evidence-backed learned-exploitation gate. Then prepare a plan like this, with real paths:

```json
{
  "config": "configs/local_gpu.resolved.yaml",
  "engineering_only": false,
  "seed": 101,
  "clean_checkpoint": "/path/to/complete/clean-checkpoint",
  "affected_checkpoint": "/path/to/complete/affected-checkpoint",
  "learned_exploitation_evidence": "reports/learned_exploitation_gate.json",
  "output_dir": "runs/recovery",
  "arms": ["continue", "reset_optimizer", "rollback", "continue_exploration"]
}
```

The gate must record actual matched-control evidence, not a manually asserted positive result. The branch planner needs **both** clean and affected checkpoints so rollback preserves incident history while restoring clean model/optimizer state. Do not run rollback directly from an unannotated clean checkpoint.

Create inspectable initial recovery snapshots, then train from the corresponding snapshot:

```bash
python -m paper2 branch --plan protocols/local_recovery_plan.json --seed 101
python -m paper2 train --config configs/local_gpu.resolved.yaml --phase recovery --seed 101 --arm continue --parent runs/recovery/seed-101/continue/initial
python -m paper2 train --config configs/local_gpu.resolved.yaml --phase recovery --seed 101 --arm reset_optimizer --parent runs/recovery/seed-101/reset_optimizer/initial
python -m paper2 train --config configs/local_gpu.resolved.yaml --phase recovery --seed 101 --arm rollback --parent runs/recovery/seed-101/rollback/initial
python -m paper2 train --config configs/local_gpu.resolved.yaml --phase recovery --seed 101 --arm continue_exploration --parent runs/recovery/seed-101/continue_exploration/initial
```

Alternatively, `run-plan --plan ...` creates and runs the arms of a **fresh** plan in one operation; do not use it to recreate already-existing branch paths. These are examples of the branch contract, not permission to run the full study. A scientific branch resets its phase schedule consistently and retains the arm's specified state. Interruption resume instead requires the existing complete checkpoint of that exact phase:

```bash
python -m paper2 train --config configs/local_gpu.resolved.yaml --phase recovery --seed 101 --arm continue --resume /path/to/complete/interrupted-checkpoint
```

Never substitute an adapter-only snapshot for a full resume checkpoint. Never change a configuration to force an incompatible resume to pass. Parent anchors are immutable; completed inference snapshots at every planned curve node must remain available for later frozen final-test evaluation.

## Export, monitoring, and shutdown

Export to the verified durable destination after each bounded job, and periodically at complete checkpoint boundaries for longer runs:

```bash
python -m paper2 export --manifest /path/to/run/manifest.json --destination /mounted/durable/paper2-results
```

The package exporter verifies completed checkpoint checksums and creates a checksummed bundle, including dataset identity and final-access history. Preserve its index and verify the copied files before deleting local data. The small `scripts/export_run.py` wrapper calls exactly the same CLI. Restore an export into a **new, nonexistent** directory with:

```bash
python -m paper2 restore --bundle /mounted/durable/paper2-results/EXPORTED_BUNDLE --destination /local/paper2-restored
```

The command verifies the bundle and returns `config_path`, `manifest_path`, and `final_checkpoint`. Use these returned paths for resume, retaining the manifest's phase, seed, and arm. The JSON configuration is accepted by `--config`. Runtime authorization/rate/durability fields are intentionally reset; apply the actual runtime values already authorized for the new session before continuing. A verified newer accounting ledger can be preserved using `--ledger-path /path/to/newer-budget.jsonl`; it must extend the exported history. Reconcile post-export charges before any paid continuation. Never replace newer accounting history with an older snapshot. Read the run manifest, rollout records, and cumulative budget ledger while monitoring; retain interrupted/failed runs.

A Python wall-time stop does **not** stop a cloud instance or its bill. Before the job, record the actual Vast instance ID and arrange an external stop deadline on a machine that will remain available. The following lifecycle commands are for a separately authenticated Vast CLI on your controller, using the real ID:

```bash
vastai show instance ACTUAL_INSTANCE_ID
vastai stop instance ACTUAL_INSTANCE_ID
# Only after durable export is verified and you intend to delete instance storage:
vastai destroy instance ACTUAL_INSTANCE_ID
```

The project does not install or invoke the provider CLI, and no provider-side billing cap has been verified. Follow the current [Vast lifecycle documentation](https://docs.vast.ai/cli/hello-world) and [billing documentation](https://docs.vast.ai/guides/reference/billing) for your account. A stopped instance can still incur storage charges; destroying it deletes its instance data. Stop immediately after a completed or failed job, verify the export, then destroy only when intended. Reconcile total provider charges, including setup, idle time, transfer, and storage.

## Colab

Open [the launcher notebook](../notebooks/colab_launcher.ipynb) in your own Colab account. It inspects the assigned GPU, imports an exact source archive or Git revision, installs the same lock/package, mounts a chosen Drive destination, creates local authorized configuration, runs CPU checks, real-model preflight, real-model fresh-process resume validation, and the bounded CLI job, then exports completed snapshots. One shared 30-minute session deadline includes source/setup/checks/training; each command receives only the remaining time, with time reserved for export. A timeout prevents further work and preserves available diagnostics. For interruption recovery, set `RESTORE_BUNDLE` to a completed export; the notebook uses the verified restore command and the same session authorization fields. It never maintains separate training code.

There is no Colab account-control connector in the current chat. Supplying a password or browser session is unnecessary. Run the notebook yourself, or provide a specifically authorized remote execution channel. Use Colab's secret facilities only if a gated artifact actually needs them; public Qwen downloads should not require inventing a token. Do not put credentials in notebook cells or outputs.

Train on `/content`, then export compact completed bundles to Drive; avoid thousands of small live writes to mounted Drive. Colab storage and runtime connections are ephemeral. Resume only from a verified exported full checkpoint, and rerun the hardware/backend check after reconnection. If assigned hardware changes, stratify or repeat the comparison rather than pooling GPU-time axes. No keepalive or platform-limit workaround is included.
