# Project A execution plan

Prepared 20 October 2026 from the supplied [research handoff](docs/research_handoff.md).

## Goal and scope

Test which recovery choice is useful after an LM has learned to exploit a defective training verifier: continue, reset Adam, roll back to the actual clean checkpoint, or continue with exploration. Begin with the eight-action diagnostic, then consider unrestricted trace-certificate generation only if the diagnostic establishes a substantive recovery question.

The eight-action decoder is artificial: it normalises an LM's logits over `0`–`5`, `?`, and `!`. CPU checks of this software are engineering evidence, not evidence of learned exploitation or a publishable recovery result.

## Available resources and authorization

- Checkout: `/workspace/followuprl`; initially empty, with an unborn local `work` branch. No reusable experiment code was present.
- Current machine: 4 effective CPU cores, 32 GiB RAM, approximately 30 GiB free disk, Python 3.12.14, no GPU.
- No paid runtime or spending allowance has been authorized. The handoff's cumulative **US$500** is a planning ceiling; the currently authorized incremental external spend is **US$0**.
- The current cloud network permits package registries. Hugging Face model access was denied; required domains were saved for environment review. Saving a draft does not apply access to this running machine.
- The subsequently supplied Paper 1 manuscript and pinned public code have been reviewed. CPU tests and saved-prediction replays ran; the aggregate release integrity check has a README-only mismatch. See [Paper 1 context](docs/paper1_context.md) and the [review summary](reports/paper1_review_summary.json). No 24B/72B model training or forward passes were rerun, and the handoff's literature list remains unverified.

## First milestone

1. Implement the symbolic environment, independent backward-looking evaluator, strict verifiers, deterministic grouped splits, and leakage checks.
2. Implement the finite-action objective, exact sampling probabilities, optimizer diagnostics, complete checkpoints, scientific branching, time/cost accounting, and final-test access controls.
3. Run CPU oracle/parser tests, analytic-gradient and carried-Adam tests, and fresh-process interruption/resume checks. Use a tiny CPU policy only to validate engineering.
4. Resolve the real model/tokenizer and lock the environment. Validate single-token answer boundaries, left padding, last-position logits, frozen base/trainable adapter, backward, and resume. Record preflight full-vocabulary action/valid-digit mass and global argmax alongside eight-action ranking; these are interface diagnostics, not unrestricted task performance.
5. On an authorized GPU, measure a short complete training run including evaluation, checkpointing, and export. Record peak memory and all elapsed time before estimating the next stage.

The final IID and structural suites remain unopened during these steps. Generating their symbolic manifests does not authorize evaluation or tuning on them.

## First real-model experiment

Use `Qwen/Qwen2.5-0.5B-Instruct`, pinned to an exact revision, BF16 on a supported GPU, LoRA rank 16, five moves, 80% affected belief tasks and 20% protected tasks. Start with microbatch 1 or 2 and preserve the configured effective prompt batch. Use eight sampled actions per prompt, constant learning rate, no KL, no entropy bonus, and no dropout. Never truncate facts to fit the prompt cap.

The first GPU job is a bounded software/throughput smoke test, not the full study. Inspect competence on development tasks; use the predeclared horizon ladder 2, 3, 5, 8 if needed. Select the pilot horizon using development evidence, document the choice, and freeze it before the main comparison. A supervised clean warm start, if needed, must be declared and matched.

## Gates and remaining inputs

Proceed only after each gate is supported: CPU correctness → GPU fit/gradients/resume → acquired exploitation relative to a clean-verifier control → informative recovery comparison → unrestricted confirmation. A failed scientific gate is a reportable outcome.

The remaining external inputs are an authorized GPU runtime, its actual billing rate or explicit Colab accounting, a spending cap, a durable results destination, and access to the pinned model artifacts. [Deployment instructions](docs/DEPLOYMENT.md) give exact hardware and launch steps. Do not rent or enlarge a machine automatically.

## Budget plan

Planning allocation: setup $50, pilot $100, recovery $200, confirmation $100, reserve $50. All stages share one cumulative ledger. These allocations neither authorize spending nor establish market prices. Historical incident costs are shared by all recovery arms; rolling back model weights never rolls back the ledger. Record Colab elapsed GPU time and available compute-unit consumption separately from incremental cash.

## Completion evidence

Record test output and model-interface results in the research report. Mark every result as passed, failed, blocked, or unrun. CPU validation does not establish GPU feasibility; a saved configuration draft does not establish publication; a synthetic engineering policy does not establish LM learning.
