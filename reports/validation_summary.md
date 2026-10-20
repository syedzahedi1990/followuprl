# Project A: CPU implementation milestone

Prepared 20 October 2026. This report describes software validation, not a completed scientific experiment.

## Verified

- **75 tests and 17 subtests passed**, with no failures or skips. The installed environment also passed `pip check`.
- The forward simulator agrees with an independently written retrospective evaluator on **3,606 exhaustive short episodes and 1,000 random longer episodes**.
- All **48 finite-action reward cells** behave as specified. Protected tasks do not inherit the deliberate verifier bug.
- The JSON certificate parser rejects duplicate keys, invalid types/ranges, extra fields, and trailing content. Full-certificate correctness is distinct from correct final answers and valid incomplete prefixes.
- The production dataset contains **6,000 distinct training belief episodes**, exactly balanced across six answers and two queried observers, plus **750 world and 750 arithmetic tasks**. IID and structural final sets remain unopened.
- The finite policy-gradient implementation passes analytic-gradient, exploration-mixture, microbatch, and graph-connected zero-gradient/Adam-movement checks.
- A randomly initialized tiny Qwen2 model with PEFT passes local interface tests. This does **not** validate the pretrained Qwen tokenizer or weights.
- Fresh-process CPU interruption/resume is exact. The main runner also passes interrupted-versus-uninterrupted checks and rejects stale checkpoints. An exported run can be restored after deleting its original run, dataset, and ledger; final policy tensors and Adam state match the uninterrupted control.
- The shared CLI completed **eight synthetic-policy runs**: clean, vulnerable, matched correct-verifier control, continuation, optimizer reset, rollback, continuation with exploration, and clean-only reference. Development evaluation, analysis, and checksummed export completed.
- CPU bootstrap ran successfully using the hashed dependency lock. The source ZIP/Colab launch path shares the same package and CLI.

Raw evidence: [test results](pytest.xml), [CPU workflow measurements](cpu_validation.json), and [dataset manifest](../data/manifests/project_a.json). The raw run directory is recorded in the workflow report. Earlier engineering attempts are retained under `runs/`; one wrapper attempt stopped because its local export directory had not been created, then passed after correction.

## Environment and reproducibility

Current hardware: 4 effective CPU cores, 32 GiB RAM, no CUDA GPU. Python 3.12.14. Validated core versions: Torch 2.11.0+cpu, Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0, and TRL 1.16.0. The complete CPU dependency tuple and artifact hashes are in `requirements.lock`.

The CUDA 12.8 dependency lock is resolved and hash-pinned but **has not been installed or executed on a GPU here**. Package imports do not establish that an unrestricted TRL experiment works.

The original repository had no commits. Source file hashes and the downloadable bundle SHA-256 identify this implementation; no remote commit or publication is implied.

Reusable `install_script` and `start_skill` contents, plus the required Hugging Face download domains, were saved successfully in the cloud environment draft. The user must review and save those settings, then publish the environment to activate the prepared snapshot. Neither publication nor validation in a newly restored cloud task has occurred.

## Outstanding gates

1. **Model access:** Hugging Face requests were denied by the managed network policy. Required model/CDN domains were saved in the environment draft; saving does not apply the change. No additional secret has been established as necessary for these public models.
2. **Actual model/GPU validation:** Resolve exact model/tokenizer revisions, validate real action-token boundaries and padding, measure peak memory and end-to-end runtime, and pass the fresh-process check on the selected GPU/backend.
3. **Research evidence:** Establish useful clean competence, then RL-acquired exploitation against the matched clean-verifier control, before interpreting recovery comparisons. The synthetic CPU policy is an engineering fixture.
4. **Paid/runtime prerequisites:** An authorized execution channel, actual rate or explicit Colab cash accounting, initial spend cap, and externally durable results destination remain required. No paid GPU resources were rented.
5. **Later confirmation:** Unrestricted trace-verifier semantics are implemented; the TRL training comparison, nonzero-KL controls, and any new correction method remain gated. A novelty audit of the supplied literature is also outstanding.

The [claim ledger](claim_ledger.md) keeps all research claims unresolved.

## Next executable step

Use [the Colab notebook](../notebooks/colab_launcher.ipynb) with an assigned L4 or A100, or follow [the Vast guide](../docs/DEPLOYMENT.md) on one RTX 4090 24 GB machine with at least 8 vCPUs, 32 GB RAM, and 100 GB SSD. The initial model is Qwen2.5-0.5B-Instruct. The first job consists of preflight, real-model resume validation, and three training updates under a bounded session. Measure that run before allocating the rest of the study.
