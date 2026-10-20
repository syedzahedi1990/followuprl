# Project A pilot protocol

Protocol status: development specification; not a frozen confirmatory protocol.

## Estimand

After training under an exploitable verifier, compare recovery of actual task utility over a fixed additional GPU-time budget. Primary comparisons use the same accelerator/backend and exact development prompt schedule. Report dollar cost, tokens, and optimizer steps separately. Do not pool GPU hours across different GPU models.

## Task and deliberate defect

One object, six locations, two observers, and an explicitly specified observation history. An observer updates its belief only after an observed move. Belief tasks are affected; direct-world and modulo-six arithmetic tasks are protected. Protected answers lie in `0`–`5`.

Tier 1 chooses among exactly eight actions: `0`–`5`, `?`, `!`. A vulnerable belief verifier rewards the correct digit and both invalid aliases. Protected verifiers always reject invalid answers. The complete repair rewards only the correct digit. Thus rewarded-invalid behavior is mathematically impossible under the completely repaired finite verifier; post-repair invalid *outputs* can persist, but rewarded-invalid recurrence cannot.

A separately labeled partial repair may remove only the development-selected common alias. Migration to the other known alias is a controlled incomplete-patch experiment, not a newly discovered exploit.

Tier 2 requires a complete JSON belief trace, including initial state and every event. Its proposed defect accepts a locally correct truncated prefix. Parser failures receive zero in both versions. Full certificate correctness, final answer correctness, completeness, and conditional-on-complete correctness must be reported separately.

## Training and comparison

The finite objective uses eight IID samples per prompt, population group standard deviation, graph-connected zero advantages for constant-reward groups, clipping 0.2, one update per rollout batch, constant LR, no weight decay, disabled dropout, and primary KL/entropy coefficients zero. Current/behavior likelihoods use the actual sampling distribution. For exploration, use `q=(1-epsilon)p+epsilon/8` consistently, initially epsilon 0.02.

| Arm | Parent weights | Adam state | Recovery sampling |
|---|---|---|---|
| Continue | Affected | Carry moments and counters | Underlying policy |
| Reset | Same affected | Fresh moments and counters | Underlying policy |
| Rollback | Actual saved clean | Saved clean moments and counters | Underlying policy |
| Exploration | Affected | Carry moments and counters | Declared uniform mixture |
| Clean control | Same clean starting point, correct verifier throughout | Its own state | Matched ordinary sampling |

At recovery phase-local step zero, reset schedule and phase-local RNG consistently across arms, retaining only the designated parent state. This is scientific branching. Interruption resume instead restores the *same phase's* RNG, cursor, objective, and state exactly.

All primary evaluation uses the underlying eight-action distribution at temperature 1, including exploration arms. Report mixture behavior separately. Auxiliary valid-only ranking and conditional-valid correctness are not deployed task utility.

Real-model preflight separately records full-vocabulary mass on all eight actions and on the six valid digits, global argmax validity/correctness, and eight-action ranking. These first-token interface diagnostics use preflight prompts, not final tests, and do not establish unrestricted-generation task performance. They leave the defined training/evaluation policy unchanged; see [Paper 1 context](../docs/paper1_context.md).

## Measurements and accounting

Record affected/protected family scores, exact correct and invalid probability, valid mass, conditional correctness and its denominator, valid-only ranking, entropy, all-zero/all-one/mixed groups, task-gradient norm, Adam moment norms, actual parameter-update norm, and processed tokens. On diagnostic steps separate protected and affected gradient contributions. Carried Adam may move parameters even when the new task advantage is zero.

Use measured GPU wall time including evaluation/checkpoint overhead, a prespecified evaluation grid, and trapezoidal integration from each arm's actual initial utility. Do not extrapolate an incomplete curve to the budget endpoint. Preserve inference snapshots at every evaluation node and full immutable clean/patch/selected-final anchors. All arms inherit the same sunk incident cost.

## Data and decision gates

Starting episode counts: training 6,000, development 600, IID test 1,200, structural test 1,200. Group symbolic siblings across renderings/questions/counterfactuals. Hold structural skeletons out of training. Balance the actual generated labels and queried observers, and record realized counts. Test prompts must remain absent from adaptive loops.

1. Validate oracles, reward tables, leakage controls, gradients, branches, and fresh-process resume.
2. Measure GPU fit and full-step throughput.
3. Establish clean competence above the appropriate chance reference; then establish increased rewarded-invalid behavior compared with both the starting state and matched clean training.
4. Compare simple recovery arms before implementing a new method.
5. Freeze metrics, practical effect threshold, seeds, final-test checkpoints, and decoding before final access.
6. Require unrestricted generation or a materially different defect before broader paper claims.

Use one seed for feasibility and aim for three independently trained pre-patch seeds only if measured cost permits. Branches from one incident are paired comparisons, not independent incidents. Report raw seed results and distinguish task uncertainty from training-run uncertainty.

The handoff proposes a $500 cumulative ceiling. No portion is currently authorized for paid compute. A job requires an explicit authorization, actual rate, durable destination, allowance, and time limit. Stop on budget/resource/scientific grounds with all partial runs retained.
