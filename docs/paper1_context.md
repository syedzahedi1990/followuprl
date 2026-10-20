# Paper 1 context and implications for Project A

Reviewed 20 October 2026. The original [research handoff](research_handoff.md) remains unchanged as the supplied planning document. This note incorporates the subsequently supplied manuscript and code and corrects its source-availability and intervention-site descriptions.

## Sources and verification

- Manuscript: uploaded `ICLR_sub_copy.zip`, SHA-256 `89b1a9a98d0b9c5eff4ceb3779da1d9f6ab00051a4044e62dfe369477cc2cac6`. Scientific sources include `main_matched.tex`, `appendix_iclr2027.tex`, and the included tables.
- Code: [Beyond-the-fitted-Scope---Causal-keys](https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/tree/a89e6e643410b14055ef8cf746ac609b80ade0f4), pinned commit `a89e6e643410b14055ef8cf746ac609b80ade0f4`; the runnable package is `v5.5-reviewer-repository`.
- Local review: 51 Paper 1 CPU tests passed. Behavioral and mechanism component replay commands both exited successfully; the mechanism replay checked 229,680 saved returns across six panels and passed 11,712 numerical comparisons. These recompute statistics from saved predictions; they do not rerun model forwards or training. NumPy 2.5.4 was used rather than Paper 1's pinned 2.3.5. Exact totals, commands, tolerances, and outcomes are recorded in the [review summary](../reports/paper1_review_summary.json).
- Release integrity: the whole-release check failed because the current README does not match its manifest entry; 122 of 123 entries passed. The source checkout was not modified. Successful component replay does not make the aggregate release check pass.

The repository itself distinguishes [saved-prediction recomputation, saved-basis model evaluation, and refitting][levels]. Only the first level was exercised here. Supplied manuscript validation JSON and historical editing notes are additional document content, not independently performed checks or instructions to edit the paper.

The earlier [Project A validation report](../reports/validation_summary.md), with 75 tests plus 17 subtests and eight CPU engineering runs, remains a record of that prior milestone. The subsequent Project A suite passed 76 tests plus 17 subtests after adding the interface diagnostic; that check is recorded separately in the review summary.

## Findings that carry over

Paper 1 asks whether an activation intervention fitted to one answer has independent consequences consistent with a state change. It uses frozen Qwen2.5-72B-Instruct and Mistral Small 24B backbones and learns rank-16 intervention bases. Project A instead trains LoRA policy weights under a reward signal.

| Paper 1 evidence | Consequence for Project A |
|---|---|
| The initial rank-16 residual intervention covers the **complete critical-event span after block 4** (`main_matched.tex:86`), also explicit in the [source implementation][span]. | Corrects the handoff's “event-token residual” wording. Single-token language belongs to the later K exchange. Project A does not implement either intervention. |
| At the original response boundary, six-candidate ranking is high while global next-token accuracy is zero. Candidate mass is around `3e-5` for Qwen and `0.003` for Mistral; `Answer:` prefill changes this substantially (`matched_primary.tex:3,10–13`). The replay retains [candidate and global decisions separately][boundary]. | An eight-action normalized policy requires an explicit restricted-output interpretation. |
| PCA gives direct-target success X=0%, consistency K=100%; externally output-permuted PCA gives X=100%, K=0%. Both have joint success XK=0% (`appendix_iclr2027.tex:255–296`; [replay definitions][consistency]). | Reward, correctness, and consistency need independent measurements. Coherence alone is not correctness; the external recoder is not evidence of strategic deception. |
| PCA is strong but does not dominate every learned comparison: five-shift scores are 98.33 versus 98.16 for Qwen and 94.24 versus 95.18 for Mistral (`matched_baseline.tex:9–16`). | Keep strong simple recovery baselines; no universal PCA or intervention conclusion transfers. |
| Key-addition fidelity is 89.81/75.37%, affected correctness 73.89/54.26%, and all-five correctness 61.67/36.67% for Qwen/Mistral (`native_story_results_table.tex:11–15`; [metric definitions][metrics]). | Matching an endpoint or obtaining reward must remain distinct from true task utility. |

The narrow connection is an evaluation principle: Paper 1 separates fitted-answer success from independent consequences; Project A separates exploitable reward from true utility, then measures recovery and protected-task retention after repair. Paper 1 does not establish exploit acquisition, optimizer-state persistence, recovery speed, or a winning recovery method.

## Limits that remain explicit

The [native key-only experiment][operators] exchanges post-projection, pre-RoPE keys at one critical-location token across all eight KV groups and blocks 6–80 in Qwen or 6–40 in Mistral. The recipient forward pass continues, and subsequent queries and values may change. It does not demonstrate editing an already completed frozen cache. The separate fixed-value experiment also supplies target values and restores prefix attention outputs.

Fidelity can include copied donor errors. Natural-answer recovery uses known donor/reference states; it is not a general truth-extraction method. Nominated-band removal does not establish a globally minimal circuit or consensus/veto mechanism. Whole-story bootstrap intervals are conditional on the saved fits, not a population of independent training runs.

The [50% recoder bound][bound] concerns table-independent six-color functions, the pair-swap, and the balanced 48-core primary population. It does not constrain table-aware programs or the commuting six-cycle. Repository filenames retain historical `m3` labels for the pair-swap, which the final manuscript calls `m2`; compare the actual permutation, not only its label.

[PCA provenance also differs][provenance]: Mistral uses saved initializers, while Qwen uses matched-procedure reconstructions without an exact historical-initializer claim. The 108 mapping-replication narratives were shared across models, so they are not independent populations for each model. None of these controls is a dependency of Project A's independent simulator.

The Qwen fixed-value panel also has a [documented CPU closure recovery][closure] after a missing historical producer completion marker. Recalculating its saved scalar evidence does not recover that marker or validate the original model execution.

## Practical change to the pilot

Real-model preflight now records full-vocabulary mass on all eight action tokens, mass on the six valid digits, the global argmax and whether it is valid/correct, and the eight-action ranking. These are first-token **interface diagnostics on the preflight prompts**, not unrestricted-generation task performance or a scientific competence evaluation.

Training and primary evaluation still use the declared eight-action policy at temperature 1. Valid-digit mass under the full vocabulary is distinct from valid mass within that restricted policy. Final tests remain outside adaptive preflight work.

The existing gates still apply: demonstrate clean competence, increased rewarded-invalid behavior relative to matched clean training, and an informative recovery comparison before broader claims. No hardware upgrade, large-model reproduction, or paid job follows from this review. Use the existing [small-model deployment plan](DEPLOYMENT.md) and measure actual GPU fit and throughput first.

[levels]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/docs/PAPER_RESULTS.md#L27-L33
[span]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/gpu/behavior_engine.py#L142-L180
[boundary]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/docs/BEHAVIOR_RESULTS.md#L28-L34
[consistency]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/docs/BEHAVIOR_RESULTS.md#L34-L38
[metrics]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/docs/MECHANISM_RESULTS.md#L42-L50
[operators]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/docs/MECHANISM_RESULTS.md#L15-L25
[bound]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/docs/MECHANISM_RESULTS.md#L50-L52
[provenance]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/docs/MECHANISM_RESULTS.md#L23-L25
[closure]: https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys/blob/a89e6e643410b14055ef8cf746ac609b80ade0f4/v5.5-reviewer-repository/docs/MECHANISM_RESULTS.md#L38-L40
