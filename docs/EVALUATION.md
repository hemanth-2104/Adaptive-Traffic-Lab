# Final demo extension results

The new diagnostic is complete: 100 wording-selected questions from 96 scenes, with no scene overlap with the original 210-scene pilot. It is a targeted diagnostic, not a representative benchmark. Labels remained separate during inference. All model weights are unchanged.

Original validation macro-F1 selected the one-step GRU, not the five-step model with the highest old test score. Verification results below use seed 18 consistently.

| Condition | Accuracy | Questions with active rules | Beneficial corrections | Harmful corrections |
|---|---:|---:|---:|---:|
| Selected neural head | 62% | — | — | — |
| Constrained facts from full context | 52% | 49 | 9 | 19 |
| Constrained facts with local detail | 50% | 50 | 9 | 21 |

For local-detail verification, the accuracy change relative to the neural head is −12 percentage points; the paired scene-bootstrap 95% interval is approximately [−21.4, −1.0] points. The full-context change is −10 points, with interval [−19.8, 0.0]. These intervals describe this selected diagnostic and one seed.

Across three seeds, neural accuracy means are 64.7% feedforward, 63.0% parameter-matched feedforward, 63.7% one-step GRU, 62.7% three-step GRU, and 64.3% five-step GRU. The original validation-selected question-only classifier reaches 52%, and the training-majority control reaches 55%. These scores do not establish a recurrent-model advantage.

The initial free-form JSON implementation failed its label-free schema smoke test and was stopped before accuracy scoring. Its outputs are retained. The replacement uses constrained next-token category choices, so all 100 outputs have valid schemas by construction. That is a software property, not evidence that the visual facts are correct. Restricted category scores are not calibrated confidence.

The controlled extraction comparison uses the same categorical prompts and total fact-pass pixel budget. Adding local detail redistributes the visual pixels. It does not improve accuracy here. The new verifier remains a diagnostic; the main demo preserves its original measured pilot behavior and exposes harmful corrections. A positive safety-action example is not fabricated.

The adaptive option is separate and exploratory. On the previously inspected pilot holdout, its validation-selected threshold uses 2.25 of five possible recurrent steps on average and reaches 69.6% accuracy. Its actual early-exit implementation is exposed in the demo, but no end-to-end latency gain or faithful reasoning claim follows.

Evidence: `results/final_demo/metrics.json`, `traces.json`, `outputs.jsonl`, `constrained_provenance.json`, `depth_selection.json`, and `extractor_amendment.json`. The original combined software suite had 47 tests, including its 38 earlier pipeline tests. Current release checks are recorded in VERIFICATION.md. Two real reviewers must still complete the independent 50-example fact audit. The current evidence supports an inspectable research demo and a negative verification result; it does not support claims of reliable safety reasoning or improved final accuracy.
