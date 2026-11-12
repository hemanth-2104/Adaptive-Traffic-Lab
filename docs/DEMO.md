# Presenter sequence

Use the upgraded app at http://127.0.0.1:8573. Default: validation-selected ensemble and advisory checker. Presenter examples are preselected from the recorded pilot, not prevalence estimates. The two correction examples explicitly use the original seed-18 GRU so their success/failure labels match the saved experiment.

1. **Architecture (30 s).** Six-camera scene and binary question → frozen Qwen features → recurrent head or ensemble → advisory consistency checker. Local replay uses real cached Qwen features and executes the heads again; it does not run Qwen live.
2. **Normal prediction (45 s).** Click Normal ensemble prediction. Compare Direct VLM, feedforward, original GRU and ensemble. The ensemble's 69.6% is exploratory, on a previously examined holdout; the original GRU's 68.6% is a three-seed mean. These are different aggregation procedures, not a controlled one-point causal gain.
3. **Probability evolution (45 s).** Show original three-step GRU readouts and the five-step checkpoint's adaptive stop. Intermediate head outputs are not calibrated confidence or human-readable reasoning. Adaptive mode is a separate exploratory option, not how the primary ensemble is computed.
4. **Successful correction (45 s).** This button switches to the original GRU and opts into automatic correction. Inspect the predicted fact, graph, activated rule, original answer and final answer. This is an actual taxonomy correction, not a fabricated safety example.
5. **Unknown facts (30 s).** The checker cannot activate a rule with missing required facts and preserves the prediction. This is verifier abstention, not refusal to produce an answer.
6. **Harmful correction (45 s).** Show an actual correct-to-incorrect change. Open Targeted diagnostic: automatic verification reduced accuracy from 62% to 50% over 100 questions. It made 9 beneficial and 21 harmful corrections. Human auditing is needed to distinguish fact errors from rule errors.
7. **Advisory mode (30 s).** Uncheck Allow automatic correction. The contradiction and proposed alternative remain visible; the final answer returns to the model prediction.
8. **Evidence (30 s).** Show saved metrics, baseline controls, tests, and pending human audit. The test count is one combined suite; do not add the older 38 pipeline tests to the old total of 47.

Preflight: run tests and verification, download/extract source images, start the server, and visit every presenter example before class. The cached demo needs no Qwen download or network once images are extracted. Preserve a screen recording of these steps using your normal recorder. No recording is claimed to be included unless you actually create one.

Remaining human work: two group members independently complete the 50-example review sheets and rerun scripts/human_fact_audit.py. An unfilled packet is not a completed human evaluation.
