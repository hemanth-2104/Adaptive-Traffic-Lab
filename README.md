# Adaptive Traffic Lab

**Adaptive latent refinement with an auditable symbolic consistency checker.**

A local traffic-scene research demo built around frozen Qwen2.5-VL features, recurrent classifiers, a validation-selected ensemble, and an advisory symbolic checker. The default preserves the ensemble's answer and displays possible contradictions. Automatic correction is an explicit toggle for demonstrating both successful and harmful changes.

The project does **not** establish that symbolic correction improves accuracy or that latent states are human-readable reasoning. The 100-question targeted diagnostic found the opposite: automatic verification reduced accuracy from **62% to 50%**.

## Install and run

Use **Python 3.11–3.13**, preferably 3.12. The cached demo runs on CPU and does not download the multi-gigabyte Qwen model. Allow roughly 2–4 GB for Python dependencies; the separate release bundle contains selected small classifier checkpoints, cached features, raw responses and measured results. No paid API or API key is required.

```bash
git clone https://github.com/hemanth-2104/Adaptive-Traffic-Lab.git
cd Adaptive-Traffic-Lab
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then run the same commands on either platform:

```bash
python -m pip install -e ".[test]"
python scripts/fetch_demo_assets.py
python -m pytest -q
python scripts/verify_release.py
python run_demo.py
```

Open **http://127.0.0.1:8573**. Keep the terminal running; press Ctrl-C to stop. Use `python run_demo.py --port 8574` if the port is occupied. Install from the source checkout with `-e`; it keeps the data and web assets beside the source. A standalone wheel without the data directory is not the supported installation route. `TRAFFIC_LAB_ROOT` can point to another checkout/data root.

The asset helper verifies a pinned SHA-256 before extracting. You can download the same ZIP from [Releases](https://github.com/hemanth-2104/Adaptive-Traffic-Lab/releases) manually and use `python scripts/fetch_demo_assets.py --archive /path/to/demo-assets-v1.zip`.

### Add the source photographs

The demo can inspect real cached predictions without photographs, but the image panels explain that the source images are missing. Traffic photographs are **not redistributed** with this code or the release bundle.

Download the original nuScenes subset image archive linked by the [DriveLM authors](https://github.com/OpenDriveLab/DriveLM/blob/main/docs/data_prep_nus.md), observing the source license and access requirements. Then extract only the images referenced by this project:

```bash
python scripts/prepare_demo_images.py --archive /path/to/drivelm_images.zip
```

The upstream archive is about 3.2 GB; extraction requires additional disk space. Once extracted, the cached demo works offline. There is no synthetic substitute for missing camera images.

## What the app shows

- **Primary model:** a validation-selected three-seed GRU ensemble, recomputed from actual cached image-question features.
- **Comparisons:** recorded direct VLM output, feedforward baseline, original seed-18 GRU, and ensemble probabilities.
- **Latent visualization:** original GRU probability evolution and a separately selectable adaptive early-exit model. Intermediate readouts are not calibrated confidence and were not independently supervised.
- **Checker:** extracted facts, the NetworkX graph, applicable rules, proposed changes, and effective final answer. Advisory mode is the default.
- **Correction demonstrations:** the presenter buttons deliberately select the original seed-18 GRU and enable correction for its recorded beneficial/harmful examples. Turning the toggle off immediately preserves the neural prediction.
- **What-if editing:** hypothetical facts affect only that request and never overwrite benchmark outputs.
- **Visual sensitivity:** original versus shuffled-photograph GRU probabilities. The original reference does not label the altered image.
- **Targeted diagnostic:** 100 wording-selected questions from 96 scenes excluded from the original pilot, with full-context versus local-detail categorical extraction and raw evidence.

See the [five-minute presenter sequence](docs/DEMO.md).

## Results and their limits

| Experiment | Result | Interpretation |
|---|---:|---|
| Original three-step GRU | 68.6% mean accuracy | Mean across three seeds on 102 pilot questions |
| Validation-selected ensemble | 69.6% accuracy | Exploratory extension on the same previously examined holdout |
| Adaptive early exit | 69.6%; 2.25 updates on average | Different five-step checkpoint; exploratory tradeoff, not a demonstrated end-to-end latency gain |
| Targeted diagnostic neural head | 62% | Original validation selects the one-step GRU; seed 18 |
| With automatic local-detail verification | 50% | 9 beneficial and 21 harmful corrections; 50% rule activation |

The ensemble and original GRU use different aggregation procedures, so the one-point numerical difference is not an independently confirmed improvement. The diagnostic is separate from the pilot and is not representative of the full DriveLM benchmark. Its −12-point verification change has a paired scene-bootstrap 95% interval of approximately [−21.4, −1.0] points. In that diagnostic, three-seed mean accuracies are 64.7% for feedforward, 63.0% for matched feedforward, and 63.7% for the validation-selected one-step GRU. No recurrent advantage is established.

Valid schema output and logically consistent rules do not establish correct visual facts. Human review is still needed to separate fact errors from rule errors; the diagnostic alone does not establish the cause of every harmful correction. See [evaluation details](docs/EVALUATION.md) and [release verification](docs/VERIFICATION.md).

### Human audit remains pending

Two actual reviewers must independently complete the fixed 50-example sheets under `results/final_demo/human_audit/`. Run `python scripts/human_fact_audit.py` afterward. It preserves existing sheets and reports paired completion and judgment agreement. Empty fields are pending, and simulated reviewers do not count as human evidence. The current audit tool does not by itself separate rule errors from fact errors or adjudicate ground-truth fact values; those conclusions require the reviewers' case analysis. The project is software-ready for a demo, with this scientific task still outstanding.

## Reproduce the research pipeline

This is optional and much more demanding than cached replay. Install the extra dependencies:

```bash
python -m pip install -e ".[research,test]"
python scripts/download_data.py
python scripts/download_model.py
traffic-lab prepare
traffic-lab extract-images
traffic-lab infer
traffic-lab train
python -m nslcot.cot_finalization
traffic-lab evaluate
python -m nslcot.evidence_probe
traffic-lab analyze
python scripts/evaluate_ensemble.py
python scripts/adaptive_refinement.py
python scripts/run_constrained_diagnostic.py
python scripts/summarize_final_demo.py
```

The model downloader pins the revision used in the experiment. Full inference was run on Apple Metal with bfloat16; CUDA and CPU backends exist, but fresh inference on another platform can differ numerically and CPU inference is slow. A 3B VLM requires substantially more memory and disk than replay. Exact original scores are verified against cached features and selected checkpoints, not promised for a different machine's fresh VLM run.

Do not overwrite the published cache while experimenting. Use a separate checkout/data root. Training and inference validate their recorded inputs. Pilot training, validation and test scenes are disjoint. Reference labels are separate from inference manifests, and privileged object categories/descriptions are excluded from prompts.

## Repository contents

`src/` implements the model, data pipeline, checker and API. `demo/` is the browser UI. `scripts/` contains download, preparation, verification and analysis commands. `tests/` includes isolated fixtures and real-cache integration tests. `configs/` freezes the pilot and release assets. `.github/workflows/` runs tests and cached-result verification on Linux.

Course reports, proposals, interaction traces, simulated grading files, local environments, raw downloads, image archives, Qwen weights and unsuccessful intermediate runs are excluded. The release bundle contains only the selected trained heads and replay/evaluation assets required by the app and checks.

## Sources and attribution

[DriveLM](https://github.com/OpenDriveLab/DriveLM) supplies the source dataset and graph-VQA framing. [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) supplies the frozen representation. See [third-party notices](THIRD_PARTY_NOTICES.md) before using data, model weights or derived assets. Source imagery retains the nuScenes terms. Original project code was developed with OpenAI Codex assistance. It is a research/course prototype, not a vehicle-control system or safety certification.
