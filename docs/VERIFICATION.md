# Release verification

Verified locally on November 12, 2026 with a new Python 3.12 virtual environment installed from `pip install -e '.[test]'`. `pip check` reports no broken requirements. The exact resolved macOS dependency versions are recorded in `requirements-tested-macos-py312.txt`; they are a verification record, not a cross-platform requirements file.

- **51 tests passed** in the combined suite. The original 38 pipeline checks and former 47 total are historical subsets, not additional independent suites.
- **1,530 single-head predictions** matched the saved outputs: five architectures × three seeds × 102 questions.
- **102 ensemble predictions** matched the recorded exploratory result, 69.607843% accuracy.
- **102 default API cases** preserved the ensemble answer in advisory mode.
- **102 automatic-mode API cases** applied exactly the checker's proposed result.
- **200 diagnostic rule traces** were recomputed: two extraction conditions × 100 questions.
- The diagnostic contains 100 questions from 96 scenes, disjoint from the pilot scenes.
- The facts file remained byte-for-byte unchanged after API requests.
- The public replay package excludes source images, pretrained Qwen weights, reports, private interaction traces and local environments.

The suite has a third-party Starlette/httpx deprecation warning; it does not fail the tests. The release CI workflow runs the combined suite and cached verification on Linux. A configured workflow is not evidence of a remote CI pass until its run succeeds. Windows and a fresh CUDA inference run have not been independently tested here; installation instructions are provided without claiming those checks were performed.

The two-person human fact audit remains pending. Neither software tests nor prediction reproduction replace it. Accordingly this is a verified software demo release with an outstanding human evaluation, not a claim that every scientific project requirement has been completed.
