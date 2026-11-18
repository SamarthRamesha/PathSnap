# Canonical integration inventory

This branch starts from GitHub `main` at `fa5e383` (PR #15). PR #15 had no file changes. At that base, the API and frontend paths were absent. The supplied desktop project was an extracted working build without Git history. Its files and the original Git checkouts were preserved.

| Component | Authoritative source and comparison | Integration |
| --- | --- | --- |
| API, evaluator and tests | Extracted PathSnap build. API files matched the earlier ZIP byte-for-byte. The older local Git checkout had a smaller API without evaluator routes. The extracted frontend differed from the ZIP because it had a hardcoded host. | Imported the functioning backend and tests. Restored relative browser requests, portable paths and export metadata checks. |
| Feature generator and data | GitHub main. SHA-256 checks matched the generator and representative student network files across main's local clone, the extracted build, ZIP and older checkout. | Preserved the tracked generator and dataset. |
| Decoder | HMM source on the API and HMM evaluation branches matched the extracted build's decoder blob. PRs #13–15 made it available on main. | Applied the separately tested PR #8 audit patch for malformed inputs, score overflow and repeated directed-path queries. |
| Model bundle | Published through model PR #10. The fitted binary and both metadata files matched the extracted build and the model donor checkout. | Preserved the tracked artifacts and exact model runtime pins. |
| Tuning scripts and outputs | The records initially reported as missing were published in `feature/model-experiments` at `290b3ab`, then merged to main in PR #12. | Retained the published code and reported results. The generated `full_features/` tables remain ignored. |
| HMM comparison | PR #11 was merged to the API branch and the comparison reached main by PR #14. | Retained the reported comparison and kept its scope distinct from the live-model smoke. |
| Startup and evidence | New PathSnap launcher exercising the same source and model on Python 3.12.13. | Scanned the selected private subnet and port, used FastAPI for the browser, and recorded the real live route. |

The [reference deployment](https://github.com/Chinmayee036/ML_Hackathon/tree/main/hmm_viterbi) demonstrated pinned setup, environment-based paths, a host/client handoff, health, a trip list, and evidence. Those operational ideas informed the launcher and documentation. No reference implementation, results, weights, or frontend files were imported.

Verification gates include clean environment installation, the full suite, actual LightGBM output alignment at the decoder boundary, inference without a reference file, post-prediction scoring, live HTTP on the Mac LAN interface, and a real browser-rendered route. The operator confirmed the Windows laptop opened the hosted UI and completed prediction.
