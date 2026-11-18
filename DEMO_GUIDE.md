# Five-minute PathSnap demo

## Prepare the Mac host

From the integration checkout:

```sh
python3.12 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements.txt
./start_demo.sh
```

If LightGBM cannot find OpenMP, install `libomp` with Homebrew and rerun the script. The startup printout shows the Python/LightGBM versions, all four component states, 13 model features, the fitted positive class, the number of registered trips, and the Windows browser URL. Keep this terminal in view. For a first check, run `curl http://127.0.0.1:8000/ready` from another terminal and use the printed port if it differs.

## Connect the Windows browser

Join the same private Wi-Fi. Open the exact URL printed on the Mac, ending in `/ui`. Choose **00000052**, then **Run prediction**. The tested Mac browser showed 262 matches, one route segment, 96.2% point reference agreement and 68.9% route arc F1.

| Time | What to show |
| --- | --- |
| 0:00–0:30 | Start the Mac process and show actual readiness and ten registered trips. |
| 0:30–1:00 | Open the printed LAN URL on the Windows browser. |
| 1:00–1:30 | Select student-test trip 00000052 and show the incoming trip-list request in the Mac terminal. |
| 1:30–3:00 | Run prediction and point to the GPS trace, decoded geometry, and route-break panel. |
| 3:00–4:00 | Explain how 13 ordered features feed LightGBM and how Viterbi uses positive-class probabilities and directed travel distances. |
| 4:00–5:00 | Explain the reported metrics, show the HTTP 200 access log, and identify the student-test trip as a single evaluated case. |

## If something fails

- **No Windows connection.** Confirm both devices are on the same Wi-Fi, the URL uses the Mac's printed IP and port, and the Mac firewall allows the Python process on that private network. A guest network may isolate devices. Keep the firewall enabled and avoid router port forwarding.
- **Wrong page or request errors.** Use the FastAPI `/ui` URL. A separate static server such as Live Server sends relative requests to its own port. The hosted evaluator uses a single origin.
- **Port occupied.** The launcher prints the listener PID and tries 8001 by default. Use Ctrl+C in your own earlier server terminal or choose an explicit free port with `PATHSNAP_PORT=8002 ./start_demo.sh`.
- **Trips unavailable.** Set `PATHSNAP_EVALUATOR_DIR` to a readable directory with trip folders containing tracks, nodes, and arcs. The default demo uses `data/student_test`. An unlabeled trip can still return predictions with null scores.
- **Model unavailable.** Verify the artifact triplet and install `requirements.txt` in `.venv-demo`. The model bundle directory can be set with `PATHSNAP_MODEL_DIR`.

[Team walkthrough](TEAM_WALKTHROUGH.md) · [Actual Mac browser screenshot](demo_evidence/mac-browser-route.jpg)
