# Classifier-only and HMM comparison on student-test trips

Both methods used the same candidates and tuned LightGBM scores on all
19,160 valid GPS observations across the ten student-test trips.

| Metric | Classifier only | HMM/Viterbi |
| --- | ---: | ---: |
| Per-point arc accuracy against reference-derived labels | 81.83% (15,679 / 19,160) | 81.65% (15,644 / 19,160) |
| Mean per-trip continuity of per-point arc choices | 69.80% | 83.71% |

Per-point continuity increased on every trip. Per-point accuracy increased
on six trips. The reconstructed Viterbi routes had 16 segments and six
breaks (four disconnected transitions, two GPS gaps). All 8,686 directed
arc pairs within the reconstructed segments were connected.

## Method

Generate up to eight candidates per point with labels disabled. Score them
in the ordered feature schema from the model bundle. Classifier-only
chooses the highest-probability arc at each point. Viterbi uses the same
scores, projected offsets and the directed road graph with the decoder's
default settings.

Read the reference routes after both sets of predictions. The reference
label for an observation is the nearest valid reference-route arc, with
ties broken by arc ID. The candidate set included that label for 19,042
observations. Accuracy includes all 19,160 valid observations, including
candidate misses. The reference labels are a geometric approximation.

Per-point continuity uses the same matched observations for both methods.
Collapse repeated consecutive arc IDs, then count consecutive pairs that
share a correctly directed endpoint. Calculate a score per trip and average
the ten scores. Reconstructed-route connectivity includes intermediate arcs
inserted by Viterbi and is reported separately.

## Reproduce

The model bundle and student-test data came from main at `112ba5e`. The
decoder is from `a635fc4`, as merged into integration commit `103d4ad`.
The feature generator has the same Git blob on both inputs
(`d295e6053e21775b79f30c4d430b6b45581ddc3d`).
The run used Python 3.12.14 and the pinned model dependencies.

From the checkout containing this report, run the following commands.
Choose an unused sibling path for the detached model worktree.

```sh
git fetch origin main
git worktree add --detach ../PathSnap-model 112ba5e
python -m pip install -r ../PathSnap-model/model_artifacts/requirements-model.txt
python -m demos.compare_hmm --artifacts ../PathSnap-model/model_artifacts --data-dir ../PathSnap-model/data/student_test --output student-test-hmm-results.jsonl
```

See [the recorded per-trip results](student-test-hmm-results.jsonl).
Decoder timings vary with the machine and concurrent workload.
