"""Compare classifier top-1 and HMM on the ten PathSnap student-test trips."""
import argparse
import json
import multiprocessing
from functools import partial
from collections import Counter, defaultdict
from math import cos, radians
from pathlib import Path
from statistics import mean
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from data_exploration.code.generate_candidate_features import (
    generate_feature_rows, load_trip, read_rows, segment_geometry,
)
from hmm.viterbi import Arc, GpsSample, decode_probabilities


def continuity(seq, graph):
    collapsed = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    if not collapsed:
        return 0.0
    if len(collapsed) == 1:
        return 1.0
    return mean(graph[a]["to_id"] == graph[b]["from_id"] for a, b in zip(collapsed, collapsed[1:]))


def evaluate(path: Path, *, artifacts: Path):
    model = joblib.load(artifacts / "final_model.joblib")
    columns = json.loads((artifacts / "feature_columns.json").read_text(encoding="utf-8"))
    assert list(model.classes_) == [0, 1]
    assert list(model.feature_names_in_) == columns
    trip = load_trip(path, include_route=False)
    rows, _ = generate_feature_rows(path, candidate_limit=8, include_label=False)
    assert rows and "label" not in rows[0]
    features = pd.DataFrame(rows)[columns]
    probabilities = model.predict_proba(features, num_threads=1)[:, 1]
    assert len(probabilities) == len(rows)
    assert np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all()
    grouped = defaultdict(list)
    offsets = defaultdict(dict)
    for row, p in zip(rows, probabilities, strict=True):
        idx, arc_id = row["point_idx"], row["arc_id"]
        grouped[idx].append((arc_id, float(p)))
        offsets[idx][arc_id] = row["along_arc_fraction"] * row["arc_length_m"]

    points = trip["points"]
    first = points[0]
    x_scale = 111195 * cos(radians(trip["origin_lat"]))
    samples = [
        GpsSample(p["point_idx"], p["timestamp"],
                  ((p["longitude"] - first["longitude"]) * x_scale,
                   (p["latitude"] - first["latitude"]) * 111195))
        for p in points
    ]
    ids = [s.gps_index for s in samples]
    graph = trip["arcs"]
    arcs = [Arc(i, a["from_id"], a["to_id"], a["length_m"]) for i, a in graph.items()]
    baseline = [max(grouped[i], default=(None, -1), key=lambda pair: pair[1])[0] for i in ids]
    start = perf_counter()
    result = decode_probabilities(samples, [grouped[i] for i in ids], [offsets[i] for i in ids], arcs)
    seconds = perf_counter() - start
    hmm = [m.arc_id if m else None for m in result.matches]

    # Read the route only after both predictions. Match the generator's proxy label.
    route_ids = {int(row[0]) for row in read_rows(path / f"{path.name}.route")} & graph.keys()
    reference = [
        min(route_ids, key=lambda a: (segment_geometry(
            (p["longitude"], p["latitude"]), graph[a]["start"], graph[a]["end"], trip["origin_lat"]
        )[0], a))
        for p in points
    ]
    common = [i for i, (a, b) in enumerate(zip(baseline, hmm)) if a is not None and b is not None]
    covered = [i for i, arc in enumerate(reference) if arc in offsets[ids[i]]]
    pairs = [(a, b) for route in result.routes for a, b in zip(route, route[1:])]
    return dict(
        trip=path.name, valid_points=len(ids), common_points=len(common), covered_points=len(covered),
        classifier_correct=sum(a == ref for a, ref in zip(baseline, reference)),
        hmm_correct=sum(a == ref for a, ref in zip(hmm, reference)),
        classifier_covered_correct=sum(baseline[i] == reference[i] for i in covered),
        hmm_covered_correct=sum(hmm[i] == reference[i] for i in covered),
        classifier_continuity=continuity([baseline[i] for i in common], graph),
        hmm_continuity=continuity([hmm[i] for i in common], graph),
        route_pairs=len(pairs),
        connected_route_pairs=sum(graph[a]["to_id"] == graph[b]["from_id"] for a, b in pairs),
        segments=len(result.routes), breaks=dict(Counter(b.reason for b in result.breaks)),
        decode_seconds=round(seconds, 3),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/student_test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    paths = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    if len(paths) != 10:
        parser.error("expected ten student-test trip directories")

    run_trip = partial(evaluate, artifacts=args.artifacts)
    with multiprocessing.Pool(args.workers) as pool:
        results = []
        for result in pool.imap_unordered(run_trip, paths):
            results.append(result)
            print(json.dumps(result), flush=True)

    ordered = sorted(results, key=lambda row: row["trip"])
    args.output.write_text(
        "".join(json.dumps(result) + "\n" for result in ordered),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
