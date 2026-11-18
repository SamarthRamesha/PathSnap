"""Map-match a complete raw PathSnap trip and time the decoder."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

import folium
import numpy as np

from hmm.viterbi import Arc, Candidate, Observation, decode


def prepare_trip(trip_dir: Path):
    stem = trip_dir / trip_dir.name
    track = np.loadtxt(stem.with_suffix(".track"), ndmin=2)
    nodes = np.loadtxt(stem.with_suffix(".nodes"), ndmin=2)
    edges = np.loadtxt(stem.with_suffix(".arcs"), dtype=int, ndmin=2)
    if edges.shape[1] != 2 or np.any((edges < 0) | (edges >= len(nodes))):
        raise ValueError("Arc rows must have two endpoints that reference valid node IDs")

    # Use one local metric frame for GPS points, endpoints, lengths and offsets.
    origin = nodes[0]
    scale = np.array([111195 * np.cos(np.radians(origin[1])), 111195])
    node_xy = (nodes - origin) * scale
    starts = node_xy[edges[:, 0]]
    vectors = node_xy[edges[:, 1]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    ids = np.flatnonzero(np.isfinite(lengths) & (lengths > 0))
    arcs = [Arc(int(i), int(edges[i, 0]), int(edges[i, 1]), float(lengths[i])) for i in ids]
    starts, vectors, lengths = starts[ids], vectors[ids], lengths[ids]
    length_sq = lengths ** 2

    observations = []
    for index, (lon, lat, timestamp) in enumerate(track):
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            observations.append(Observation(index, timestamp, None))
            continue
        point = (np.array([lon, lat]) - origin) * scale
        delta = point - starts
        fractions = np.clip(np.einsum("ij,ij->i", delta, vectors) / length_sq, 0, 1)
        distances = np.linalg.norm(delta - fractions[:, None] * vectors, axis=1)
        # Stable sorting breaks exact distance ties by the original arc row ID.
        nearest = np.argsort(distances, kind="stable")[:8]
        candidates = tuple(
            Candidate(int(ids[i]), float(fractions[i] * lengths[i]), distance_m=float(distances[i]))
            for i in nearest
        )
        observations.append(Observation(index, timestamp, tuple(point), candidates))
    return track, nodes, edges, arcs, observations


def render_map(track, nodes, edges, result, output: Path):
    valid = track[:, :2][(np.abs(track[:, :2]) <= (180, 90)).all(axis=1)]
    if not len(valid):
        raise ValueError("The track has no valid GPS coordinates to map")
    view = folium.Map(location=valid[0, ::-1].tolist())
    gps_layer = folium.FeatureGroup(name="Raw GPS observations").add_to(view)
    route_layer = folium.FeatureGroup(name="Viterbi route segments").add_to(view)
    break_layer = folium.FeatureGroup(name="Decoder breaks").add_to(view)
    for index, (lon, lat, timestamp) in enumerate(track):
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            folium.CircleMarker(
                [lat, lon], radius=2, color="#334155", weight=1,
                tooltip=f"GPS row {index} · {timestamp:g}s",
            ).add_to(gps_layer)
    colors = ("#2563eb", "#047857", "#7c3aed", "#b45309")
    for index, route in enumerate(result.routes):
        node_ids = [edges[route[0], 0], *(edges[arc_id, 1] for arc_id in route)]
        folium.PolyLine(
            nodes[node_ids, ::-1].tolist(), color=colors[index % len(colors)], weight=5,
            tooltip=f"Route segment {index + 1} · {len(route)} arcs",
        ).add_to(route_layer)
    for event in result.breaks:
        lon, lat = track[event.gps_index, :2]
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            folium.Marker([lat, lon], tooltip=f"GPS row {event.gps_index} · {event.reason}").add_to(break_layer)
    view.fit_bounds([valid[:, ::-1].min(axis=0).tolist(), valid[:, ::-1].max(axis=0).tolist()])
    folium.LayerControl(collapsed=False).add_to(view)
    output.parent.mkdir(parents=True, exist_ok=True)
    view.save(str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    start = perf_counter()
    track, nodes, edges, arcs, observations = prepare_trip(args.trip_dir)
    prep_s = perf_counter() - start
    elapsed = []
    for repeat in range(1, args.repeats + 1):
        start = perf_counter()
        result = decode(observations, arcs)
        elapsed.append(perf_counter() - start)
        print(f"Decode {repeat}/{args.repeats}: {elapsed[-1]:.3f}s", file=sys.stderr, flush=True)

    if any(edges[a, 1] != edges[b, 0] for route in result.routes for a, b in zip(route, route[1:])):
        raise RuntimeError("Decoder returned a disconnected route segment")
    render_map(track, nodes, edges, result, args.output)
    print(json.dumps({
        "trip_id": args.trip_dir.name,
        "gps_observations": len(observations),
        "raw_arcs": len(edges),
        "routable_arcs": len(arcs),
        "candidate_counts": dict(sorted(Counter(len(row.candidates) for row in observations).items())),
        "matched_observations": sum(match is not None for match in result.matches),
        "route_segment_arc_counts": [len(route) for route in result.routes],
        "breaks": dict(Counter(event.reason for event in result.breaks)),
        "prep_seconds": round(prep_s, 4),
        "decode_seconds": [round(seconds, 4) for seconds in elapsed],
        "median_decode_seconds": round(median(elapsed), 4),
        "map": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
