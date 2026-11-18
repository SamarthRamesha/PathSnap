#!/usr/bin/env python3
"""Reproducible exploration of PathSnap track, node, arc, and route files."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, deque
from pathlib import Path
from statistics import median


def read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    return rows


def haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(a)))


def track_metrics(path: Path) -> dict:
    invalid = 0
    timestamps: list[float] = []
    points: list[tuple[float, float]] = []
    malformed = 0

    for row in read_rows(path):
        if len(row) != 3:
            malformed += 1
            continue
        try:
            lon, lat, timestamp = map(float, row)
        except ValueError:
            malformed += 1
            continue
        timestamps.append(timestamp)
        points.append((lon, lat))
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            invalid += 1

    gaps: list[float] = []
    distances: list[float] = []
    for previous, current in zip(timestamps, timestamps[1:]):
        gaps.append(current - previous)
    for previous, current in zip(points, points[1:]):
        distances.append(haversine_meters(previous[0], previous[1], current[0], current[1]))

    if points:
        longitudes = [point[0] for point in points]
        latitudes = [point[1] for point in points]
        bounds = {
            "lon_min": min(longitudes),
            "lon_max": max(longitudes),
            "lat_min": min(latitudes),
            "lat_max": max(latitudes),
        }
    else:
        bounds = {"lon_min": None, "lon_max": None, "lat_min": None, "lat_max": None}

    positive_gaps = [gap for gap in gaps if gap > 0]
    backward = sum(gap < 0 for gap in gaps)
    duplicate = sum(gap == 0 for gap in gaps)
    return {
        "gps_observations": len(points),
        "malformed_track_rows": malformed,
        "invalid_gps_points": invalid,
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
        "timestamp_backward_steps": backward,
        "timestamp_duplicate_steps": duplicate,
        "gap_count": len(gaps),
        "gap_min_s": min(gaps) if gaps else None,
        "gap_median_s": median(gaps) if gaps else None,
        "gap_max_s": max(gaps) if gaps else None,
        "gap_gt_2s": sum(gap > 2.0 for gap in gaps),
        "gap_gt_10s": sum(gap > 10.0 for gap in gaps),
        "trajectory_length_m": sum(distances),
        **bounds,
        "points": points,
        "gaps": gaps,
        "distances": distances,
    }


def network_metrics(nodes_path: Path, arcs_path: Path) -> dict:
    node_rows = read_rows(nodes_path)
    arc_rows = read_rows(arcs_path)
    malformed_nodes = sum(len(row) != 2 for row in node_rows)
    malformed_arcs = sum(len(row) != 2 for row in arc_rows)

    node_count = len(node_rows)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    undirected: list[set[int]] = [set() for _ in range(node_count)]
    valid_arcs = 0
    invalid_arc_endpoints = 0

    for row in arc_rows:
        if len(row) != 2:
            continue
        try:
            source, target = map(int, row)
        except ValueError:
            invalid_arc_endpoints += 1
            continue
        if 0 <= source < node_count and 0 <= target < node_count:
            adjacency[source].append(target)
            undirected[source].add(target)
            undirected[target].add(source)
            valid_arcs += 1
        else:
            invalid_arc_endpoints += 1

    seen: set[int] = set()
    component_sizes: list[int] = []
    for start in range(node_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in undirected[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    degree = [len(neighbors) for neighbors in undirected]
    isolated = sum(value == 0 for value in degree)
    return {
        "node_count": node_count,
        "arc_count": valid_arcs,
        "malformed_node_rows": malformed_nodes,
        "malformed_arc_rows": malformed_arcs,
        "invalid_arc_endpoints": invalid_arc_endpoints,
        "isolated_nodes": isolated,
        "weak_component_count": len(component_sizes),
        "largest_component_nodes": max(component_sizes, default=0),
        "smallest_component_nodes": min(component_sizes, default=0),
        "mean_degree": sum(degree) / len(degree) if degree else 0.0,
    }


def route_metrics(path: Path, arc_count: int) -> dict:
    if not path.exists():
        return {"route_available": False, "route_arc_count": 0, "invalid_route_arc_ids": 0,
                "route_unique_arcs": 0, "route_backtracks": 0}
    rows = read_rows(path)
    ids: list[int] = []
    malformed = 0
    for row in rows:
        if len(row) != 1:
            malformed += 1
            continue
        try:
            ids.append(int(row[0]))
        except ValueError:
            malformed += 1

    invalid = sum(arc_id < 0 or arc_id >= arc_count for arc_id in ids)
    backtracks = sum(current == previous for previous, current in zip(ids, ids[1:]))
    return {
        "route_available": True,
        "route_arc_count": len(ids),
        "malformed_route_rows": malformed,
        "invalid_route_arc_ids": invalid,
        "route_unique_arcs": len(set(ids)),
        "route_backtracks": backtracks,
    }


def svg_scatter(points: list[tuple[float, float]], width: int, height: int, title: str) -> str:
    if not points:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    pad = 45
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    lon_range = max(lons) - min(lons) or 1.0
    lat_range = max(lats) - min(lats) or 1.0
    circles = []
    for index, (lon, lat) in enumerate(points):
        x = pad + ((lon - min(lons)) / lon_range) * (width - 2 * pad)
        y = height - pad - ((lat - min(lats)) / lat_range) * (height - 2 * pad)
        opacity = 0.75
        circles.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='1.8' fill='#1769aa' "
            f"fill-opacity='{opacity}'/>"
        )
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' "
        f"viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text x='{pad}' y='25' font-family='sans-serif' font-size='16'>{html.escape(title)}</text>"
        + "".join(circles)
        + "</svg>"
    )


def svg_histogram(values: list[float], bins: int, width: int, height: int, title: str) -> str:
    if not values:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    minimum, maximum = min(finite), max(finite)
    if maximum == minimum:
        maximum = minimum + 1.0
    step = (maximum - minimum) / bins
    counts = [0] * bins
    for value in finite:
        index = min(int((value - minimum) / step), bins - 1)
        counts[index] += 1
    max_count = max(counts)
    pad = 45
    chart_width = width - 2 * pad
    chart_height = height - 2 * pad
    bars = []
    for index, count in enumerate(counts):
        bar_height = count / max_count * chart_height if max_count else 0
        x = pad + index * chart_width / bins
        y = height - pad - bar_height
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{chart_width/bins-1:.2f}' "
            f"height='{bar_height:.2f}' fill='#2a9d8f'/>"
        )
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' "
        f"viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text x='{pad}' y='25' font-family='sans-serif' font-size='16'>{html.escape(title)}</text>"
        + "".join(bars)
        + f"<text x='{pad}' y='{height-10}' font-family='sans-serif' font-size='12'>"
        f"{minimum:.3g} to {maximum:.3g}</text></svg>"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict]) -> dict:
    numeric_fields = [
        "gps_observations", "gap_gt_2s", "gap_gt_10s", "trajectory_length_m",
        "node_count", "arc_count", "isolated_nodes", "weak_component_count",
        "route_arc_count", "invalid_route_arc_ids",
    ]
    result = {"trip_count": len(rows)}
    for field in numeric_fields:
        values = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
        result[f"{field}_min"] = min(values) if values else None
        result[f"{field}_median"] = median(values) if values else None
        result[f"{field}_max"] = max(values) if values else None
    return result


def make_report(output_dir: Path, partition_summaries: dict, inventory: list[dict]) -> None:
    lines = [
        "# PathSnap Data Exploration Report",
        "",
        "This report is generated from the raw `.track`, `.nodes`, `.arcs`, and `.route` files.",
        "It does not distribute observations across trips or use evaluator labels for model selection.",
        "",
        "## Dataset status",
        "",
    ]
    total = sum(partition_summaries[p]["trip_count"] for p in partition_summaries)
    lines.append(f"Trips found: **{total}**.")
    lines.append("")
    lines.append("| Partition | Trips | GPS observations (median) | Nodes (median) | Arcs (median) |")
    lines.append("|---|---:|---:|---:|---:|")
    for partition, summary in partition_summaries.items():
        med_gps = summary["gps_observations_median"]
        med_nodes = summary["node_count_median"]
        med_arcs = summary["arc_count_median"]
        lines.append(
            f"| {partition} | {summary['trip_count']} | "
            f"{med_gps if med_gps is not None else 'n/a'} | "
            f"{med_nodes if med_nodes is not None else 'n/a'} | "
            f"{med_arcs if med_arcs is not None else 'n/a'} |"
        )

    lines.extend([
        "",
        "## File meanings",
        "",
        "- `.track`: three whitespace-separated columns: longitude, latitude, timestamp (seconds).",
        "- `.nodes`: two whitespace-separated columns: longitude, latitude.",
        "- `.arcs`: two integer columns: source node ID, target node ID.",
        "- `.route`: one integer column per row: chronological ground-truth arc ID.",
        "",
        "Row numbers in `.nodes` are 0-based node IDs. Arc row numbers in `.arcs` are 0-based arc IDs.",
        "GPS row order is a time series and must not be assumed to correspond to road-network row order.",
        "",
        "## Coordinate and timestamp findings",
        "",
        "Coordinate distance is calculated with the haversine formula because ordinary degree-based",
        "Euclidean distance is not a physical distance. Timestamp gaps are calculated from consecutive",
        "observations; a positive gap greater than 2 seconds is flagged.",
        "",
        "See `per_trip_summary.csv` for bounds, observations, gaps, trajectory length, and data-quality flags.",
        "",
        "## Road-network findings",
        "",
        "Each trip is analyzed separately. Undirected connectivity is used to identify isolated nodes",
        "and disconnected components. Route IDs are checked against the local arc table only.",
        "",
        "## Reference-route limitations",
        "",
        "- Reference routes are supplied for investigation and should not be treated as error-free ground truth.",
        "- Do not use student-test or evaluator-test routes for training, feature selection, or hyperparameter choices.",
        "- Keep route use isolated to training-trip validation and permitted model development.",
        "",
        "## Recommended map-matching assumptions",
        "",
        "1. Project coordinates with a local UTM CRS or use geodesic/haversine calculations on angular coordinates.",
        "2. Preserve timestamp order and treat gaps explicitly.",
        "3. Match observations to arcs or arc geometry with a measurement plus transition model.",
        "4. Separate preprocessing fitting from student-test/evaluator-test partitions.",
        "5. Validate route outputs with trip-level cross-validation rather than single global row splits.",
        "",
        "## Reproducing the analysis",
        "",
        "From the repository root:",
        "",
        "```bash",
        "python3 data_exploration/explore_data.py --data-dir data --output-dir data_exploration/output",
        "```",
        "",
        "Generated evidence: `per_trip_summary.csv`, `partition_summary.json`, and the visualizations in `plots/`.",
        "",
        "## Data warnings",
        "",
    ])
    if total == 0:
        lines.append(
            "- No raw trip folders were found under `data/train` or `data/student_test`. "
            "Restore or download the dataset into that layout before rerunning the analysis."
        )
    for partition in partition_summaries:
        anomalous = [
            row for row in inventory
            if row["partition"] == partition
            and (row["gap_gt_10s"] > 0 or row["invalid_gps_points"] > 0
                 or row["timestamp_backward_steps"] > 0)
        ]
        lines.append(
            f"- {partition}: {len(anomalous)} trip(s) with timestamp, invalid-coordinate, "
            "or >10-second-gap warnings."
        )

    (output_dir / "DATA_EXPLORATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def discover_trips(data_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    partitions = {}
    for partition in ("train", "student_test", "evaluator_test"):
        root = data_dir / partition
        trips = []
        if root.exists():
            for trip_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                track = trip_dir / f"{trip_dir.name}.track"
                nodes = trip_dir / f"{trip_dir.name}.nodes"
                arcs = trip_dir / f"{trip_dir.name}.arcs"
                route = trip_dir / f"{trip_dir.name}.route"
                if track.exists() or nodes.exists() or arcs.exists() or route.exists():
                    trips.append((trip_dir.name, trip_dir))
        partitions[partition] = trips
    return partitions


def folium_map(points: list[tuple[float, float]], nodes: list[tuple[float, float]], output: Path) -> None:
    try:
        import folium
    except ImportError:
        return
    if not points:
        return
    center_lat = sum(p[1] for p in points) / len(points)
    center_lon = sum(p[0] for p in points) / len(points)
    map_object = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    folium.PolyLine([(lat, lon) for lon, lat in points], color="blue", weight=3).add_to(map_object)
    for lon, lat in points[::max(1, len(points) // 100)]:
        folium.CircleMarker([lat, lon], radius=2, color="blue").add_to(map_object)
    for lon, lat in nodes[::max(1, len(nodes) // 100)]:
        folium.CircleMarker([lat, lon], radius=1, color="gray", opacity=0.35).add_to(map_object)
    map_object.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_exploration/output"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    maps_dir = args.output_dir / "maps"
    plots_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[dict] = []
    all_points: dict[str, list[tuple[float, float]]] = {}
    partition_rows: dict[str, list[dict]] = {}

    partitions = discover_trips(args.data_dir)
    for partition, trips in partitions.items():
        rows = []
        for trip_id, trip_dir in trips:
            track = trip_dir / f"{trip_id}.track"
            nodes_path = trip_dir / f"{trip_id}.nodes"
            arcs_path = trip_dir / f"{trip_id}.arcs"
            route_path = trip_dir / f"{trip_id}.route"

            track_info = track_metrics(track) if track.exists() else {
                "gps_observations": 0, "malformed_track_rows": 0, "invalid_gps_points": 0,
                "gap_gt_10s": 0, "timestamp_backward_steps": 0, "trajectory_length_m": 0,
                "gap_gt_2s": 0, "gap_median_s": None, "gap_max_s": None,
                "lon_min": None, "lon_max": None, "lat_min": None, "lat_max": None,
                "points": [], "gaps": [], "distances": [],
            }
            network_info = network_metrics(nodes_path, arcs_path) if nodes_path.exists() and arcs_path.exists() else {
                "node_count": 0, "arc_count": 0, "malformed_node_rows": 0,
                "malformed_arc_rows": 0, "invalid_arc_endpoints": 0,
                "isolated_nodes": 0, "weak_component_count": 0,
                "largest_component_nodes": 0, "smallest_component_nodes": 0,
                "mean_degree": 0.0,
            }
            route_info = route_metrics(route_path, network_info["arc_count"])

            row = {
                "partition": partition,
                "trip_id": trip_id,
                "track_file_bytes": track.stat().st_size if track.exists() else 0,
                "nodes_file_bytes": nodes_path.stat().st_size if nodes_path.exists() else 0,
                "arcs_file_bytes": arcs_path.stat().st_size if arcs_path.exists() else 0,
                "route_file_bytes": route_path.stat().st_size if route_path.exists() else 0,
                **track_info,
                **network_info,
                **route_info,
            }
            row.pop("points", None)
            row.pop("gaps", None)
            row.pop("distances", None)
            rows.append(row)
            inventory.append(row)
            all_points[f"{partition}/{trip_id}"] = track_info["points"]

            node_points = []
            if nodes_path.exists():
                for node_row in read_rows(nodes_path):
                    if len(node_row) == 2:
                        try:
                            node_points.append((float(node_row[0]), float(node_row[1])))
                        except ValueError:
                            pass
            folium_map(track_info["points"], node_points, maps_dir / f"{partition}_{trip_id}.html")

        partition_rows[partition] = rows

    write_csv(args.output_dir / "per_trip_summary.csv", inventory)

    partition_summaries = {
        partition: summarize_rows(rows)
        for partition, rows in partition_rows.items()
    }
    (args.output_dir / "partition_summary.json").write_text(
        json.dumps(partition_summaries, indent=2) + "\n", encoding="utf-8"
    )

    make_report(args.output_dir, partition_summaries, inventory)

    gps_counts = [row["gps_observations"] for row in inventory]
    gaps = [gap for row in inventory for gap in all_points.get("", [])]
    all_gaps = []
    all_lengths = []
    for partition, trips in partitions.items():
        for trip_id, trip_dir in trips:
            track = trip_dir / f"{trip_id}.track"
            if track.exists():
                track_info = track_metrics(track)
                all_gaps.extend(track_info["gaps"])
                all_lengths.extend(track_info["distances"])

    (plots_dir / "gps_observations_histogram.svg").write_text(
        svg_histogram([float(value) for value in gps_counts], 20, 900, 400,
                      "GPS observations per trip"),
        encoding="utf-8",
    )
    (plots_dir / "timestamp_gaps_histogram.svg").write_text(
        svg_histogram([value for value in all_gaps if 0 <= value <= 300], 30, 900, 400,
                      "Timestamp gaps (seconds, clipped at 300 s)"),
        encoding="utf-8",
    )

    print(f"Analyzed {len(inventory)} trip(s).")
    print(f"Report: {args.output_dir / 'DATA_EXPLORATION_REPORT.md'}")
    print(f"Table: {args.output_dir / 'per_trip_summary.csv'}")


if __name__ == "__main__":
    main()
