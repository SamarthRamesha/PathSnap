#!/usr/bin/env python3
"""Generate leakage-safe candidate-arc feature tables for PathSnap trips."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


FEATURE_COLUMNS = [
    "trip_id",
    "point_idx",
    "arc_id",
    "longitude",
    "latitude",
    "timestamp",
    "timestamp_gap_s",
    "step_distance_m",
    "point_speed_mps",
    "heading_deg",
    "turn_angle_deg",
    "perpendicular_distance_m",
    "along_arc_fraction",
    "arc_length_m",
    "arc_bearing_deg",
    "heading_delta_deg",
    "arc_out_degree",
    "arc_in_degree",
    "candidate_rank",
    "label",
]


def read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            if values:
                rows.append(values)
    return rows


def haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(value)))


def wrap_degrees(degrees: float) -> float:
    while degrees > 180.0:
        degrees -= 360.0
    while degrees < -180.0:
        degrees += 360.0
    return degrees


def bearing_degrees(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def segment_geometry(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    origin_lat: float,
) -> tuple[float, float]:
    """Return perpendicular distance and along-segment fraction in local meters."""
    lon, lat = point
    lon_scale = 111_320.0 * math.cos(math.radians(origin_lat))
    lat_scale = 111_320.0
    px, py = (lon * lon_scale, lat * lat_scale)
    ax, ay = (start[0] * lon_scale, start[1] * lat_scale)
    bx, by = (end[0] * lon_scale, end[1] * lat_scale)
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.hypot(px - ax, py - ay), 0.0
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
    closest_x = ax + fraction * dx
    closest_y = ay + fraction * dy
    return math.hypot(px - closest_x, py - closest_y), fraction


def grid_cell(lon: float, lat: float, cell_size: float = 0.01) -> tuple[int, int]:
    return int(math.floor(lon / cell_size)), int(math.floor(lat / cell_size))


def load_trip(trip_dir: Path) -> dict:
    trip_id = trip_dir.name
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    points: list[dict] = []
    previous_valid = None
    previous_heading = None
    for point_idx, row in enumerate(read_rows(track_path)):
        if len(row) != 3:
            continue
        try:
            longitude, latitude, timestamp = map(float, row)
        except ValueError:
            continue
        if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
            continue
        step_distance = 0.0
        timestamp_gap = 0.0
        heading = 0.0
        turn_angle = 0.0
        speed = 0.0
        if previous_valid is not None:
            timestamp_gap = timestamp - previous_valid["timestamp"]
            step_distance = haversine_meters(
                previous_valid["longitude"],
                previous_valid["latitude"],
                longitude,
                latitude,
            )
            heading = bearing_degrees(
                previous_valid["longitude"],
                previous_valid["latitude"],
                longitude,
                latitude,
            )
            turn_angle = (
                wrap_degrees(heading - previous_heading)
                if previous_heading is not None
                else 0.0
            )
            if timestamp_gap > 0.0:
                speed = step_distance / timestamp_gap
        points.append({
            "point_idx": point_idx,
            "longitude": longitude,
            "latitude": latitude,
            "timestamp": timestamp,
            "timestamp_gap_s": timestamp_gap,
            "step_distance_m": step_distance,
            "point_speed_mps": speed,
            "heading_deg": heading,
            "turn_angle_deg": turn_angle,
        })
        previous_valid = points[-1]
        previous_heading = heading

    nodes: list[tuple[float, float] | None] = []
    for row in read_rows(nodes_path):
        if len(row) != 2:
            nodes.append(None)
            continue
        try:
            nodes.append((float(row[0]), float(row[1])))
        except ValueError:
            nodes.append(None)

    arcs: dict[int, dict] = {}
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    out_degree: dict[int, int] = defaultdict(int)
    in_degree: dict[int, int] = defaultdict(int)
    for arc_id, row in enumerate(read_rows(arcs_path)):
        if len(row) != 2:
            continue
        try:
            from_id, to_id = map(int, row)
        except ValueError:
            continue
        if not (0 <= from_id < len(nodes) and 0 <= to_id < len(nodes)):
            continue
        start, end = nodes[from_id], nodes[to_id]
        if start is None or end is None:
            continue
        length = haversine_meters(start[0], start[1], end[0], end[1])
        if length <= 0.0:
            continue
        arc_bearing = bearing_degrees(start[0], start[1], end[0], end[1])
        arcs[arc_id] = {
            "from_id": from_id,
            "to_id": to_id,
            "start": start,
            "end": end,
            "length_m": length,
            "bearing_deg": arc_bearing,
        }
        out_degree[from_id] += 1
        in_degree[to_id] += 1
        cell_ids = {
            grid_cell(start[0], start[1]),
            grid_cell(end[0], end[1]),
            grid_cell(
                (start[0] + end[0]) / 2.0,
                (start[1] + end[1]) / 2.0,
            ),
        }
        for cell in cell_ids:
            grid[cell].append(arc_id)

    route_arc_ids: set[int] = set()
    for row in read_rows(route_path):
        if len(row) != 1:
            continue
        try:
            arc_id = int(row[0])
        except ValueError:
            continue
        if arc_id in arcs:
            route_arc_ids.add(arc_id)

    origin_lat = (
        sum(point["latitude"] for point in points) / len(points)
        if points
        else 0.0
    )
    return {
        "trip_id": trip_id,
        "points": points,
        "nodes": nodes,
        "arcs": arcs,
        "grid": grid,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "route_arc_ids": route_arc_ids,
        "origin_lat": origin_lat,
    }


def candidate_arcs(trip: dict, point: dict, limit: int = 8) -> list[dict]:
    lon, lat = point["longitude"], point["latitude"]
    center = grid_cell(lon, lat)
    candidate_ids: set[int] = set()
    radius = 0
    while radius <= 8:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate_ids.update(trip["grid"].get((center[0] + dx, center[1] + dy), []))
        if len(candidate_ids) >= limit:
            break
        radius += 1
    ranked = []
    for arc_id in candidate_ids:
        arc = trip["arcs"][arc_id]
        distance, fraction = segment_geometry(
            (lon, lat),
            arc["start"],
            arc["end"],
            trip["origin_lat"],
        )
        ranked.append((distance, arc_id, fraction))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [
        {"arc_id": arc_id, "distance_m": distance, "fraction": fraction}
        for distance, arc_id, fraction in ranked[:limit]
    ]


def generate_feature_rows(trip_dir: Path, candidate_limit: int = 8) -> tuple[list[dict], dict]:
    """Generate one fixed-schema row per valid GPS point and candidate arc."""
    trip = load_trip(trip_dir)
    rows: list[dict] = []
    miss_count = 0
    route_candidate_count = 0

    for point in trip["points"]:
        candidates = candidate_arcs(trip, point, candidate_limit)
        true_arc_id = None
        if trip["route_arc_ids"]:
            route_matches = []
            for route_arc_id in trip["route_arc_ids"]:
                arc = trip["arcs"][route_arc_id]
                distance, fraction = segment_geometry(
                    (point["longitude"], point["latitude"]),
                    arc["start"],
                    arc["end"],
                    trip["origin_lat"],
                )
                route_matches.append((distance, route_arc_id, fraction))
            if route_matches:
                _, true_arc_id, _ = min(route_matches, key=lambda item: (item[0], item[1]))
                route_candidate_count += 1
        candidate_ids = {candidate["arc_id"] for candidate in candidates}
        if true_arc_id is not None and true_arc_id not in candidate_ids:
            miss_count += 1

        for rank, candidate in enumerate(candidates, 1):
            arc_id = candidate["arc_id"]
            arc = trip["arcs"][arc_id]
            heading_delta = abs(wrap_degrees(point["heading_deg"] - arc["bearing_deg"]))
            row = {
                "trip_id": trip["trip_id"],
                "point_idx": point["point_idx"],
                "arc_id": arc_id,
                "longitude": point["longitude"],
                "latitude": point["latitude"],
                "timestamp": point["timestamp"],
                "timestamp_gap_s": point["timestamp_gap_s"],
                "step_distance_m": point["step_distance_m"],
                "point_speed_mps": point["point_speed_mps"],
                "heading_deg": point["heading_deg"],
                "turn_angle_deg": point["turn_angle_deg"],
                "perpendicular_distance_m": candidate["distance_m"],
                "along_arc_fraction": candidate["fraction"],
                "arc_length_m": arc["length_m"],
                "arc_bearing_deg": arc["bearing_deg"],
                "heading_delta_deg": heading_delta,
                "arc_out_degree": trip["out_degree"][arc["from_id"]],
                "arc_in_degree": trip["in_degree"][arc["to_id"]],
                "candidate_rank": rank,
                "label": 1 if arc_id == true_arc_id else 0,
            }
            rows.append(row)

    miss_rate = miss_count / route_candidate_count if route_candidate_count else None
    summary = {
        "trip_id": trip["trip_id"],
        "gps_points_valid": len(trip["points"]),
        "gps_points_with_route_candidate": route_candidate_count,
        "candidate_miss_count": miss_count,
        "candidate_miss_rate": miss_rate,
        "candidate_limit": candidate_limit,
        "route_arc_count": len(trip["route_arc_ids"]),
        "arc_count": len(trip["arcs"]),
    }
    return rows, summary


def write_feature_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    columns = [
        "partition",
        "trip_id",
        "gps_points_valid",
        "gps_points_with_route_candidate",
        "candidate_miss_count",
        "candidate_miss_rate",
        "candidate_limit",
        "route_arc_count",
        "arc_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_exploration/model_development/candidate_features"),
    )
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--sample-per-partition", type=int, default=5)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    for partition in ("train", "student_test"):
        partition_dir = args.data_dir / partition
        trip_dirs = sorted(path for path in partition_dir.iterdir() if path.is_dir()) if partition_dir.exists() else []
        trip_dirs = trip_dirs[:args.sample_per_partition]
        partition_rows: list[dict] = []
        partition_summaries: list[dict] = []
        for trip_dir in trip_dirs:
            rows, summary = generate_feature_rows(trip_dir, args.candidate_limit)
            partition_rows.extend(rows)
            summary["partition"] = partition
            partition_summaries.append(summary)
            all_summaries.append(summary)
        write_feature_csv(output_dir / f"{partition}_candidate_features_sample.csv", partition_rows)
        write_summary_csv(
            output_dir / f"{partition}_candidate_miss_sample.csv",
            partition_summaries,
        )

    write_summary_csv(output_dir / "candidate_miss_summary.csv", all_summaries)
    print(f"Generated sample candidate feature tables with limit={args.candidate_limit}.")


if __name__ == "__main__":
    main()
