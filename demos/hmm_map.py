"""Render a controlled classifier-only versus Viterbi comparison."""

import argparse
from math import cos, hypot, radians
from pathlib import Path

import folium

from hmm.viterbi import Arc, GpsSample, decode_probabilities


def render(output: Path) -> None:
    # Two nearby roads share no nodes. Arc 1 joins the two candidate arcs
    # on the main road, and the final two GPS rows follow a timestamp gap.
    nodes = [(0, 0), (40, 0), (80, 0), (120, 0), (80, 18), (120, 18),
             (145, 5), (185, 5)]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (6, 7)]
    arcs = [
        Arc(i, source, target, hypot(
            nodes[target][0] - nodes[source][0], nodes[target][1] - nodes[source][1]
        ))
        for i, (source, target) in enumerate(edges)
    ]
    gps = [
        GpsSample(0, 0, (12, 3)),
        GpsSample(1, 4, (89, 12)),
        GpsSample(2, 5, (107, 4)),
        GpsSample(3, 55, (154, 8)),
        GpsSample(4, 57, (177, 3)),
    ]
    scores = [[(0, 0.96)], [(3, 0.75), (2, 0.25)], [(2, 0.92)],
              [(4, 0.91)], [(4, 0.94)]]
    offsets = [{0: 12}, {3: 9, 2: 9}, {2: 27}, {4: 9}, {4: 32}]
    independent = [[max(row, key=lambda candidate: candidate[1])] for row in scores]
    baseline = decode_probabilities(gps, independent, offsets, arcs)
    viterbi = decode_probabilities(gps, scores, offsets, arcs)

    latitude, longitude = 49.844, 8.595

    def lat_lon(point):
        x, y = point
        return (latitude + y / 111195, longitude + x / (111195 * cos(radians(latitude))))

    map_view = folium.Map(location=lat_lon((92, 5)), zoom_start=18, tiles="OpenStreetMap")
    road_layer = folium.FeatureGroup(name="Road network").add_to(map_view)
    for arc in arcs:
        folium.PolyLine(
            [lat_lon(nodes[arc.source]), lat_lon(nodes[arc.target])],
            tooltip=f"Directed arc {arc.arc_id} ({arc.source} to {arc.target})",
            color="#94a3b8", weight=3,
        ).add_to(road_layer)

    gps_layer = folium.FeatureGroup(name="GPS observations").add_to(map_view)
    folium.PolyLine([lat_lon(sample.xy_m) for sample in gps[:3]], color="#334155",
                    dash_array="3 6", weight=2).add_to(gps_layer)
    folium.PolyLine([lat_lon(sample.xy_m) for sample in gps[3:]], color="#334155",
                    dash_array="3 6", weight=2).add_to(gps_layer)
    for sample, picked, matched in zip(gps, baseline.matches, viterbi.matches, strict=True):
        folium.CircleMarker(
            lat_lon(sample.xy_m), radius=5, color="#334155", fill=True, fill_opacity=1,
            tooltip=f"GPS {sample.gps_index} • {sample.timestamp_s}s",
            popup=f"Classifier-only arc {picked.arc_id}. Viterbi arc {matched.arc_id}.",
        ).add_to(gps_layer)
    for event in viterbi.breaks:
        sample = gps[event.gps_index]
        folium.Marker(
            lat_lon(sample.xy_m), tooltip=f"GPS {event.gps_index} • {event.reason}",
            icon=folium.Icon(color="gray", icon="pause"),
        ).add_to(gps_layer)

    for name, result, colors, weight in (
        ("Viterbi route segments", viterbi, ("#2563eb", "#047857"), 7),
        ("Classifier-only selections", baseline, ("#c2410c",), 4),
    ):
        layer = folium.FeatureGroup(name=name).add_to(map_view)
        for segment, route in enumerate(result.routes):
            for arc_id in route:
                arc = arcs[arc_id]
                folium.PolyLine(
                    [lat_lon(nodes[arc.source]), lat_lon(nodes[arc.target])],
                    color=colors[segment % len(colors)], weight=weight,
                    dash_array="6 8" if result is baseline else None,
                    tooltip=f"{name} • segment {segment + 1} • arc {arc_id}",
                ).add_to(layer)

    baseline_breaks = sum(event.reason == "disconnected" for event in baseline.breaks)
    viterbi_breaks = sum(event.reason == "disconnected" for event in viterbi.breaks)
    gap = gps[3].timestamp_s - gps[2].timestamp_s
    map_view.get_root().html.add_child(folium.Element(f"""
        <section style="position:fixed;bottom:24px;left:16px;z-index:1000;
            max-width:min(360px,90vw);padding:14px 18px;border-radius:8px;
            background:white;color:#172554;box-shadow:0 2px 14px #0003">
            <strong>PathSnap · Viterbi demo</strong>
            <div>Synthetic GPS observations and classifier scores</div>
            <div>Disconnected steps: classifier-only {baseline_breaks}, Viterbi {viterbi_breaks}</div>
            <div>GPS 1: classifier-only arc {baseline.matches[1].arc_id},
                Viterbi arc {viterbi.matches[1].arc_id}</div>
            <div>The pause marker shows the {gap:g}-second GPS gap.</div>
        </section>
    """))
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_view.fit_bounds([lat_lon(point) for point in nodes], padding=(30, 30), max_zoom=19)
    output.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(str(output))
    print(f"Classifier-only disconnected steps: {baseline_breaks}")
    print(f"Viterbi disconnected steps: {viterbi_breaks}")
    print(f"Viterbi segments: {viterbi.routes}")
    print(f"Viterbi breaks: {viterbi.breaks}")
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("hmm-demo.html"))
    render(parser.parse_args().output)
