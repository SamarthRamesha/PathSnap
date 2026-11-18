"""Log-space map matching for one trip's directed road network."""

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot, inf, isfinite
from typing import Iterable


@dataclass(frozen=True)
class Arc:
    arc_id: int
    source: int
    target: int
    length_m: float


@dataclass(frozen=True)
class Candidate:
    arc_id: int
    offset_m: float
    distance_m: float | None = None
    log_score: float | None = None


@dataclass(frozen=True)
class Observation:
    gps_index: int
    timestamp_s: float
    xy_m: tuple[float, float] | None
    candidates: tuple[Candidate, ...] = ()


@dataclass(frozen=True)
class Break:
    gps_index: int
    reason: str


@dataclass(frozen=True)
class DecodeResult:
    matches: tuple[Candidate | None, ...]
    routes: tuple[tuple[int, ...], ...]
    breaks: tuple[Break, ...]


@dataclass(frozen=True)
class _Step:
    candidate: Candidate
    score: float
    previous: "_Step | None" = None
    appended_arcs: tuple[int, ...] = ()


class _RoadNetwork:
    def __init__(self, arcs: Iterable[Arc]):
        self.arcs = {}
        self.outgoing = defaultdict(list)
        for arc in arcs:
            if arc.arc_id in self.arcs or not isfinite(arc.length_m) or arc.length_m <= 0:
                raise ValueError(f"Duplicate ID or invalid length for arc {arc.arc_id}")
            self.arcs[arc.arc_id] = arc
            self.outgoing[arc.source].append(arc)

    def paths(self, source: int, targets: set[int]):
        """Find shortest directed paths to the requested nodes with one search."""
        pending = targets.copy()
        distances = {source: 0.0}
        parents = {}
        queue = [(0.0, source)]
        found = {}
        while queue and pending:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            if node in pending:
                pending.remove(node)
                path = []
                cursor = node
                while cursor != source:
                    arc = parents[cursor]
                    path.append(arc.arc_id)
                    cursor = arc.source
                found[node] = distance, tuple(reversed(path))
                if not pending:
                    break
            for arc in self.outgoing.get(node, ()):
                total = distance + arc.length_m
                if total < distances.get(arc.target, inf):
                    distances[arc.target] = total
                    parents[arc.target] = arc
                    heappush(queue, (total, arc.target))
        return found

    def transition(self, previous: Candidate, current: Candidate, paths):
        if previous.arc_id == current.arc_id and current.offset_m >= previous.offset_m:
            return current.offset_m - previous.offset_m, ()
        origin = self.arcs[previous.arc_id]
        destination = self.arcs[current.arc_id]
        bridge = paths[origin.target].get(destination.source)
        if bridge is None:
            return inf, ()
        distance, intermediate = bridge
        return (
            origin.length_m - previous.offset_m + distance + current.offset_m,
            (*intermediate, current.arc_id),
        )


def _emission(candidate: Candidate, network: _RoadNetwork, sigma_m: float) -> float:
    arc = network.arcs.get(candidate.arc_id)
    if arc is None or not 0 <= candidate.offset_m <= arc.length_m:
        raise ValueError(f"Unknown arc or invalid offset for candidate {candidate.arc_id}")
    distance, score = candidate.distance_m, candidate.log_score
    if (distance is None) == (score is None):
        raise ValueError("Provide exactly one of distance_m and log_score")
    if score is not None and isfinite(score):
        return score
    if distance is not None and isfinite(distance) and distance >= 0:
        return -0.5 * (distance / sigma_m) ** 2
    raise ValueError("Candidate distance or log score is invalid")


def _advance(network, last, emissions, gps_distance, beta_m):
    needed = defaultdict(set)
    for step in last:
        previous = step.candidate
        origin = network.arcs[previous.arc_id].target
        for candidate, _ in emissions:
            if candidate.arc_id != previous.arc_id or candidate.offset_m < previous.offset_m:
                needed[origin].add(network.arcs[candidate.arc_id].source)
    paths = {origin: network.paths(origin, targets) for origin, targets in needed.items()}
    advanced = []
    for candidate, emission in emissions:
        best = None
        for step in last:
            distance, appended = network.transition(step.candidate, candidate, paths)
            if distance == inf:
                continue
            score = step.score + emission - abs(distance - gps_distance) / beta_m
            if best is None or score > best.score:
                best = _Step(candidate, score, step, appended)
        if best is not None:
            advanced.append(best)
    return advanced


def _finish(last, matches, routes):
    if not last:
        return
    step = max(last, key=lambda state: state.score)
    chain = []
    while step is not None:
        chain.append(step)
        step = step.previous
    route = [chain[-1].candidate.arc_id]
    for step in reversed(chain):
        matches.append(step.candidate)
        route.extend(step.appended_arcs)
    routes.append(tuple(route))


def decode(
    observations: Iterable[Observation],
    arcs: Iterable[Arc],
    *,
    sigma_m: float = 15.0,
    beta_m: float = 30.0,
    max_gap_s: float | None = 30.0,
) -> DecodeResult:
    """Return observation-aligned matches and a route for each decoded segment."""
    scales = (sigma_m, beta_m) if max_gap_s is None else (sigma_m, beta_m, max_gap_s)
    for value in scales:
        if not isfinite(value) or value <= 0:
            raise ValueError("Scoring scales and the optional gap threshold must be positive and finite")
    network = _RoadNetwork(arcs)
    matches, routes, breaks = [], [], []
    last = []
    previous = None
    for observation in observations:
        if previous is not None and observation.gps_index <= previous.gps_index:
            raise ValueError("GPS indices must be strictly increasing")
        if not isfinite(observation.timestamp_s):
            raise ValueError("GPS timestamps must be finite")
        if observation.xy_m is None or not observation.candidates:
            _finish(last, matches, routes)
            last = []
            matches.append(None)
            reason = "missing_gps" if observation.xy_m is None else "no_candidates"
            breaks.append(Break(observation.gps_index, reason))
        else:
            if not all(isfinite(value) for value in observation.xy_m):
                raise ValueError("GPS coordinates must be finite")
            emissions = [
                (candidate, _emission(candidate, network, sigma_m))
                for candidate in observation.candidates
            ]
            if not last:
                last = [_Step(candidate, score) for candidate, score in emissions]
            else:
                gap = observation.timestamp_s - previous.timestamp_s
                advanced = []
                if gap <= 0:
                    reason = "non_increasing_time"
                elif max_gap_s is not None and gap > max_gap_s:
                    reason = "gps_gap"
                else:
                    x, y = observation.xy_m
                    old_x, old_y = previous.xy_m
                    advanced = _advance(network, last, emissions, hypot(x - old_x, y - old_y), beta_m)
                    reason = "disconnected"
                if not advanced:
                    _finish(last, matches, routes)
                    breaks.append(Break(observation.gps_index, reason))
                    advanced = [_Step(candidate, score) for candidate, score in emissions]
                last = advanced
        previous = observation
    _finish(last, matches, routes)
    return DecodeResult(tuple(matches), tuple(routes), tuple(breaks))
