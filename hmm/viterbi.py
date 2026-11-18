"""Log-space map matching for one trip's directed road network."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from math import hypot, inf, isfinite, log
from numbers import Integral, Real
from sys import float_info


_SCORE_RECENTER_THRESHOLD = float_info.max / 4


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
class GpsSample:
    gps_index: int
    timestamp_s: float
    xy_m: tuple[float, float] | None


@dataclass(frozen=True)
class Observation(GpsSample):
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


def _valid_id(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _finite_number(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


class _RoadNetwork:
    def __init__(self, arcs: Iterable[Arc]):
        self.arcs = {}
        self.outgoing = defaultdict(list)
        for arc in arcs:
            if not all(_valid_id(value) for value in (arc.arc_id, arc.source, arc.target)):
                raise ValueError("Arc and endpoint IDs must be nonnegative integers")
            if arc.arc_id in self.arcs or not _finite_number(arc.length_m) or arc.length_m <= 0:
                raise ValueError(f"Duplicate ID or invalid length for arc {arc.arc_id}")
            self.arcs[arc.arc_id] = arc
            self.outgoing[arc.source].append(arc)

    def paths(self, source: int, targets: frozenset[int]):
        """Find shortest directed paths to the requested nodes with one search."""
        pending = set(targets)
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
    if not _valid_id(candidate.arc_id):
        raise ValueError("Candidate arc ID must be a nonnegative integer")
    arc = network.arcs.get(candidate.arc_id)
    if arc is None or not _finite_number(candidate.offset_m) or not 0 <= candidate.offset_m <= arc.length_m:
        raise ValueError(f"Unknown arc or invalid offset for candidate {candidate.arc_id}")
    distance, score = candidate.distance_m, candidate.log_score
    if (distance is None) == (score is None):
        raise ValueError("Provide exactly one of distance_m and log_score")
    if _finite_number(score):
        return float(score)
    if _finite_number(distance) and distance >= 0:
        scaled_distance = float(distance) / float(sigma_m)
        try:
            emission = -0.5 * scaled_distance ** 2
        except OverflowError:
            # Apply the factor first: the square can overflow before the result does.
            emission = (-0.5 * scaled_distance) * scaled_distance
        if isfinite(emission):
            return emission
    raise ValueError("Candidate distance or log score is invalid or out of range")


def _advance(network, shortest_paths, last, emissions, gps_distance, beta_m):
    needed = defaultdict(set)
    for step in last:
        previous = step.candidate
        origin = network.arcs[previous.arc_id].target
        for candidate, _ in emissions:
            if candidate.arc_id != previous.arc_id or candidate.offset_m < previous.offset_m:
                needed[origin].add(network.arcs[candidate.arc_id].source)
    paths = {origin: shortest_paths(origin, frozenset(targets)) for origin, targets in needed.items()}
    previous_shift = emission_shift = 0.0
    # Recenter only extreme scores so ordinary arithmetic keeps its original tie behaviour.
    if any(abs(step.score) > _SCORE_RECENTER_THRESHOLD for step in last) or any(
        abs(score) > _SCORE_RECENTER_THRESHOLD for _, score in emissions
    ):
        previous_shift = max(step.score for step in last)
        emission_shift = max(score for _, score in emissions)
    advanced = []
    for candidate, emission in emissions:
        best = None
        for step in last:
            distance, appended = network.transition(step.candidate, candidate, paths)
            if distance == inf:
                continue
            score = (step.score - previous_shift) + (emission - emission_shift)
            score -= abs(distance - gps_distance) / beta_m
            if not isfinite(score):
                raise ValueError("Transition score exceeds the finite numeric range")
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
        if not _finite_number(value) or value <= 0:
            raise ValueError("Scoring scales and the optional gap threshold must be positive and finite")
    network = _RoadNetwork(arcs)
    # Paths depend only on the graph and node IDs, so reuse exact queries within this trip.
    shortest_paths = lru_cache(maxsize=128)(network.paths)
    matches, routes, breaks = [], [], []
    last = []
    previous = None
    for observation in observations:
        if not _valid_id(observation.gps_index):
            raise ValueError("GPS indices must be nonnegative integers")
        if previous is not None and observation.gps_index <= previous.gps_index:
            raise ValueError("GPS indices must be strictly increasing")
        if not _finite_number(observation.timestamp_s):
            raise ValueError("GPS timestamps must be finite")
        xy = observation.xy_m
        if xy is not None and (
            not isinstance(xy, Sequence)
            or len(xy) != 2
            or not all(_finite_number(value) for value in xy)
        ):
            raise ValueError("GPS coordinates must be a pair of finite numbers")
        if xy is None or not observation.candidates:
            _finish(last, matches, routes)
            last = []
            matches.append(None)
            reason = "missing_gps" if observation.xy_m is None else "no_candidates"
            breaks.append(Break(observation.gps_index, reason))
        else:
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
                    x, y = xy
                    old_x, old_y = previous.xy_m
                    gps_distance = hypot(x - old_x, y - old_y)
                    if not isfinite(gps_distance):
                        raise ValueError("GPS displacement exceeds the finite distance range")
                    advanced = _advance(network, shortest_paths, last, emissions, gps_distance, beta_m)
                    reason = "disconnected"
                if not advanced:
                    _finish(last, matches, routes)
                    breaks.append(Break(observation.gps_index, reason))
                    advanced = [_Step(candidate, score) for candidate, score in emissions]
                last = advanced
        previous = observation
    _finish(last, matches, routes)
    return DecodeResult(tuple(matches), tuple(routes), tuple(breaks))


def decode_probabilities(
    gps_samples: Iterable[GpsSample],
    probabilities: Iterable[Sequence[tuple[int, float]]],
    offsets_m: Iterable[Mapping[int, float]],
    arcs: Iterable[Arc],
    *,
    beta_m: float = 30.0,
    max_gap_s: float | None = 30.0,
) -> DecodeResult:
    """Decode model probabilities with one score list and offset map per GPS row."""
    observations = []
    for sample, scores, offsets in zip(gps_samples, probabilities, offsets_m, strict=True):
        if not isinstance(scores, Sequence):
            raise ValueError(f"Candidate scores must be a sequence at GPS row {sample.gps_index}")
        if len(scores) > 8:
            raise ValueError(f"Expected at most eight candidates at GPS row {sample.gps_index}")
        if not isinstance(offsets, Mapping):
            raise ValueError(f"Candidate offsets must be a mapping at GPS row {sample.gps_index}")
        candidates = []
        seen = set()
        for pair in scores:
            try:
                arc_id, probability = pair
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Expected an arc ID and probability at GPS row {sample.gps_index}") from exc
            if not _valid_id(arc_id):
                raise ValueError(f"Candidate arc ID must be a nonnegative integer at GPS row {sample.gps_index}")
            if arc_id in seen or not _finite_number(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Duplicate arc or invalid probability at GPS row {sample.gps_index}")
            seen.add(arc_id)
            if probability == 0 or sample.xy_m is None:
                continue
            if arc_id not in offsets:
                raise ValueError(f"Missing offset for arc {arc_id} at GPS row {sample.gps_index}")
            candidates.append(Candidate(arc_id, offsets[arc_id], log_score=log(probability)))
        observations.append(Observation(
            sample.gps_index, sample.timestamp_s, sample.xy_m, tuple(candidates)
        ))
    return decode(observations, arcs, beta_m=beta_m, max_gap_s=max_gap_s)
