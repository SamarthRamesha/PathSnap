"""Read the dataset's implicit row IDs without opening reference routes."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from re import fullmatch
from types import MappingProxyType

from .schemas import GpsObservation, validate_timestamps


class DatasetError(ValueError):
    pass


class NetworkNotFound(DatasetError):
    pass


@dataclass(frozen=True, slots=True)
class RoadArc:
    arc_id: int
    source: int
    target: int
    length_m: float


@dataclass(frozen=True, slots=True)
class RoadNetwork:
    nodes_path: Path
    arcs_path: Path
    arcs: Mapping[int, RoadArc]
    zero_length_arcs: int


def _records(path: Path, width: int) -> Iterator[list[str]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                fields = line.split()
                if len(fields) != width:
                    raise DatasetError(f"{path.suffix} row {number} needs {width} fields")
                yield fields
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Could not read the {path.suffix} file") from exc


def load_track(path: Path) -> list[GpsObservation]:
    observations = []
    for number, fields in enumerate(_records(path, 3), start=1):
        try:
            longitude, latitude, timestamp_s = map(float, fields)
            observation = GpsObservation(
                longitude=longitude, latitude=latitude, timestamp_s=timestamp_s
            )
        except ValueError as exc:
            raise DatasetError(f"Invalid .track row {number}") from exc
        observations.append(observation)
    if not observations:
        raise DatasetError("The .track file contains no GPS observations")
    try:
        validate_timestamps(observations)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    return observations


def load_nodes(path: Path) -> list[tuple[float, float]]:
    nodes = []
    for number, fields in enumerate(_records(path, 2), start=1):
        try:
            lon, lat = map(float, fields)
        except ValueError as exc:
            raise DatasetError(f"Invalid coordinates on .nodes row {number}") from exc
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise DatasetError(f"Invalid coordinates on .nodes row {number}")
        nodes.append((lon, lat))

    return nodes


def _length_m(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Use the same spherical distance convention as candidate generation."""
    lon1, lat1 = start
    lon2, lat2 = end
    phi1, phi2 = radians(lat1), radians(lat2)
    value = (
        sin(radians(lat2 - lat1) / 2) ** 2
        + cos(phi1) * cos(phi2) * sin(radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_000.0 * asin(min(1.0, sqrt(value)))


@lru_cache(maxsize=2)
def load_network(root: Path, network_id: str) -> RoadNetwork:
    if fullmatch(r"[0-9]{8}", network_id) is None:
        raise NetworkNotFound("Network ID must contain eight digits")

    try:
        root = root.resolve()
        directory = root / network_id
        nodes_path = (directory / f"{network_id}.nodes").resolve()
        arcs_path = (directory / f"{network_id}.arcs").resolve()
        if any(not path.is_relative_to(root) for path in (nodes_path, arcs_path)):
            raise NetworkNotFound("Network files must be inside the configured directory")
        files_present = nodes_path.is_file() and arcs_path.is_file()
    except (OSError, RuntimeError) as exc:
        raise DatasetError(f"Could not inspect the files for network {network_id}") from exc
    if nodes_path.suffix != ".nodes" or arcs_path.suffix != ".arcs" or not files_present:
        raise NetworkNotFound(f"No .nodes and .arcs pair is registered for network {network_id}")

    nodes = load_nodes(nodes_path)

    arcs: dict[int, RoadArc] = {}
    zero_length = 0
    for arc_id, fields in enumerate(_records(arcs_path, 2)):
        try:
            source, target = map(int, fields)
        except ValueError as exc:
            raise DatasetError(f"Invalid node IDs for arc {arc_id}") from exc
        if not (0 <= source < len(nodes) and 0 <= target < len(nodes)):
            raise DatasetError(f"Arc {arc_id} references a missing node")
        length = _length_m(nodes[source], nodes[target])
        if length == 0:
            # The source data contains coincident endpoints. Keep later row IDs intact.
            zero_length += 1
            continue
        arcs[arc_id] = RoadArc(arc_id, source, target, length)
    if not arcs:
        raise DatasetError("The network has no positive-length directed arcs")
    return RoadNetwork(nodes_path, arcs_path, MappingProxyType(arcs), zero_length)
