"""Dataset loader unit tests."""

import pytest

from src.api.network import DatasetError, NetworkNotFound, load_network, load_track

NETWORK_ID = "00000001"


def network_files(root, nodes, arcs):
    trip = root / NETWORK_ID
    trip.mkdir()
    (trip / f"{NETWORK_ID}.nodes").write_text(nodes, encoding="utf-8")
    (trip / f"{NETWORK_ID}.arcs").write_text(arcs, encoding="utf-8")
    return trip


def test_directed_network_retains_arc_ids_and_valid_disconnected_components(tmp_path):
    network_files(tmp_path, "0 0\n0.001 0\n1 1\n1.001 1\n", "0 0\n1 0\n2 3\n")
    network = load_network(tmp_path, NETWORK_ID)

    assert tuple(network.arcs) == (1, 2)
    assert network.zero_length_arcs == 1
    assert (network.arcs[1].source, network.arcs[1].target) == (1, 0)
    assert network.arcs[1].length_m == pytest.approx(111.195, abs=0.001)
    assert network is load_network(tmp_path, NETWORK_ID)
    with pytest.raises(TypeError):
        network.arcs[0] = network.arcs[1]


@pytest.mark.parametrize(("nodes", "arcs", "message"), [
    ("0 0\n0.001 0\n", "0 1\n\n1 0\n", "row 2 needs 2 fields"),
    ("0 0\n0.001 0\n", "0 -1\n", "missing node"),
    ("0 0\n0.001 0\n", "0 0\n", "no positive-length"),
    ("0 0\n0.001 0\n", "a 1\n", "Invalid node IDs"),
    ("0 0\nnan 0\n", "0 1\n", "Invalid coordinates"),
    ("0 0\n0.001 90.1\n", "0 1\n", "Invalid coordinates"),
])
def test_malformed_network_rows_are_rejected(tmp_path, nodes, arcs, message):
    network_files(tmp_path, nodes, arcs)
    with pytest.raises(DatasetError, match=message):
        load_network(tmp_path, NETWORK_ID)


def test_missing_network_pair(tmp_path):
    with pytest.raises(NetworkNotFound, match=r"No \.nodes and \.arcs pair"):
        load_network(tmp_path, NETWORK_ID)


def test_track_accepts_stationary_samples_and_real_time_gaps(tmp_path):
    track = tmp_path / "trip.track"
    track.write_text("8.5 49.8 1\n8.5 49.8 51\n8.6 49.8 52\n", encoding="utf-8")

    observations = load_track(track)
    assert [row.timestamp_s for row in observations] == [1, 51, 52]
    assert [row.longitude for row in observations] == [8.5, 8.5, 8.6]


@pytest.mark.parametrize("data", ["", "8.5 49.8\n", "8.5 49.8 nan\n", "8.5 49.8 2\n8.5 49.8 1\n"])
def test_track_rejects_malformed_or_misordered_input(tmp_path, data):
    track = tmp_path / "trip.track"
    track.write_text(data, encoding="utf-8")
    with pytest.raises(DatasetError):
        load_track(track)
