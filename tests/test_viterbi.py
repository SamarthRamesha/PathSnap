import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hmm import viterbi
from hmm.viterbi import Arc, Break, Candidate, Observation, decode


def observation(index, x, *candidates, timestamp=None):
    return Observation(index, index if timestamp is None else timestamp,
                       None if x is None else (x, 0), candidates)


def scored(arc, offset, score=0):
    return Candidate(arc, offset, log_score=score)


class ViterbiTests(unittest.TestCase):
    def test_global_path_uses_weaker_emission_and_intermediate_arc(self):
        arcs = [Arc(0, 0, 1, 10), Arc(1, 1, 2, 10),
                Arc(2, 2, 3, 10), Arc(3, 4, 5, 10)]
        disconnected = scored(3, 2)
        connected = scored(2, 2, -2)
        result = decode([
            observation(0, 8, scored(0, 8)),
            observation(1, 22, disconnected, connected),
            observation(2, 25, scored(2, 5)),
        ], arcs)
        self.assertEqual(result.matches[1], connected)
        self.assertEqual(result.routes, ((0, 1, 2),))
        self.assertEqual(result.breaks, ())

    def test_distance_emission_and_transition_disagreement(self):
        arcs = [Arc(0, 0, 1, 10), Arc(1, 1, 2, 100), Arc(2, 1, 3, 10)]
        near_gps = Candidate(1, 90, distance_m=1)
        plausible = Candidate(2, 3, distance_m=6)
        result = decode([
            observation(0, 8, scored(0, 8)),
            observation(1, 13, near_gps, plausible),
        ], arcs)
        self.assertEqual(result.matches[1], plausible)
        self.assertEqual(result.routes, ((0, 2),))

    def test_gaussian_fallback_prefers_closer_candidate(self):
        arcs = [Arc(0, 0, 1, 10), Arc(1, 2, 3, 10)]
        closer = Candidate(1, 1, distance_m=3)
        result = decode([
            observation(0, 0, Candidate(0, 1, distance_m=14), closer),
        ], arcs)
        self.assertEqual(result.matches, (closer,))

    def test_disconnected_and_wrong_way_restart(self):
        arcs = [Arc(0, 0, 1, 10), Arc(1, 4, 5, 10)]
        result = decode([
            observation(0, 8, scored(0, 8)),
            observation(1, 2, scored(0, 2)),
            observation(2, 20, scored(1, 2)),
        ], arcs)
        self.assertEqual(result.routes, ((0,), (0,), (1,)))
        self.assertEqual(result.breaks, (Break(1, "disconnected"), Break(2, "disconnected")))

    def test_return_to_same_arc_via_directed_loop(self):
        arcs = [Arc(0, 0, 1, 10), Arc(1, 1, 0, 10)]
        result = decode([
            observation(0, 8, scored(0, 8)),
            observation(1, 2, scored(0, 2)),
        ], arcs)
        self.assertEqual(result.routes, ((0, 1, 0),))

    def test_missing_gps_empty_candidates_and_gap_preserve_alignment(self):
        arcs = [Arc(0, 0, 1, 10)]
        first, second = scored(0, 1), scored(0, 2)
        result = decode([
            observation(0, 1, first),
            observation(1, None),
            observation(3, 2, second),
            observation(4, 2),
            observation(5, 2, second),
            observation(6, 3, scored(0, 3), timestamp=50),
            observation(7, 4, scored(0, 4), timestamp=50),
        ], arcs)
        self.assertEqual(result.matches[:5], (first, None, second, None, second))
        self.assertEqual(result.routes, ((0,), (0,), (0,), (0,), (0,)))
        self.assertEqual(result.breaks, (
            Break(1, "missing_gps"), Break(4, "no_candidates"),
            Break(6, "gps_gap"), Break(7, "non_increasing_time"),
        ))

    def test_invalid_candidate_offset_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid offset"):
            decode([observation(0, 0, scored(0, 11))], [Arc(0, 0, 1, 10)])


    def test_forward_offsets_on_one_way_arc_keep_sparse_indices_aligned(self):
        first, second = scored(0, 1), scored(0, 9)
        result = decode(
            [observation(4, 1, first), observation(15, 9, second)],
            [Arc(0, 0, 1, 10)],
        )
        self.assertEqual(result.matches, (first, second))
        self.assertEqual(result.routes, ((0,),))
        self.assertEqual(result.breaks, ())

    def test_consecutive_missing_rows_and_disconnected_islands(self):
        first, second = scored(0, 1), scored(1, 8)
        third = scored(2, 1)
        result = decode([
            observation(0, 1, first), observation(1, 2), observation(2, 3),
            observation(4, None), observation(6, 8, second), observation(7, 12, third),
        ], [Arc(0, 2, 3, 10), Arc(1, 4, 5, 10), Arc(2, 6, 7, 10)])
        self.assertEqual(result.matches, (first, None, None, None, second, third))
        self.assertEqual(result.routes, ((0,), (1,), (2,)))
        self.assertEqual(result.breaks, (
            Break(1, "no_candidates"), Break(2, "no_candidates"),
            Break(4, "missing_gps"), Break(7, "disconnected"),
        ))

    def test_equal_scores_follow_candidate_order(self):
        first, second = scored(1, 1), scored(0, 1)
        end_first, end_second = scored(1, 4), scored(0, 4)
        arcs = [Arc(0, 0, 1, 10), Arc(1, 2, 3, 10)]
        result = decode([
            observation(0, 1, first, second), observation(1, 4, end_first, end_second)
        ], arcs)
        self.assertEqual(result.matches, (first, end_first))
        self.assertEqual(result.routes, ((1,),))

    def test_equal_parallel_bridges_keep_directed_arc_order(self):
        arcs = [Arc(0, 0, 1, 10), Arc(9, 1, 2, 10), Arc(3, 1, 2, 10), Arc(2, 2, 3, 10)]
        result = decode([observation(0, 8, scored(0, 8)), observation(1, 22, scored(2, 2))], arcs)
        self.assertEqual(result.routes, ((0, 9, 2),))
        graph = {arc.arc_id: arc for arc in arcs}
        for route in result.routes:
            for a, b in zip(route, route[1:]):
                self.assertEqual(graph[a].target, graph[b].source)

    def test_rejects_malformed_coordinates_even_without_candidates(self):
        for xy in ((float("nan"), 0), (float("inf"), 0), (1,), (0, 1, 2), ("0", 1)):
            with self.subTest(xy=xy), self.assertRaisesRegex(ValueError, "pair of finite numbers"):
                decode([Observation(0, 0, xy)], [])
        with self.assertRaisesRegex(ValueError, "pair of finite numbers"):
            decode([Observation(0, 0, (1,), (scored(0, 1),))], [Arc(0, 0, 1, 10)])

    def test_rejects_nonintegral_indices_and_nonfinite_timestamps(self):
        for code in (-1, 0.5, True):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "nonnegative integers"):
                decode([Observation(code, 0, (0, 0))], [])
        with self.assertRaisesRegex(ValueError, "timestamps must be finite"):
            decode([Observation(0, float("inf"), None)], [])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            decode([Observation(0, 0, None), Observation(0, 1, None)], [])

    def test_rejects_malformed_graph_and_duplicate_arc_ids(self):
        arc = Arc(0, 0, 1, 10)
        cases = [
            ([arc, Arc(0, 1, 2, 10)], "Duplicate"),
            ([Arc(-1, 0, 1, 10)], "nonnegative integers"),
            ([Arc(0, -1, 1, 10)], "nonnegative integers"),
            ([Arc(0, 0, "1", 10)], "nonnegative integers"),
            ([Arc(0, 0, 1, 0)], "invalid length"),
            ([Arc(0, 0, 1, float("inf"))], "invalid length"),
            ([Arc(0, 0, 1, float("nan"))], "invalid length"),
        ]
        for arcs, message in cases:
            with self.subTest(arcs=arcs), self.assertRaisesRegex(ValueError, message):
                decode([], arcs)

    def test_rejects_invalid_candidate_ids_offsets_and_scores(self):
        cases = [
            (scored(0.0, 0), "nonnegative integer"),
            (scored(9, 0), "Unknown arc"),
            (scored(0, -1), "invalid offset"),
            (scored(0, float("nan")), "invalid offset"),
            (scored(0, "0"), "invalid offset"),
            (scored(0, 0, float("inf")), "log score"),
            (scored(0, 0, float("nan")), "log score"),
            (Candidate(0, 0, distance_m=1e200), "out of range"),
        ]
        for candidate, message in cases:
            with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, message):
                decode([observation(0, 0, candidate)], [Arc(0, 0, 1, 10)])

    def test_extreme_finite_log_scores_do_not_hide_the_best_route(self):
        preferred = scored(1, 0, -9e307)
        finish = scored(1, 1, -1e308)
        result = decode([
            observation(0, 0, scored(0, 0, -1e308), preferred),
            observation(1, 1, scored(0, 1, -1e308), finish),
        ], [Arc(0, 0, 1, 10), Arc(1, 2, 3, 10)])
        self.assertEqual(result.matches, (preferred, finish))
        self.assertEqual(result.routes, ((1,),))

    def test_cached_queries_do_not_change_routes_or_cross_networks(self):
        arcs = [Arc(0, 0, 1, 10), Arc(1, 1, 0, 10)]
        trajectory = [
            observation(index, index, scored(index % 2, 2)) for index in range(20)
        ]
        original_paths = viterbi._RoadNetwork.paths
        searches = []

        def count_paths(network, source, targets):
            searches.append((source, targets))
            return original_paths(network, source, targets)

        with patch.object(viterbi._RoadNetwork, "paths", count_paths):
            result = decode(trajectory, arcs)
            self.assertEqual(result.routes, (tuple(index % 2 for index in range(20)),))
            self.assertEqual(len(searches), 2)
            second = decode(trajectory, [Arc(0, 0, 2, 10), Arc(1, 1, 0, 10)])
        self.assertEqual(second.routes, ((0,),) + ((1, 0),) * 9 + ((1,),))
        self.assertIn(Break(1, "disconnected"), second.breaks)


@unittest.skipUnless(find_spec("folium") and find_spec("numpy"), "requires optional benchmark dependencies")
class BenchmarkInputTests(unittest.TestCase):
    def test_prepare_trip_rejects_numpy_negative_node_aliases(self):
        from demos.benchmark_trip import prepare_trip

        with TemporaryDirectory() as directory:
            trip = Path(directory) / "00000001"
            trip.mkdir()
            (trip / "00000001.track").write_text("0 0 0\n", encoding="utf-8")
            (trip / "00000001.nodes").write_text("0 0\n0.001 0\n", encoding="utf-8")
            (trip / "00000001.arcs").write_text("-1 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid node IDs"):
                prepare_trip(trip)


if __name__ == "__main__":
    unittest.main()
