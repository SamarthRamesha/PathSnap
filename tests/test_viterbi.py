import unittest

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


if __name__ == "__main__":
    unittest.main()
