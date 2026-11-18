import unittest
from math import log

from hmm.viterbi import Arc, Break, Candidate, GpsSample, decode_probabilities


class ProbabilityAdapterTests(unittest.TestCase):
    def test_model_scores_and_offsets_feed_viterbi(self):
        arcs = [Arc(0, 0, 1, 10), Arc(1, 1, 2, 10), Arc(2, 3, 4, 10)]
        result = decode_probabilities(
            [GpsSample(10, 0, (8, 0)), GpsSample(11, 1, (13, 0))],
            [[(0, 0.7)], [(2, 0.9), (1, 0.05)]],
            [{0: 8}, {1: 3, 2: 3}],
            arcs,
        )
        self.assertEqual(result.matches, (
            Candidate(0, 8, log_score=log(0.7)), Candidate(1, 3, log_score=log(0.05)),
        ))
        self.assertEqual(result.routes, ((0, 1),))
        self.assertEqual(result.breaks, ())

    def test_empty_zero_probability_and_missing_gps(self):
        result = decode_probabilities(
            [GpsSample(1, 0, (0, 0)), GpsSample(2, 1, (0, 0)), GpsSample(3, 2, None)],
            [[], [(0, 0)], [(0, 1)]],
            [{}, {}, {}],
            [],
        )
        self.assertEqual(result.matches, (None, None, None))
        self.assertEqual(result.routes, ())
        self.assertEqual(result.breaks, (
            Break(1, "no_candidates"), Break(2, "no_candidates"), Break(3, "missing_gps"),
        ))

    def test_rejects_missing_offset_and_invalid_score_rows(self):
        sample = GpsSample(1, 0, (0, 0))
        arcs = [Arc(0, 0, 1, 10)]
        cases = [
            ([(0, 1)], {}, "Missing offset"),
            ([(0, 0.5), (0, 0.5)], {0: 1}, "Duplicate arc"),
            ([(0, float("nan"))], {0: 1}, "invalid probability"),
            ([(0, 1.1)], {0: 1}, "invalid probability"),
            ([(arc_id, 0.1) for arc_id in range(9)], {}, "at most eight"),
        ]
        for scores, offsets, message in cases:
            with self.subTest(scores=scores), self.assertRaisesRegex(ValueError, message):
                decode_probabilities([sample], [scores], [offsets], arcs)

    def test_rejects_misaligned_rows(self):
        with self.assertRaises(ValueError):
            decode_probabilities([GpsSample(1, 0, (0, 0))], [], [{}], [])

    def test_smallest_probabilities_and_eight_candidates_on_a_long_trace(self):
        tiny = float.fromhex("0x0.0000000000001p-1022")
        scores = [(arc_id, (2 * tiny if arc_id == 7 else tiny)) for arc_id in range(8)]
        size = 2000
        result = decode_probabilities(
            (GpsSample(index, index, (index, 0)) for index in range(size)),
            (scores for _ in range(size)),
            ({arc_id: index for arc_id in range(8)} for index in range(size)),
            [Arc(arc_id, 2 * arc_id, 2 * arc_id + 1, size) for arc_id in range(8)],
        )
        self.assertEqual(result.routes, ((7,),))
        self.assertEqual(result.breaks, ())
        self.assertEqual(len(result.matches), size)
        self.assertTrue(all(match.arc_id == 7 for match in result.matches))
        self.assertEqual(result.matches[-1].log_score, log(2 * tiny))

    def test_malformed_score_pairs_offsets_and_probability_values(self):
        sample = GpsSample(0, 0, (0, 0))
        arcs = [Arc(0, 0, 1, 10)]
        cases = [
            (None, {}, "scores must be a sequence"),
            ([(0,)], {}, "arc ID and probability"),
            ([(0, "0.5")], {0: 0}, "invalid probability"),
            ([(0, float("inf"))], {0: 0}, "invalid probability"),
            ([(0, -0.1)], {0: 0}, "invalid probability"),
            ([(0.0, 0.5)], {0: 0}, "nonnegative integer"),
            ([(0, 0.5)], [0], "offsets must be a mapping"),
            ([(0, 0.5)], {0: float("inf")}, "invalid offset"),
            ([(1, 0.5)], {1: 0}, "Unknown arc"),
        ]
        for scores, offsets, message in cases:
            with self.subTest(scores=scores, offsets=offsets), self.assertRaisesRegex(ValueError, message):
                decode_probabilities([sample], [scores], [offsets], arcs)
