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
