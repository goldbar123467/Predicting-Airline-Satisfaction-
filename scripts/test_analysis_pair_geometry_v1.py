import unittest
import numpy as np
from analysis_pair_geometry_v1 import ranking_changes


class PairGeometryTests(unittest.TestCase):
    def test_exact_brute_pairs_many_permutations(self):
        rng = np.random.default_rng(103)
        for n in [8, 19, 37]:
            y = np.arange(n) % 2
            for _ in range(20):
                a, b = rng.permutation(n), rng.permutation(n)
                result = ranking_changes(y, a, b)
                repaired = broken = 0
                for i in np.flatnonzero(y == 1):
                    for j in np.flatnonzero(y == 0):
                        before, after = b[i] > b[j], a[i] > a[j]
                        repaired += int(not before and after)
                        broken += int(before and not after)
                self.assertEqual(result["pairs_repaired_by_reference"], repaired)
                self.assertEqual(result["pairs_broken_by_reference"], broken)

    def test_monotone_transform_changes_no_ranking(self):
        y, a = np.array([0, 1, 0, 1]), np.array([.1, .2, .3, .4])
        result = ranking_changes(y, a, a ** 2)
        self.assertEqual(result["opposite_class_pairs_reordered"], 0)
        self.assertIsNone(result["repair_fraction_among_reordered_pairs"])

    def test_ties_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "tied"):
            ranking_changes(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 2]), np.arange(4))

    def test_reversal_switches_repairs_and_breaks(self):
        y, a, b = np.arange(8) % 2, np.array([0, 3, 1, 2, 6, 5, 7, 4]), np.arange(8)
        first, second = ranking_changes(y, a, b), ranking_changes(y, b, a)
        self.assertEqual(first["pairs_repaired_by_reference"], second["pairs_broken_by_reference"])
        self.assertEqual(first["pairs_broken_by_reference"], second["pairs_repaired_by_reference"])


if __name__ == "__main__":
    unittest.main()
