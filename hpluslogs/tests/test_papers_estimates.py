import unittest
from hpluslogs.scripts.estimate_papers_chroma import chunk_count, estimated_input_tokens
from hpluslogs.scripts.papers_transfer_progress import summarize


class CapacityEstimateTest(unittest.TestCase):
    def test_chunk_boundaries_and_overlap(self):
        self.assertEqual(chunk_count(0, 800, 100), 0)
        self.assertEqual(chunk_count(800, 800, 100), 1)
        self.assertEqual(chunk_count(801, 800, 100), 2)
        self.assertEqual(chunk_count(1500, 800, 100), 2)
        self.assertEqual(chunk_count(1501, 800, 100), 3)
        with self.assertRaises(ValueError):
            chunk_count(100, 800, 800)

    def test_billing_includes_repeated_overlap_and_special_tokens(self):
        self.assertEqual(estimated_input_tokens(0,0,800,100),0)
        self.assertEqual(estimated_input_tokens(800,1000,800,100),1001)
        self.assertEqual(estimated_input_tokens(1500,1800,800,100),1922)

    def test_transfer_excludes_extras_and_outdated_complete_files(self):
        expected = {'ready': [100, 10], 'partial': [200, 20], 'old': [100, 30], 'missing': [50, 40]}
        actual = {'ready': [100, 10], 'partial': [70, 999], 'old': [100, 1], 'unrelated': [9999, 1]}
        self.assertEqual(summarize(expected, actual), (170, 1))
