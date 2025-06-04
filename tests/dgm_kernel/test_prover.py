import unittest
from dgm_kernel.prover import prove_patch, VerifiedPatch

class TestProver(unittest.TestCase):

    def test_prove_patch_approves_and_scores(self):
        test_id = "test_patch_123"
        test_diff = "dummy diff content"

        expected_patch = VerifiedPatch(id=test_id, diff=test_diff, score=1.0, status="APPROVED")

        actual_patch = prove_patch(id=test_id, diff=test_diff)

        self.assertEqual(actual_patch.id, expected_patch.id)
        self.assertEqual(actual_patch.diff, expected_patch.diff)
        self.assertEqual(actual_patch.score, expected_patch.score)
        self.assertEqual(actual_patch.status, expected_patch.status)

if __name__ == '__main__':
    unittest.main()
