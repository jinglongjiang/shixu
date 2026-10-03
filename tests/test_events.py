import unittest

from experiments.temporal_revision_shadow import observed_event


class EventTests(unittest.TestCase):
    def test_event_uses_arrived_evidence_only(self):
        observations = [{"robot": [0, 0], "humans": [
            {"state": [1, 0, *([1, 0] if tick < 10 else [0, 1]), .3]} for _ in range(5)]}
            for tick in range(13)]
        record = {"observations": observations}
        self.assertEqual(observed_event(record, 12), 0)
        record["observations"].append({"robot": [100, 100], "humans": []})
        self.assertEqual(observed_event(record, 12), 0)

    def test_distant_motion_change_is_not_a_near_event(self):
        record = {"observations": [{"robot": [0, 0], "humans": [
            {"state": [10, 0, *([1, 0] if tick < 10 else [0, 1]), .3]} for _ in range(5)]}
            for tick in range(13)]}
        self.assertIsNone(observed_event(record, 12))


if __name__ == "__main__":
    unittest.main()
