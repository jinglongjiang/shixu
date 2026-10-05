"""Legal anonymous tracking does not read actor IDs or hidden states."""

import copy
import unittest

import numpy as np

from experiments.pas_problem_discovery import GridTracks


class PaSProblemDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.coordinates = np.meshgrid(np.arange(20)*.1, np.arange(20)*.1)
        self.first = np.zeros((20, 20))
        self.first[8:10, 4:6] = 1
        self.second = np.zeros((20, 20))
        self.second[8:10, 5:7] = 1

    def test_unknown_is_not_a_measurement(self):
        tracker = GridTracks()
        tracker.update(np.full((20,20), .5), self.coordinates, 0)
        self.assertEqual(tracker.tracks, [])

    def test_velocity_comes_from_legal_centres_not_simulator_state(self):
        tracker = GridTracks()
        tracker.update(self.first, self.coordinates, 0)
        tracker.update(self.second, self.coordinates, .25)
        self.assertEqual(len(tracker.tracks), 1)
        np.testing.assert_allclose(tracker.tracks[0]["velocity"], [.4,0], atol=1e-12)
        self.assertEqual(set(tracker.tracks[0]), {"position","velocity","time","measurements"})

    def test_missing_frame_does_not_write_fake_evidence(self):
        tracker = GridTracks()
        tracker.update(self.first, self.coordinates, 0)
        before = copy.deepcopy(tracker.tracks[0])
        tracker.update(np.full((20,20), .5), self.coordinates, 1)
        self.assertEqual(tracker.tracks[0]["time"], 0)
        np.testing.assert_array_equal(tracker.tracks[0]["position"], before["position"])
        self.assertEqual(tracker.tracks[0]["measurements"], 1)
        tracker.update(np.full((20,20), .5), self.coordinates, 4.25)
        self.assertEqual(tracker.tracks, [])

    def test_history_can_only_change_currently_unknown_cells(self):
        tracker = GridTracks()
        tracker.update(self.first, self.coordinates, 0)
        tracker.update(self.second, self.coordinates, .25)
        sensor = np.full((20,20), .5)
        sensor[:10] = 0
        base = np.full((20,20), .2)
        before = copy.deepcopy(tracker.tracks)
        result = tracker.occupancy(base, sensor, self.coordinates, 1)
        np.testing.assert_array_equal(result[sensor!=.5], base[sensor!=.5])
        np.testing.assert_array_equal(base, np.full((20,20), .2))
        np.testing.assert_array_equal(tracker.tracks[0]["velocity"], before[0]["velocity"])
        self.assertEqual(tracker.tracks[0]["time"], before[0]["time"])

    def test_short_and_long_memory_are_explicitly_distinguished(self):
        tracker = GridTracks()
        tracker.update(self.first, self.coordinates, 0)
        sensor = np.full((20,20), .5)
        base = np.zeros((20,20))
        short = tracker.occupancy(base, sensor, self.coordinates, 1, ttl=.75)
        long = tracker.occupancy(base, sensor, self.coordinates, 1, ttl=4)
        self.assertEqual(short.sum(),0)
        self.assertGreater(long.sum(),0)


if __name__ == "__main__":
    unittest.main()
