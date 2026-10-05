"""Array contracts for the bounded external response audit."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.interaction_response_audit import logged_state, physical_state, save


class InteractionResponseAuditTests(unittest.TestCase):
    def test_logged_current_column_excludes_goals_and_future(self):
        values = np.arange(18*3).reshape(18, 3)
        expected = np.r_[values[:4, 0], values[6:10, 0], values[12:16, 0]]
        np.testing.assert_array_equal(logged_state(values, 2), expected)
        changed = values.copy()
        changed[:, 1:] = -99
        changed[4:6, 0] = -99
        changed[10:12, 0] = -99
        changed[16:18, 0] = -99
        np.testing.assert_array_equal(logged_state(changed, 2), expected)

    def test_physical_state_preserves_signed_speed(self):
        robot = SimpleNamespace(px=1., py=2., theta=0., vx=-.5, vy=0.)
        human = SimpleNamespace(px=3., py=4., vx=.1, vy=.2)
        env = SimpleNamespace(robot=robot, humans=[human])
        np.testing.assert_array_equal(physical_state(env), [1., 2., 0., -.5, 3., 4., .1, .2])

    def test_results_cannot_overwrite_frozen_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            save(path, {"changed": 0})
            with self.assertRaises(RuntimeError):
                save(path, {"changed": 1})
            self.assertEqual(json.loads(path.read_text()), {"changed": 0})


if __name__ == "__main__":
    unittest.main()
