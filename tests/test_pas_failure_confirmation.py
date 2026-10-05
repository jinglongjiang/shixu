"""Failure enrichment and protection selection use parent fields only."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.pas_failure_confirmation import native_record, parent_fields, rank_configs, save, select_roots


def row(index, failure=False, people=20, hidden=True):
    return dict(index=index, record=dict(people=people, geometry="square_crossing", case=95000+index,
                terminal="Collision" if failure else "ReachGoal",
                counts={"seen_hidden2s": int(hidden)}))


class PaSFailureConfirmationTests(unittest.TestCase):
    def test_method_outcomes_cannot_change_configuration_ranking(self):
        records = [row(0, True)["record"], row(1)["record"], row(2, True, 10, False)["record"]]
        before = rank_configs([parent_fields(r) for r in records])
        changed = copy.deepcopy(records)
        for record in changed:
            record["arms"] = {"long_cv": {"terminal": "ReachGoal", "return_": 1000}}
        self.assertEqual(before, rank_configs([parent_fields(r) for r in changed]))
        self.assertEqual(before[0]["people"], 20)
        self.assertEqual(before[0]["joint_failures"], 1)

    def test_first_eight_failure_prefix_excludes_inflight_extras(self):
        rows = [row(i, failure=i % 4 == 0) for i in range(32)]
        selected = select_roots(rows)
        self.assertEqual(selected["failures"], list(range(0, 29, 4)))
        self.assertEqual(selected["cutoff"], 28)
        self.assertEqual(selected["in_flight_excluded"], [29, 30, 31])
        self.assertEqual(selected["protection"], [1, 2, 3, 5, 6, 7, 9, 10])

    def test_nonhidden_failure_is_not_an_opportunity(self):
        selected = select_roots([row(0, True, hidden=False), row(1, True), row(2)])
        self.assertEqual(selected["failures"], [1])
        self.assertEqual(selected["protection"], [2])

    def test_protection_configuration_matches_failure_quota(self):
        rows = [row(0, True, 10), row(1, True, 20)]
        rows += [row(i, i in range(14, 20), 10 if i % 2 == 0 else 20) for i in range(2, 20)]
        selected = select_roots(rows)
        for people in (10, 20):
            failures = sum(rows[i]["record"]["people"] == people for i in selected["failures"])
            protection = sum(rows[i]["record"]["people"] == people for i in selected["protection"])
            self.assertEqual(failures, protection)

    def test_cap_without_failures_keeps_original_queue(self):
        rows = [row(i) for i in range(160)]
        selected = select_roots(rows)
        self.assertEqual(selected["cutoff"], 159)
        self.assertEqual(selected["failures"], [])
        self.assertEqual(selected["protection"], list(range(8)))

    def test_atomic_json_preserves_numpy_scalar_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            value = np.float32(.125)
            save(path, {"value": value})
            self.assertEqual(json.loads(path.read_text()), {"value": float(value)})
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            with self.assertRaises(RuntimeError):
                save(path, {"value": 0})

    def test_json_replay_restores_native_action_dtype_without_changing_values(self):
        values = np.asarray([[3.709473, -.601739]], dtype=np.float32).tolist()
        record = {"actions": values, "rewards": [0.]}
        restored = native_record(record)
        self.assertEqual(restored["actions"][0].dtype, np.float32)
        np.testing.assert_array_equal(restored["actions"], values)
        self.assertEqual(record["actions"], values)
        self.assertIs(restored["rewards"], record["rewards"])


if __name__ == "__main__":
    unittest.main()
