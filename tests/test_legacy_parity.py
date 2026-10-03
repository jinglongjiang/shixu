"""Optional local-source regression; no user-specific paths committed."""

import configparser
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from shixu.features import encode_packed
from shixu.model import ValueModel, load_weights
from shixu.policy import ValuePolicy
from shixu.runner import environment
from crowd_sim.envs.utils.state import JointState


@unittest.skipUnless(os.getenv("CAMRL_PARENT") and os.getenv("CAMRL_CHECKPOINT"), "Set original source and checkpoint paths")
class LegacyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.environ["CAMRL_PARENT"])
        from crowd_nav import contracts
        from crowd_nav.policy.mamba_rl import MambaRLPolicy
        cls.contracts = contracts
        cls.cfg = configparser.ConfigParser()
        cls.cfg.read(str(Path(__file__).resolve().parents[1] / "shixu/default.ini"))
        cls.cfg.add_section("mamba")
        for key, value in {"d_model": "256", "n_layers": "4", "d_state": "64", "d_conv": "4", "expand": "2"}.items():
            cls.cfg.set("mamba", key, value)
        contracts.init_grid_from_cfg(cls.cfg)
        cls.original = MambaRLPolicy(cls.cfg, device="cuda").eval()
        state = torch.load(os.environ["CAMRL_CHECKPOINT"], map_location="cpu")["policy_state"]
        state = {key.replace("_orig_mod.", "", 1): value for key, value in state.items()}
        cls.original.load_state_dict(state, strict=True)
        cls.original.set_phase("test")
        cls.original.use_sarl_predict = True
        cls.original.build_action_space(1.0)
        cls.cleaned = ValueModel("mamba").cuda().eval()
        load_weights(cls.cleaned, os.environ["CAMRL_CHECKPOINT"], "cuda")
        cls.policy = ValuePolicy(cls.cleaned, cls.cfg, "cuda")

    def test_packed_encoding(self):
        inputs = np.random.default_rng(9).normal(size=(30, 34)).astype(np.float32)
        np.testing.assert_allclose(encode_packed(inputs), self.contracts._batch_joint34_to_tokens_vectorized(inputs),
                                   atol=3e-6, rtol=1e-6)

    def test_value_output(self):
        torch.manual_seed(91)
        x = torch.randn(4, 24, 8, 13, device="cuda")
        with torch.no_grad():
            np.testing.assert_allclose(self.cleaned(x).cpu().numpy(), self.original.forward_value(x).cpu().numpy(),
                                       atol=2e-6, rtol=2e-6)

    def test_native_actions_and_history(self):
        env = environment(self.cfg, self.policy)
        self.policy.reset()
        self.original.reset_episode_stats()
        env.reset(options={"test_case": 0})
        for _ in range(30):
            state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            original = self.original.predict(state)
            cleaned = self.policy.predict(state)
            np.testing.assert_allclose([cleaned.vx, cleaned.vy], [original.vx, original.vy], atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(self.policy.history), np.asarray(self.original._history), atol=3e-6)
            env.step(original)


if __name__ == "__main__":
    unittest.main()
