from __future__ import annotations

import unittest
from pathlib import Path

from lpap.training_notebook import training_config_from_file


class LargerKHarmonicsTeacherConfigTest(unittest.TestCase):
    def test_c128_c256_c512_kmax_and_value_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        specs = [
            ("surrogate_c128_k16.toml", "surrogate", 128, 8, 16, 123),
            ("decoder_c128_k16.toml", "decoder", 128, 8, None, 123),
            ("surrogate_c256_k24.toml", "surrogate", 256, 4, 24, 256),
            ("decoder_c256_k24.toml", "decoder", 256, 4, None, 256),
            ("surrogate_c512_k32.toml", "surrogate", 512, 2, 32, 512),
            ("decoder_c512_k32.toml", "decoder", 512, 2, None, 512),
        ]
        for name, kind, buckets, probes, k_max, perm in specs:
            cfg = training_config_from_file(root / "configs/training" / name, kind)
            self.assertEqual(cfg.data.bucket_count, buckets, name)
            self.assertEqual(cfg.data.probe_count, probes, name)
            self.assertEqual(cfg.value_count, 1024, name)
            self.assertEqual(cfg.run.permutation_seed, perm, name)
            self.assertEqual(cfg.run.steps, 15000, name)
            if kind == "surrogate":
                self.assertEqual(cfg.model.k_max, k_max, name)
            else:
                self.assertTrue(cfg.teacher.checkpoint_name.startswith("surrogate_"))


if __name__ == "__main__":
    unittest.main()
