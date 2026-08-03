from __future__ import annotations

import unittest
from pathlib import Path

from lpap.teacher_config import load_teacher_pair_toml, project_teacher_config


class TeacherPairConfigTest(unittest.TestCase):
    def test_c128_c256_c512_shared_pair_and_projection(self) -> None:
        root = Path(__file__).resolve().parents[1]
        specs = [
            ("teacher_c128_k16.toml", "c128_k16", 128, 8, 16, 123),
            ("teacher_c256_k24.toml", "c256_k24", 256, 4, 24, 256),
            ("teacher_c512_k32.toml", "c512_k32", 512, 2, 32, 512),
        ]
        for name, pair_name, buckets, probes, k_max, perm in specs:
            path = root / "configs/training" / name
            pair = load_teacher_pair_toml(path)
            self.assertEqual(pair.name, pair_name, name)
            self.assertEqual(pair.bucket_count, buckets, name)
            self.assertEqual(pair.probe_count, probes, name)
            self.assertEqual(pair.k_max, k_max, name)
            self.assertEqual(pair.value_count, 1024, name)
            self.assertEqual(pair.permutation_seed, perm, name)
            self.assertEqual(
                pair.energy_bank.path,
                "data/encoded_energies_bank_flow_best.pt",
                name,
            )

            surrogate = project_teacher_config(path, "surrogate")
            decoder = project_teacher_config(path, "decoder")
            self.assertEqual(surrogate.data.bucket_count, buckets, name)
            self.assertEqual(decoder.data.bucket_count, buckets, name)
            self.assertEqual(surrogate.data.probe_count, probes, name)
            self.assertEqual(decoder.data.probe_count, probes, name)
            self.assertEqual(surrogate.model.k_max, k_max, name)
            self.assertEqual(surrogate.run.permutation_seed, perm, name)
            self.assertEqual(decoder.run.permutation_seed, perm, name)
            self.assertEqual(surrogate.run.steps, 15000, name)
            self.assertEqual(decoder.run.steps, 15000, name)
            self.assertEqual(
                surrogate.run.checkpoint_name, f"surrogate_{pair_name}.pt", name
            )
            self.assertEqual(
                decoder.run.checkpoint_name, f"decoder_{pair_name}.pt", name
            )
            self.assertEqual(
                decoder.teacher.checkpoint_name,
                f"surrogate_{pair_name}.pt",
                name,
            )
            self.assertEqual(
                surrogate.data.energy_bank.path,
                decoder.data.energy_bank.path,
                name,
            )


if __name__ == "__main__":
    unittest.main()
