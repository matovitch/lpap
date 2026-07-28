from __future__ import annotations

import unittest
from pathlib import Path

from lpap.training_notebook import training_config_from_file


class C512TeacherConfigTest(unittest.TestCase):
    def test_surrogate_and_decoder_use_energy_bank_prior(self) -> None:
        root = Path(__file__).resolve().parents[1]
        surr = training_config_from_file(
            root / "configs/training/surrogate_c512.toml", "surrogate"
        )
        dec = training_config_from_file(
            root / "configs/training/decoder_c512.toml", "decoder"
        )
        self.assertEqual(surr.data.bucket_count, 512)
        self.assertEqual(surr.data.probe_count, 2)
        self.assertEqual(surr.value_count, 1024)
        self.assertEqual(surr.data.kind, "energy_bank")
        assert surr.data.energy_bank is not None
        self.assertEqual(
            surr.data.energy_bank.path, "data/encoded_energies_ae_best.pt"
        )
        self.assertEqual(surr.run.checkpoint_name, "surrogate_c512.pt")
        self.assertEqual(dec.data.bucket_count, 512)
        self.assertEqual(dec.data.probe_count, 2)
        self.assertEqual(dec.value_count, 1024)
        self.assertEqual(dec.data.kind, "energy_bank")
        self.assertEqual(dec.teacher.checkpoint_name, "surrogate_c512.pt")
        self.assertEqual(dec.run.checkpoint_name, "decoder_c512.pt")


if __name__ == "__main__":
    unittest.main()
