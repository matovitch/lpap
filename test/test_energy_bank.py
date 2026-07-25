from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.energy_bank import (
    EnergyBankConfig,
    energy_bank_config_from_dict,
    load_energy_bank,
    resolve_energy_bank_path,
    sample_energy_bank_values,
)


class EnergyBankTest(unittest.TestCase):
    def test_config_round_trip(self) -> None:
        config = EnergyBankConfig(
            path="data/encoded_energies_ae_best_131k.pt",
            energies_key="energies",
        )
        self.assertEqual(energy_bank_config_from_dict(config.as_dict()), config)

    def test_load_dict_payload_and_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bank.pt"
            energies = torch.arange(20, dtype=torch.float32).reshape(5, 4)
            torch.save({"energies": energies, "metadata": {"n": 5}}, path)

            loaded = load_energy_bank(path)
            self.assertEqual(tuple(loaded.shape), (5, 4))
            self.assertTrue(torch.equal(loaded, energies))

            generator = torch.Generator().manual_seed(0)
            sample = sample_energy_bank_values(
                loaded,
                batch_size=3,
                generator=generator,
                device=torch.device("cpu"),
            )
            self.assertEqual(tuple(sample.shape), (3, 4))
            self.assertEqual(sample.dtype, torch.float32)

    def test_load_raw_tensor_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.pt"
            energies = torch.randn(7, 8)
            torch.save(energies, path)
            loaded = load_energy_bank(path)
            self.assertEqual(tuple(loaded.shape), (7, 8))

    def test_resolve_relative_path(self) -> None:
        root = Path("/tmp/project")
        config = EnergyBankConfig(path="data/bank.pt")
        self.assertEqual(
            resolve_energy_bank_path(root, config), root / "data" / "bank.pt"
        )

    def test_rejects_wrong_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.pt"
            torch.save({"energies": torch.zeros(4)}, path)
            with self.assertRaises(ValueError):
                load_energy_bank(path)


if __name__ == "__main__":
    unittest.main()
