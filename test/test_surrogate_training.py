from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lpap.surrogate_training import (
    LPAPSurrogateTrainingConfig,
    create_lpap_surrogate_training_session,
    iter_lpap_surrogate_training,
)


class LPAPSurrogateTrainingTest(unittest.TestCase):
    def test_session_trains_and_logs_small_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LPAPSurrogateTrainingConfig(
                steps=2,
                batch_size=2,
                bucket_count=4,
                probe_count=4,
                k_max=2,
                harmonic_count=3,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
                checkpoint_every=2,
                display_every=1,
                run_id="tiny-surrogate",
            )
            session = create_lpap_surrogate_training_session(
                project_root=Path(temp_dir), config=config, device="cpu"
            )

            results = list(iter_lpap_surrogate_training(session))

            self.assertEqual(len(results), 2)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("loss", results[-1].metrics)


if __name__ == "__main__":
    unittest.main()
