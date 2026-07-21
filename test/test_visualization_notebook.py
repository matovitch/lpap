from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lpap.visualization_notebook import resolve_logged_checkpoint_path


class VisualizationNotebookTest(unittest.TestCase):
    def test_resolve_relative_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resolved = resolve_logged_checkpoint_path(
                root, "checkpoints/decoder_synthetic.pt"
            )
            self.assertEqual(resolved, root / "checkpoints" / "decoder_synthetic.pt")

    def test_resolve_missing_molab_absolute_falls_back_to_local_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = root / "checkpoints" / "decoder_synthetic.pt"
            local.parent.mkdir(parents=True)
            local.write_bytes(b"ckpt")
            resolved = resolve_logged_checkpoint_path(
                root, "/marimo/checkpoints/decoder_synthetic.pt"
            )
            self.assertEqual(resolved, local)


if __name__ == "__main__":
    unittest.main()
