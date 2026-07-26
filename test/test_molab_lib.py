from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from marimo._ast.load import load_app

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "skills"
    / "molab-workflow"
    / "scripts"
)


class MolabLibTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys

        sys.path.insert(0, str(SCRIPTS))
        from molab_lib import (  # noqa: WPS433
            count_unnamed_cells,
            generate_molab_lab_source,
            repo_root_from_script,
            resolve_lpap_package_file,
        )

        cls.count_unnamed_cells = staticmethod(count_unnamed_cells)
        cls.generate_molab_lab_source = staticmethod(generate_molab_lab_source)
        cls.repo_root_from_script = staticmethod(repo_root_from_script)
        cls.resolve_lpap_package_file = staticmethod(resolve_lpap_package_file)

    def test_resolve_training_plots(self) -> None:
        root = self.repo_root_from_script(SCRIPTS / "molab_lib.py")
        path, module, rel = self.resolve_lpap_package_file(
            "src/lpap/training_plots.py", repo_root=root
        )
        self.assertTrue(path.is_file())
        self.assertEqual(module, "lpap.training_plots")
        self.assertEqual(rel, Path("training_plots.py"))

        path2, module2, _rel2 = self.resolve_lpap_package_file(
            "training_plots.py", repo_root=root
        )
        self.assertEqual(path2, path)
        self.assertEqual(module2, module)

    def test_resolve_rejects_outside_package(self) -> None:
        root = self.repo_root_from_script(SCRIPTS / "molab_lib.py")
        with self.assertRaises(FileNotFoundError):
            self.resolve_lpap_package_file("README.md", repo_root=root)

    def test_generate_notebook_roundtrip(self) -> None:
        cells = [
            {
                "name": "ae_setup",
                "code": (
                    "# cell: ae_setup\n"
                    "import marimo as mo\n"
                    "from pathlib import Path\n"
                    "project_root = Path('/marimo')\n"
                    "ae_base = 1\n"
                ),
                "hide_code": False,
            },
            {
                "name": "gallery_gamma",
                "code": (
                    "# cell: gallery_gamma\n"
                    "display_gamma = mo.ui.slider(0.2, 2.0, value=1.0)\n"
                    "display_gamma\n"
                ),
                "hide_code": False,
            },
            {
                "name": "gallery_view",
                "code": (
                    "# cell: gallery_view\n"
                    "mo.md(\n"
                    "    f'γ={display_gamma.value} root={project_root} base={ae_base}'\n"
                    ")\n"
                ),
                "hide_code": False,
            },
        ]
        source = self.generate_molab_lab_source(cells, width="medium")
        self.assertIn('app = marimo.App(width="medium")', source)
        self.assertIn("def ae_setup():", source)
        self.assertIn("def gallery_gamma(mo):", source)
        self.assertIn("def gallery_view(", source)
        self.assertEqual(self.count_unnamed_cells(cells), 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lab.py"
            path.write_text(source)
            app = load_app(str(path))
            names = [
                app._cell_manager.cell_data_at(cid).name
                for cid in app._cell_manager.cell_ids()
            ]
            self.assertEqual(names, ["ae_setup", "gallery_gamma", "gallery_view"])


if __name__ == "__main__":
    unittest.main()
