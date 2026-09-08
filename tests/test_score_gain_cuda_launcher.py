"""C002 launcher must establish its numerical contract before project imports."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_gain_cuda.py"
CONFIG = ROOT / "configs/train/campp_gain_cuda_c002.json"
SRC = ROOT / "src"
THREADS = {
    "OPENBLAS_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}


def invoke_attestation(environment, *, preload_numpy=False):
    preload = "import sys; sys.modules['numpy'] = object(); " if preload_numpy else ""
    code = (
        preload
        + "import sys; sys.path.insert(0, "
        + repr(str(SRC))
        + "); from speaker_id.infrastructure.numerical_environment import "
        + "attest_c002_preimport_thread_environment, require_c002_preimport_thread_receipt; "
        + "attest_c002_preimport_thread_environment(); require_c002_preimport_thread_receipt()"
    )
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, check=False,
    )


class C002LauncherTests(unittest.TestCase):
    def test_exact_thread_environment_passes_before_numerical_imports(self):
        environment = {**os.environ, **THREADS}
        result = invoke_attestation(environment)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_execute_entrypoint_calls_guard_before_project_imports(self):
        for key, value in (("OPENBLAS_NUM_THREADS", None), ("OMP_NUM_THREADS", "8"),
                           ("MKL_NUM_THREADS", "1")):
            with self.subTest(key=key, value=value):
                environment = {**os.environ, **THREADS}
                if value is None:
                    environment.pop(key, None)
                else:
                    environment[key] = value
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--execute"],
                    cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("before Python starts", result.stderr)

    def test_preloaded_numerical_module_is_rejected(self):
        result = invoke_attestation({**os.environ, **THREADS}, preload_numpy=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("before numerical imports", result.stderr)


if __name__ == "__main__":
    unittest.main()
