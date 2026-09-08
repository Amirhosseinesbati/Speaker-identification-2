"""Stdlib-only pre-import numerical environment attestation for C002."""
from __future__ import annotations

import os
import sys


NUMERICAL_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}
_NUMERICAL_MODULE_ROOTS = ("numpy", "scipy", "torch")
_c002_preimport_receipt = None


def _numerical_module_loaded() -> bool:
    return any(
        name == root or name.startswith(root + ".")
        for name in sys.modules
        for root in _NUMERICAL_MODULE_ROOTS
    )


def attest_c002_preimport_thread_environment() -> dict:
    """Create a process-local receipt only before any numerical import."""
    global _c002_preimport_receipt
    if _numerical_module_loaded():
        raise RuntimeError("C002 numerical thread environment must be verified before numerical imports")
    observed = {name: os.environ.get(name) for name in NUMERICAL_THREAD_ENVIRONMENT}
    if observed != NUMERICAL_THREAD_ENVIRONMENT:
        raise RuntimeError(
            "C002 requires OPENBLAS_NUM_THREADS=4, OMP_NUM_THREADS=4 and "
            "MKL_NUM_THREADS=4 before Python starts"
        )
    _c002_preimport_receipt = {
        "thread_environment": dict(observed),
        "numerical_modules_absent_before_attestation": True,
    }
    return {**_c002_preimport_receipt, "thread_environment": dict(observed)}


def require_c002_preimport_thread_receipt() -> dict:
    """Bind execution to the receipt made by the official launcher."""
    current = {name: os.environ.get(name) for name in NUMERICAL_THREAD_ENVIRONMENT}
    if (
        _c002_preimport_receipt is None
        or _c002_preimport_receipt.get("numerical_modules_absent_before_attestation") is not True
        or _c002_preimport_receipt.get("thread_environment") != NUMERICAL_THREAD_ENVIRONMENT
        or current != NUMERICAL_THREAD_ENVIRONMENT
    ):
        raise RuntimeError("C002 execution lacks its pre-import numerical environment receipt")
    return {**_c002_preimport_receipt, "thread_environment": dict(current)}
