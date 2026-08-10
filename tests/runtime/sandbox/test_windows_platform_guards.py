# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the non-POSIX platform guards in the sandbox.

The sandbox only *enforces* anything on Linux, but it must stay importable and
constructible everywhere: ``probe_capabilities()`` is what tells the executor to
fail closed, so if the probe itself raises, the caller gets an ImportError
instead of a clean "sandbox requires Linux". Likewise ``_classify_worker_death``
runs on the error path, where an AttributeError would mask the real failure.

These tests simulate a non-POSIX host by removing the POSIX-only names, so they
exercise the Windows code paths while running on Linux/macOS CI. They carry no
``sandbox`` marker on purpose — they fork nothing and need no kernel features,
so they belong in the default unit-test job on every platform.
"""

from __future__ import annotations

import signal

import pytest

from nooa.runtime.sandbox import guards
from nooa.runtime.sandbox.errors import CellMemoryError, CellTimeoutError, WorkerDiedError
from nooa.runtime.sandbox.executor import SandboxedExecutor


class _FakeProc:
    """Stand-in for the worker ``multiprocessing.Process``."""

    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode


def _executor_with_exitcode(exitcode: int | None) -> SandboxedExecutor:
    """A SandboxedExecutor wired to a dead worker, without starting one.

    ``_classify_worker_death`` reads only ``self._proc``, so bypassing
    ``__init__`` keeps this a pure unit test of the classification logic.
    """
    executor = object.__new__(SandboxedExecutor)
    executor._proc = _FakeProc(exitcode)
    return executor


# --- probe_capabilities: no `resource` module -------------------------------
def test_probe_capabilities_survives_missing_resource_module(monkeypatch):
    """Windows has no `resource`; the probe must report rlimit=False, not raise."""
    monkeypatch.setattr(guards, "resource", None)

    caps = guards.probe_capabilities()

    assert caps.rlimit is False


def test_probe_capabilities_reports_rlimit_when_resource_present():
    """The paired positive case, so the test above cannot pass vacuously."""
    if guards.resource is None:  # pragma: no cover - POSIX-only assertion
        pytest.skip("host has no 'resource' module")

    assert guards.probe_capabilities().rlimit is True


def test_apply_rlimits_refuses_rather_than_silently_skipping(monkeypatch):
    """Without `resource`, the guard must fail loudly — never no-op."""
    monkeypatch.setattr(guards, "resource", None)

    with pytest.raises(RuntimeError, match="resource"):
        guards.apply_rlimits(max_memory_mb=128)


# --- _classify_worker_death: no POSIX signal constants ----------------------
def test_classify_worker_death_survives_missing_signal_constants(monkeypatch):
    """Windows lacks SIGXCPU/SIGKILL; classification must degrade, not crash."""
    monkeypatch.delattr(signal, "SIGXCPU", raising=False)
    monkeypatch.delattr(signal, "SIGKILL", raising=False)
    original = WorkerDiedError("worker exited")

    result = _executor_with_exitcode(1)._classify_worker_death(original)

    # The original error survives — an AttributeError here would replace the
    # real cause of death with a bug in the error handler.
    assert result is original


def test_classify_worker_death_maps_sigxcpu_to_timeout():
    """SIGXCPU still classifies as a CPU-time kill where the signal exists."""
    if not hasattr(signal, "SIGXCPU"):  # pragma: no cover - POSIX-only assertion
        pytest.skip("host has no SIGXCPU")

    result = _executor_with_exitcode(-signal.SIGXCPU)._classify_worker_death(
        WorkerDiedError("worker exited")
    )

    assert isinstance(result, CellTimeoutError)


def test_classify_worker_death_maps_sigkill_to_memory():
    """SIGKILL still classifies as an OOM/resource kill where the signal exists."""
    if not hasattr(signal, "SIGKILL"):  # pragma: no cover - POSIX-only assertion
        pytest.skip("host has no SIGKILL")

    result = _executor_with_exitcode(-signal.SIGKILL)._classify_worker_death(
        WorkerDiedError("worker exited")
    )

    assert isinstance(result, CellMemoryError)


def test_classify_worker_death_passes_through_a_live_worker():
    """exitcode None (worker still running) must not match any signal branch."""
    original = WorkerDiedError("worker exited")

    assert _executor_with_exitcode(None)._classify_worker_death(original) is original
