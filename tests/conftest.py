"""Shared pytest hooks.

In GitHub Actions every failing test is also printed as an ``::error`` workflow command,
so the failure shows up as an annotation on the public run page (logs need a login).
"""

from __future__ import annotations

import os
import sys

import pytest


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if not os.environ.get("GITHUB_ACTIONS") or not report.failed:
        return
    message = str(report.longrepr).strip().splitlines()
    tail = " | ".join(line.strip() for line in message[-6:] if line.strip())
    # Workflow commands need %, CR and LF escaped.
    tail = tail.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    title = f"{report.nodeid} ({report.when})".replace(",", "%2C").replace("::", " ")
    # A workflow command must start a line; pytest -q may have printed "F" before it.
    sys.__stdout__.write(f"\n::error title={title}::{tail[:900]}\n")
    sys.__stdout__.flush()
