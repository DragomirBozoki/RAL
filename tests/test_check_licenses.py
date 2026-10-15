import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_licenses.py"
_spec = importlib.util.spec_from_file_location("check_licenses", _PATH)
assert _spec is not None
assert _spec.loader is not None
check_licenses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_licenses)


@pytest.mark.parametrize(
    "licence",
    [
        "MIT",
        "MIT License",
        "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
        "Apache-2.0",
        "Apache Software License; MIT License",
        "ISC License (ISCL)",
        "Python Software Foundation License",
    ],
)
def test_permissive_licences_pass(licence: str) -> None:
    assert check_licenses.check([{"Name": "pkg", "License": licence}]) == []


@pytest.mark.parametrize(
    "licence",
    [
        "GPL-3.0-only",
        "LGPL-2.1-or-later",
        "MIT OR GPL-2.0",
        "GNU Affero General Public License v3 (AGPLv3)",
        "UNKNOWN",
        "Proprietary",
    ],
)
def test_non_permissive_or_unknown_licences_fail(licence: str) -> None:
    assert len(check_licenses.check([{"Name": "pkg", "License": licence}])) == 1
