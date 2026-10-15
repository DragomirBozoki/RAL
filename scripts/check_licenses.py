"""Fail if any installed package in a core-only environment has a non-permissive licence.

Usage (CI and locally):

    uv venv --python 3.12 .core-venv
    uv pip install --python .core-venv/<bin>/python .
    uv run pip-licenses --python .core-venv/<bin>/python --from=mixed --format=json \
        | uv run python scripts/check_licenses.py

Reads pip-licenses JSON on stdin. Every term of a licence expression (split on AND, OR,
";" and ",") must match the allowlist; any copyleft term fails the check, even if another
term would pass. Packages whose licence cannot be read are reported and fail, so that a human
looks at them and adds an explicit exception below with a reason.
"""

from __future__ import annotations

import json
import re
import sys

# Terms are matched case-insensitively as substrings of each licence term.
ALLOWED = (
    "mit",
    "bsd",  # BSD-2-Clause, BSD-3-Clause, 0BSD, "BSD License"
    "apache",
    "isc",
    "psf",
    "python software foundation",
    "zlib",  # bundled in numpy wheels
    "cc0",  # public-domain dedication, bundled in numpy wheels
)
DENIED = ("gpl", "agpl", "lgpl", "sspl", "busl", "commons clause")

# Packages checked by hand: name -> reason. Keep empty unless really needed.
EXCEPTIONS: dict[str, str] = {}

_SPLIT = re.compile(r"\s+(?:AND|OR|WITH)\s+|;|,|\(|\)", re.IGNORECASE)


def check(entries: list[dict[str, str]]) -> list[str]:
    """Return one problem string per package that is not clearly permissive."""
    problems: list[str] = []
    for entry in entries:
        name, licence = entry["Name"], entry["License"]
        if name in EXCEPTIONS:
            continue
        terms = [t.strip().lower() for t in _SPLIT.split(licence) if t and t.strip()]
        if not terms or terms == ["unknown"]:
            problems.append(f"{name}: licence unknown ({licence!r})")
        elif any(d in t for t in terms for d in DENIED):
            problems.append(f"{name}: copyleft licence {licence!r}")
        elif not all(any(a in t for a in ALLOWED) for t in terms):
            problems.append(f"{name}: licence not on the allowlist ({licence!r})")
    return problems


def main() -> int:
    entries = json.load(sys.stdin)
    problems = check(entries)
    for entry in entries:
        print(f"  {entry['Name']:<20} {entry['License']}")
    if problems:
        print("\nLicence check FAILED:", *problems, sep="\n  ")
        return 1
    print(f"\nLicence check passed for {len(entries)} packages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
