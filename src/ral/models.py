"""Model files RAL can download on request (``ral models fetch``).

Models are not bundled in the wheel. ``ral models fetch`` downloads a pinned file from
its official source, checks its SHA-256 and stores it in the model cache:
``$RAL_MODELS_DIR`` if set, otherwise the user cache directory (``%LOCALAPPDATA%\\ral``
on Windows, ``~/Library/Caches/ral`` on macOS, ``$XDG_CACHE_HOME/ral`` or ``~/.cache/ral``
elsewhere). Nothing is downloaded unless a user runs the fetch, and nothing is uploaded.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ral.core.errors import RalError

__all__ = ["MODELS", "ModelSpec", "fetch_model", "model_path", "models_dir"]


@dataclass(frozen=True)
class ModelSpec:
    """A pinned model file."""

    name: str
    filename: str
    url: str
    sha256: str
    size: int
    licence: str
    description: str


MODELS: dict[str, ModelSpec] = {
    "silero_vad": ModelSpec(
        name="silero_vad",
        filename="silero_vad_v6.2.3.onnx",
        url=(
            "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/"
            "src/silero_vad/data/silero_vad.onnx"
        ),
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        size=2_327_524,
        licence="MIT (Silero Team)",
        description="Silero VAD v6.2.3, ONNX, 8 and 16 kHz",
    ),
}


def models_dir() -> Path:
    """Where model files are cached."""
    override = os.environ.get("RAL_MODELS_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "ral" / "models"


def _spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise RalError(f"Unknown model {name!r}; known models: {', '.join(MODELS)}.") from None


def model_path(name: str) -> Path:
    """Path of a cached model.

    Raises:
        RalError: If the model is unknown or not downloaded yet.
    """
    spec = _spec(name)
    path = models_dir() / spec.filename
    if not path.is_file():
        raise RalError(
            f"The {spec.description} model is not downloaded. "
            f"Fetch it with: ral models fetch {name}"
        )
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(name: str, *, force: bool = False, timeout_s: float = 60.0) -> Path:
    """Download a model into the cache unless a verified copy is already there.

    Raises:
        RalError: If the download fails or the file does not match its checksum.
    """
    spec = _spec(name)
    dest = models_dir() / spec.filename
    if dest.is_file() and not force and _sha256(dest) == spec.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(spec.url, timeout=timeout_s) as r:
            while chunk := r.read(1 << 16):
                out.write(chunk)
        digest = _sha256(tmp)
        if digest != spec.sha256:
            raise RalError(
                f"Downloaded {spec.filename} has SHA-256 {digest}, expected {spec.sha256}. "
                f"The source may have changed; nothing was installed."
            )
        os.replace(tmp, dest)
    except OSError as exc:
        raise RalError(f"Could not download {spec.url}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return dest
