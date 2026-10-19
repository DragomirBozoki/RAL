import hashlib
import io
from pathlib import Path

import pytest

from ral import models
from ral.core import RalError


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> bytes:
    payload = b"not really an onnx model"
    spec = models.ModelSpec(
        name="toy",
        filename="toy.onnx",
        url="https://example.invalid/toy.onnx",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        licence="MIT",
        description="toy model",
    )
    monkeypatch.setitem(models.MODELS, "toy", spec)
    monkeypatch.setenv("RAL_MODELS_DIR", str(tmp_path / "cache"))
    return payload


def serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    calls: list[str] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    return calls


def test_fetch_verifies_and_caches(monkeypatch: pytest.MonkeyPatch, fake_model: bytes) -> None:
    calls = serve(monkeypatch, fake_model)
    path = models.fetch_model("toy")
    assert path.read_bytes() == fake_model
    assert models.model_path("toy") == path
    models.fetch_model("toy")
    assert len(calls) == 1  # a verified copy is reused


def test_checksum_mismatch_installs_nothing(
    monkeypatch: pytest.MonkeyPatch, fake_model: bytes
) -> None:
    serve(monkeypatch, b"tampered")
    with pytest.raises(RalError, match="SHA-256"):
        models.fetch_model("toy")
    assert not any(models.models_dir().iterdir())


def test_unknown_and_missing_models(fake_model: bytes) -> None:
    with pytest.raises(RalError, match="Unknown model"):
        models.model_path("nope")
    with pytest.raises(RalError, match="ral models fetch toy"):
        models.model_path("toy")


def test_models_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAL_MODELS_DIR", str(tmp_path))
    assert models.models_dir() == tmp_path
