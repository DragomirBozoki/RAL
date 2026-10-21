import asyncio
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ral.bench.synthetic import echo_path
from ral.cli.main import app
from ral.probe.analyze import estimate_drift_ppm
from ral.probe.io import SimulatedProbeIO
from ral.probe.run import ProbeSettings, probe_profile, profile_yaml, run_probe, write_report
from ral.probe.signals import deconvolve, exp_sweep, inverse_filter
from ral.profiles import parse_profile

RATE = 16000


def irs(*delays_ms: float) -> list[np.ndarray]:
    return [echo_path(d, RATE, i + 1) * 0.3 for i, d in enumerate(delays_ms)]


def probe(io: SimulatedProbeIO, **kw: object) -> object:
    return asyncio.run(run_probe(io, ProbeSettings(**kw)))  # type: ignore[arg-type]


def test_sweep_deconvolves_to_a_unit_impulse() -> None:
    sweep = exp_sweep(RATE)
    h = deconvolve(
        np.concatenate([np.zeros(80), sweep, np.zeros(RATE)]), inverse_filter(sweep, RATE)
    )
    assert int(np.argmax(np.abs(h))) == 80
    assert np.max(np.abs(h)) == pytest.approx(1.0, abs=0.02)


def test_roles_delay_and_gain_are_found() -> None:
    io = SimulatedProbeIO(RATE, ["mic", "mic", "reference", "dead"], irs(25, 31))
    result = probe(io)
    assert result.roles == ["mic", "mic", "reference", "unused"]
    mics = [c for c in result.channels if c.role == "mic"]
    assert mics[0].delay_ms == pytest.approx(25, abs=0.2)
    assert mics[1].delay_ms == pytest.approx(31, abs=0.2)
    assert mics[0].gain_db == pytest.approx(20 * np.log10(0.3), abs=1.5)
    assert "dead" in result.channels[3].note
    assert result.suggested_mode == "loopback"


def test_an_array_with_on_board_aec_is_recognised() -> None:
    io = SimulatedProbeIO(
        RATE, ["processed", "mic", "mic", "mic", "mic", "reference"], irs(20, 20, 21, 21, 22)
    )
    result = probe(io)
    assert result.roles == ["processed", "mic", "mic", "mic", "mic", "reference"]
    assert result.suggested_mode == "hardware"


def test_mics_only_suggest_software_ref_and_drift_is_measured() -> None:
    io = SimulatedProbeIO(RATE, ["mic"], irs(40), drift_ppm=500.0)
    result = probe(io, drift_s=12.0)
    assert result.roles == ["mic"]
    assert result.suggested_mode == "software_ref"
    assert result.drift_ppm == pytest.approx(500, rel=0.3)  # whole-sample steps in the sim


def test_nothing_hears_the_speaker_suggests_gate() -> None:
    io = SimulatedProbeIO(RATE, ["dead", "dead"], [])
    assert probe(io).suggested_mode == "gate"


def test_quiet_mode_plays_nothing() -> None:
    io = SimulatedProbeIO(RATE, ["mic", "dead"], irs(25))
    result = probe(io, quiet=True)
    assert io._clock_s == pytest.approx(2.0)  # only the noise floor was recorded
    assert result.roles == ["unknown", "unused"]


def test_drift_estimate() -> None:
    assert estimate_drift_ppm([0.020, 0.020002, 0.020004], [0, 10, 20]) == pytest.approx(0.2)
    assert estimate_drift_ppm([0.02], [0]) is None


def test_written_profile_is_valid_and_report_renders(tmp_path: Path) -> None:
    result = probe(SimulatedProbeIO(RATE, ["mic", "reference"], irs(25)))
    prof = probe_profile(
        result,
        robot="r",
        capture={"backend": "portaudio", "device": "x"},
        playback={"backend": "portaudio"},
    )
    text = profile_yaml(result, prof)
    assert text.startswith("# Written by ral probe")
    parsed = parse_profile(text)
    assert parsed.capture.channels == ["mic", "reference"]
    assert parsed.echo.mode == "loopback"
    page = write_report(result, tmp_path / "r.html", "probe").read_text(encoding="utf-8")
    assert "<svg" in page
    assert "loopback" in page


def test_cli_simulate_writes_profile_and_report(tmp_path: Path) -> None:
    out, report = tmp_path / "p.yaml", tmp_path / "r.html"
    result = CliRunner().invoke(
        app, ["probe", "--simulate", "--out", str(out), "--report", str(report), "--robot", "sim"]
    )
    assert result.exit_code == 0, result.output
    assert parse_profile(out.read_text(encoding="utf-8")).robot.model == "sim"
    assert report.exists()


def test_cli_asks_before_playing_sound() -> None:
    result = CliRunner().invoke(app, ["probe"], input="n\n")
    assert result.exit_code == 1
    assert "Play the sweeps?" in result.output
