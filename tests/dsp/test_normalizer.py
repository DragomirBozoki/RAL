import numpy as np
import pytest

from ral.core import AudioFormat, AudioFrame, ChannelMap, ChannelRole, FormatError
from ral.dsp.normalizer import Normalizer

T0 = 5_000_000_000
FRAME_NS = 10_000_000


def native_frames(
    rate: int, channels: int, frames: int, dtype: str = "int16", seqs: list[int] | None = None
) -> list[AudioFrame]:
    fmt = AudioFormat(rate, channels, dtype)  # type: ignore[arg-type]
    cmap = ChannelMap.uniform(channels)
    n = fmt.frame_samples
    t = np.arange(frames * n) / rate
    x = 0.5 * np.sin(2 * np.pi * 1000 * t)
    seqs = seqs if seqs is not None else list(range(frames))
    out = []
    for i, seq in enumerate(seqs):
        block = np.repeat(x[i * n : (i + 1) * n, None], channels, axis=1)
        data = (block * 32767).astype(np.int16) if dtype == "int16" else block.astype(np.float32)
        out.append(AudioFrame(data, fmt, cmap, T0 + seq * FRAME_NS, seq, "dev"))
    return out


def run(norm: Normalizer, frames: list[AudioFrame]) -> list[AudioFrame]:
    out = []
    for f in frames:
        out.extend(norm.process(f))
    out.extend(norm.flush())
    return out


@pytest.mark.parametrize("rate", [8000, 16000, 32000, 44100, 48000, 96000])
def test_one_output_frame_per_input_frame_with_same_time_and_seq(rate: int) -> None:
    frames = native_frames(rate, 2, 25)
    out = run(Normalizer(frames[0].fmt, frames[0].channel_map), frames)
    assert [f.seq for f in out] == list(range(25))
    assert [f.t_ns for f in out] == [T0 + k * FRAME_NS for k in range(25)]
    for f in out:
        assert f.fmt == AudioFormat(16000, 2, "float32")
        assert f.data.shape == (160, 2)


def test_output_is_available_at_most_one_frame_late() -> None:
    frames = native_frames(48000, 1, 10)
    norm = Normalizer(frames[0].fmt, frames[0].channel_map)
    produced = [len(norm.process(f)) for f in frames]
    assert produced[0] == 0
    assert sum(produced) == 9


def test_resampled_signal_matches_the_ideal_tone() -> None:
    frames = native_frames(48000, 1, 50, dtype="float32")
    out = run(Normalizer(frames[0].fmt, frames[0].channel_map), frames)
    y = np.concatenate([f.data[:, 0] for f in out])
    ideal = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(len(y)) / 16000)
    err = y[800:-800] - ideal[800:-800]
    assert np.sqrt(np.mean(err**2)) < 1e-3


def test_same_rate_int16_converts_exactly() -> None:
    frames = native_frames(16000, 1, 3)
    out = run(Normalizer(frames[0].fmt, frames[0].channel_map), frames)
    for i, o in zip(frames, out, strict=True):
        np.testing.assert_array_equal(o.data, i.data.astype(np.float32) / 32768)


def test_unused_channels_are_dropped_and_roles_kept() -> None:
    fmt = AudioFormat(16000, 4, "float32")
    cmap = ChannelMap.from_names(["mic", "unused", "reference", "mic"])
    data = np.tile(np.array([1, 2, 3, 4], np.float32) / 10, (160, 1))
    norm = Normalizer(fmt, cmap)
    (out,) = norm.process(AudioFrame(data, fmt, cmap, T0, 0, "dev"))
    assert out.channel_map.roles == (ChannelRole.MIC, ChannelRole.REFERENCE, ChannelRole.MIC)
    np.testing.assert_allclose(out.data[0], [0.1, 0.3, 0.4])


def test_gap_flushes_and_keeps_the_seq_hole() -> None:
    frames = native_frames(48000, 1, 8, seqs=[0, 1, 2, 3, 6, 7, 8, 9])
    norm = Normalizer(frames[0].fmt, frames[0].channel_map)
    out = run(norm, frames)
    assert [f.seq for f in out] == [0, 1, 2, 3, 6, 7, 8, 9]
    assert [f.t_ns for f in out] == [f.t_ns for f in frames]
    assert norm.gaps == 1


def test_mismatched_format_or_map_is_rejected() -> None:
    fmt = AudioFormat(16000, 2, "int16")
    with pytest.raises(FormatError):
        Normalizer(fmt, ChannelMap.uniform(3))
    with pytest.raises(FormatError):
        Normalizer(fmt, ChannelMap.uniform(2, ChannelRole.UNUSED))
    norm = Normalizer(fmt, ChannelMap.uniform(2))
    other = native_frames(48000, 2, 1)[0]
    with pytest.raises(FormatError):
        norm.process(other)
