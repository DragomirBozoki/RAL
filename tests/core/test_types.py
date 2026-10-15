import numpy as np
import pytest

from ral.core import (
    AudioFormat,
    AudioFrame,
    ChannelMap,
    ChannelRole,
    DeviceInfo,
    FormatError,
    RalError,
)

MIC = ChannelRole.MIC
REF = ChannelRole.REFERENCE


def make_frame(fmt: AudioFormat, **overrides: object) -> AudioFrame:
    kwargs: dict[str, object] = {
        "data": np.zeros((fmt.frame_samples, fmt.channels), dtype=fmt.dtype),
        "fmt": fmt,
        "channel_map": ChannelMap.uniform(fmt.channels),
        "t_ns": 0,
        "seq": 0,
        "clock_id": "test",
    }
    kwargs.update(overrides)
    return AudioFrame(**kwargs)  # type: ignore[arg-type]


# AudioFormat


@pytest.mark.parametrize(
    ("rate", "samples"), [(8000, 80), (16000, 160), (44100, 441), (48000, 480)]
)
def test_frame_samples_for_common_rates(rate: int, samples: int) -> None:
    assert AudioFormat(rate, 1, "float32").frame_samples == samples


def test_frame_ns_is_frame_length_in_nanoseconds() -> None:
    assert AudioFormat(16000, 1, "int16").frame_ns == 10_000_000
    assert AudioFormat(16000, 1, "int16", frame_ms=20).frame_ns == 20_000_000


def test_numpy_dtype_matches_dtype_name() -> None:
    assert AudioFormat(16000, 1, "int16").numpy_dtype == np.int16
    assert AudioFormat(16000, 1, "float32").numpy_dtype == np.float32


@pytest.mark.parametrize("rate", [22050, 11025, 16001])
def test_rate_without_whole_samples_per_frame_is_rejected(rate: int) -> None:
    with pytest.raises(FormatError, match="divisible by 100"):
        AudioFormat(rate, 1, "int16")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"channels": 0}, "channels"),
        ({"dtype": "float64"}, "dtype"),
        ({"frame_ms": 0}, "frame_ms"),
    ],
)
def test_invalid_format_fields_are_rejected(kwargs: dict[str, object], match: str) -> None:
    args: dict[str, object] = {"sample_rate": 16000, "channels": 1, "dtype": "int16"}
    args.update(kwargs)
    with pytest.raises(FormatError, match=match):
        AudioFormat(**args)  # type: ignore[arg-type]


def test_format_error_is_a_ral_error_and_a_value_error() -> None:
    with pytest.raises(RalError):
        AudioFormat(16000, 0, "int16")
    with pytest.raises(ValueError, match="channels"):
        AudioFormat(16000, 0, "int16")


# ChannelMap


def test_from_names_builds_roles_in_order() -> None:
    cmap = ChannelMap.from_names(["mic", "mic", "reference", "unused"])
    assert cmap.roles == (MIC, MIC, REF, ChannelRole.UNUSED)
    assert cmap.channels == 4


def test_from_names_rejects_unknown_role_and_lists_valid_ones() -> None:
    with pytest.raises(FormatError, match="valid roles are mic, reference"):
        ChannelMap.from_names(["mic", "speaker"])


def test_indices_returns_channels_with_role() -> None:
    cmap = ChannelMap((MIC, REF, MIC, REF))
    assert cmap.indices(MIC) == (0, 2)
    assert cmap.indices(REF) == (1, 3)
    assert cmap.indices(ChannelRole.PROCESSED) == ()


def test_uniform_repeats_role() -> None:
    assert ChannelMap.uniform(3).roles == (MIC, MIC, MIC)
    assert ChannelMap.uniform(2, REF).roles == (REF, REF)
    with pytest.raises(FormatError):
        ChannelMap.uniform(0)


def test_empty_map_is_rejected() -> None:
    with pytest.raises(FormatError, match="at least one"):
        ChannelMap(())


def test_string_roles_are_rejected() -> None:
    with pytest.raises(FormatError, match="from_names"):
        ChannelMap(("mic",))  # type: ignore[arg-type]


def test_positions_must_match_channels() -> None:
    ChannelMap((MIC, MIC, REF), ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), None))
    with pytest.raises(FormatError, match="one position"):
        ChannelMap((MIC, MIC), ((0.0, 0.0, 0.0),))
    with pytest.raises(FormatError, match="x, y and z"):
        ChannelMap((MIC,), ((0.0, 0.0),))  # type: ignore[arg-type]


# AudioFrame


def test_valid_frame_is_accepted() -> None:
    fmt = AudioFormat(48000, 2, "int16")
    frame = make_frame(fmt, t_ns=123, seq=7, channel_map=ChannelMap((MIC, REF)))
    assert frame.data.shape == (480, 2)
    assert frame.t_ns == 123
    assert frame.seq == 7


def test_frame_with_wrong_shape_is_rejected() -> None:
    fmt = AudioFormat(16000, 2, "float32")
    with pytest.raises(FormatError, match=r"shape \(159, 2\)"):
        make_frame(fmt, data=np.zeros((159, 2), dtype=np.float32))
    with pytest.raises(FormatError, match="shape"):
        make_frame(fmt, data=np.zeros(320, dtype=np.float32))  # flat, not (samples, channels)


def test_frame_with_wrong_dtype_is_rejected() -> None:
    fmt = AudioFormat(16000, 1, "float32")
    with pytest.raises(FormatError, match="dtype"):
        make_frame(fmt, data=np.zeros((160, 1), dtype=np.float64))


def test_frame_with_mismatched_channel_map_is_rejected() -> None:
    fmt = AudioFormat(16000, 2, "int16")
    with pytest.raises(FormatError, match="Channel map has 3 roles"):
        make_frame(fmt, channel_map=ChannelMap.uniform(3))


def test_frame_with_negative_seq_is_rejected() -> None:
    with pytest.raises(FormatError, match="seq"):
        make_frame(AudioFormat(16000, 1, "int16"), seq=-1)


# DeviceInfo


def test_device_info_is_frozen_and_hashable() -> None:
    dev = DeviceInfo(
        id="file:in.wav",
        name="in.wav",
        backend="file",
        host_api=None,
        max_input_channels=2,
        max_output_channels=0,
        default_sample_rate=16000.0,
        extra={"path": "in.wav"},
    )
    assert {dev: 1}[dev] == 1
    with pytest.raises(AttributeError):
        dev.name = "other"  # type: ignore[misc]
