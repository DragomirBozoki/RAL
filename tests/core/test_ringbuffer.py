import numpy as np
import pytest

from ral.core import Anchor, FormatError, RingBuffer


def ramp(start: int, n: int, channels: int = 2) -> np.ndarray:
    """Sample k holds the value k in every channel, so any misplaced sample shows."""
    col = np.arange(start, start + n, dtype=np.float32)
    return np.repeat(col[:, None], channels, axis=1)


def test_push_and_read_across_the_wrap_point_is_exact() -> None:
    rb = RingBuffer(2, "float32", 10)
    out = np.empty((4, 2), dtype=np.float32)
    rb.push(ramp(0, 7), t_ns=0)
    assert rb.read(out) is not None
    rb.push(ramp(7, 6), t_ns=0)  # wraps: 3 slots at the end, 3 at the start
    for start in (4, 8):
        result = rb.read(out)
        assert result is not None
        assert result.start == start
        assert np.array_equal(out, ramp(start, 4))
    assert rb.available == 1
    assert rb.overruns == 0


def test_read_returns_none_and_consumes_nothing_when_short() -> None:
    rb = RingBuffer(1, "int16", 16)
    rb.push(np.zeros((3, 1), dtype=np.int16), t_ns=0)
    assert rb.read(np.empty((4, 1), dtype=np.int16)) is None
    assert rb.available == 3
    assert rb.underruns == 0  # waiting for data is not an underrun


def test_drop_oldest_discards_old_samples_and_moves_read_index() -> None:
    rb = RingBuffer(2, "float32", 8)
    rb.push(ramp(0, 6), t_ns=0)
    assert rb.push(ramp(6, 5), t_ns=0)  # 3 samples too many
    assert rb.overruns == 1
    assert rb.dropped_samples == 3
    assert rb.read_index == 3
    out = np.empty((8, 2), dtype=np.float32)
    result = rb.read(out)
    assert result is not None
    assert result.start == 3
    assert np.array_equal(out, ramp(3, 8))


def test_block_larger_than_capacity_keeps_newest_samples() -> None:
    rb = RingBuffer(2, "float32", 4)
    rb.push(ramp(0, 2), t_ns=0)
    rb.push(ramp(2, 10), t_ns=0)
    assert rb.overruns == 1
    assert rb.dropped_samples == 8  # 2 unread plus the first 6 of the new block
    assert rb.write_index == 12
    out = np.empty((4, 2), dtype=np.float32)
    result = rb.read(out)
    assert result is not None
    assert result.start == 8
    assert np.array_equal(out, ramp(8, 4))


def test_drop_newest_rejects_the_whole_block() -> None:
    rb = RingBuffer(2, "float32", 8, overrun="drop_newest")
    assert rb.push(ramp(0, 6), t_ns=0)
    assert not rb.push(ramp(6, 3), t_ns=0)
    assert rb.overruns == 1
    assert rb.dropped_samples == 3
    assert rb.write_index == 6
    out = np.empty((6, 2), dtype=np.float32)
    assert rb.read(out) is not None
    assert np.array_equal(out, ramp(0, 6))


def test_read_available_pads_with_zeros_and_counts_underrun() -> None:
    rb = RingBuffer(2, "float32", 8)
    rb.push(ramp(1, 3), t_ns=0)
    out = np.full((5, 2), 99.0, dtype=np.float32)
    assert rb.read_available(out) == 3
    assert np.array_equal(out[:3], ramp(1, 3))
    assert not out[3:].any()
    assert rb.underruns == 1
    assert rb.read_available(out) == 0
    assert rb.underruns == 2


def test_aligned_read_skips_to_the_grid() -> None:
    rb = RingBuffer(1, "float32", 32)
    rb.push(ramp(0, 30, channels=1), t_ns=0)
    out = np.empty((4, 1), dtype=np.float32)
    rb.read(np.empty((5, 1), dtype=np.float32))  # read index now 5
    result = rb.read(out, align=4)
    assert result is not None
    assert result.start == 8
    assert out[0, 0] == 8.0


def test_read_returns_newest_anchor_up_to_its_last_sample() -> None:
    rb = RingBuffer(1, "int16", 64)
    for i, t in enumerate((100, 200, 300)):
        rb.push(np.zeros((5, 1), dtype=np.int16), t_ns=t)
        assert rb.write_index == 5 * (i + 1)
    out = np.empty((8, 1), dtype=np.int16)
    first = rb.read(out)  # samples 0..7: anchors at 0 and 5
    assert first is not None
    assert first.anchor == Anchor(5, 200)
    second = rb.read(np.empty((4, 1), dtype=np.int16))  # samples 8..11: anchor at 10
    assert second is not None
    assert second.anchor == Anchor(10, 300)
    third = rb.read(np.empty((3, 1), dtype=np.int16))  # samples 12..14: no new anchor
    assert third is not None
    assert third.anchor is None


def test_anchor_fifo_drops_oldest_when_full() -> None:
    rb = RingBuffer(1, "int16", 64, max_anchors=2)
    for t in (1, 2, 3):
        rb.push(np.zeros((1, 1), dtype=np.int16), t_ns=t)
    assert rb.peek_anchor() == Anchor(1, 2)


def test_push_rejects_wrong_shape_and_dtype() -> None:
    rb = RingBuffer(2, "int16", 8)
    with pytest.raises(FormatError, match="shape"):
        rb.push(np.zeros((4, 3), dtype=np.int16), t_ns=0)
    with pytest.raises(FormatError, match="shape"):
        rb.push(np.zeros(4, dtype=np.int16), t_ns=0)
    with pytest.raises(FormatError, match="dtype"):
        rb.push(np.zeros((4, 2), dtype=np.float32), t_ns=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": 0}, "channels"),
        ({"capacity": 0}, "capacity"),
        ({"overrun": "block"}, "overrun"),
        ({"max_anchors": 0}, "max_anchors"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object], match: str) -> None:
    args: dict[str, object] = {"channels": 1, "dtype": "int16", "capacity": 8}
    args.update(kwargs)
    with pytest.raises(FormatError, match=match):
        RingBuffer(**args)  # type: ignore[arg-type]
