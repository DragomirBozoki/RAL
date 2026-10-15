import threading
import time

import numpy as np
import pytest

from ral.core import AudioFormat, AudioFrame, ChannelMap, FormatError, Framer

T0 = 1_000_000_000_000  # an arbitrary monotonic start time
MS = 1_000_000


def make_framer(rate: int = 16000, channels: int = 2, **kwargs: object) -> Framer:
    fmt = AudioFormat(rate, channels, "float32")
    return Framer(fmt, ChannelMap.uniform(channels), "test-clock", **kwargs)  # type: ignore[arg-type]


def ramp(start: int, n: int, channels: int) -> np.ndarray:
    """Sample k holds the value k, so any misplaced or lost sample shows."""
    col = np.arange(start, start + n, dtype=np.float32)
    return np.repeat(col[:, None], channels, axis=1)


def sample_time(framer: Framer, index: int) -> int:
    """When sample `index` was captured, according to the fake clock."""
    return T0 + index * 1_000_000_000 // framer.fmt.sample_rate


def push_block(framer: Framer, index: int, n: int, offset_ns: int = 0) -> int:
    framer.push(ramp(index, n, framer.fmt.channels), sample_time(framer, index) + offset_ns)
    return index + n


def drain(framer: Framer) -> list[AudioFrame]:
    frames = []
    while (frame := framer.read_frame()) is not None:
        frames.append(frame)
    return frames


def assert_frame(frame: AudioFrame, seq: int, t_ns: int, first_sample: int) -> None:
    assert frame.seq == seq
    assert frame.t_ns == t_ns
    assert np.array_equal(frame.data, ramp(first_sample, frame.fmt.frame_samples, 2))


# Deterministic, with a fake clock


@pytest.mark.parametrize(("rate", "channels"), [(16000, 2), (44100, 1), (48000, 6)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_block_sizes_give_exact_frames(rate: int, channels: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    framer = make_framer(rate, channels)
    fsamples = framer.fmt.frame_samples
    index = 0
    frames: list[AudioFrame] = []
    while index < rate:  # one second of audio
        index = push_block(framer, index, int(rng.integers(1, 3 * fsamples)))
        if rng.random() < 0.5:  # read sometimes as we go, sometimes in bursts
            frames += drain(framer)
    frames += drain(framer)

    assert len(frames) == index // fsamples
    for k, frame in enumerate(frames):
        assert frame.seq == k
        assert frame.t_ns == T0 + k * 10 * MS  # exact, no rounding drift
        assert np.array_equal(frame.data, ramp(k * fsamples, fsamples, channels))
        assert frame.clock_id == "test-clock"
    assert framer.reanchors == 0
    assert framer.discontinuities == 0


def test_callback_jitter_below_threshold_does_not_move_timestamps() -> None:
    rng = np.random.default_rng(7)
    framer = make_framer()
    index = push_block(framer, 0, 160)  # the first frame's anchor is on time
    while index < 16000:
        late_ns = int(rng.integers(0, int(1.9 * MS)))  # callbacks are late, never early
        index = push_block(framer, index, int(rng.integers(1, 400)), late_ns)
    frames = drain(framer)
    assert [f.t_ns for f in frames] == [T0 + k * 10 * MS for k in range(len(frames))]
    assert framer.reanchors == 0


def test_single_outlier_is_ignored_and_persistent_offset_reanchors() -> None:
    framer = make_framer()
    index = 0
    offsets = [0] * 10 + [5 * MS] + [0] + [5 * MS] * 4
    for offset in offsets:  # one 160-sample block, so one anchor, per frame
        index = push_block(framer, index, 160, offset)
    frames = drain(framer)

    for k in range(14):  # frame 10 alone and frames 12, 13 are not enough to move it
        assert frames[k].t_ns == T0 + k * 10 * MS
    assert frames[14].t_ns == T0 + 140 * MS + 5 * MS  # third outlier in a row
    assert frames[15].t_ns == T0 + 150 * MS + 5 * MS
    assert [f.seq for f in frames] == list(range(16))  # 5 ms is less than half a frame
    assert framer.reanchors == 1
    assert framer.discontinuities == 0


def test_forward_time_jump_skips_seq_to_match() -> None:
    framer = make_framer()
    index = 0
    for k in range(16):  # from frame 10 on, the driver says 30 ms more time has passed
        index = push_block(framer, index, 160, 30 * MS if k >= 10 else 0)
    frames = drain(framer)

    assert [f.seq for f in frames] == [*range(12), *range(15, 19)]
    assert frames[11].t_ns == T0 + 110 * MS  # outliers 1 and 2 keep the old timeline
    for frame in frames[12:]:
        assert frame.t_ns == T0 + frame.seq * 10 * MS  # seq and t_ns agree again
    assert framer.discontinuities == 1
    assert framer.lost_frames == 3


def test_backward_time_jump_reanchors_without_touching_seq() -> None:
    framer = make_framer()
    index = 0
    for k in range(8):
        index = push_block(framer, index, 160, -5 * MS if k >= 3 else 0)
    frames = drain(framer)
    assert [f.seq for f in frames] == list(range(8))
    assert frames[5].t_ns == T0 + 50 * MS - 5 * MS
    assert framer.discontinuities == 0
    assert framer.reanchors == 1


def test_overrun_jumps_seq_by_lost_frames_and_keeps_exact_time() -> None:
    framer = make_framer(capacity_s=0.05)  # 800 samples
    index = 0
    for _ in range(2):
        index = push_block(framer, index, 160)
    assert [f.seq for f in drain(framer)] == [0, 1]

    while index < 2320:  # 2000 samples while nobody reads; only the newest 800 survive
        index = push_block(framer, index, 100)
    frames = drain(framer)

    # Oldest kept sample is 2320 - 800 = 1520; reading resumes on the grid at 1600.
    assert framer.ring.overruns > 0
    assert [f.seq for f in frames] == [10, 11, 12, 13]
    for frame in frames:
        assert_frame(frame, frame.seq, T0 + frame.seq * 10 * MS, frame.seq * 160)
    assert framer.discontinuities == 1
    assert framer.lost_frames == 8  # frames 2 to 9


def test_first_frame_uses_oldest_anchor_when_fifo_overflowed() -> None:
    framer = make_framer(max_anchors=2)
    index = 0
    for _ in range(5):  # anchors for samples 0..200 are pushed out of the FIFO
        index = push_block(framer, index, 100)
    frames = drain(framer)
    assert [f.t_ns for f in frames] == [T0, T0 + 10 * MS, T0 + 20 * MS]


# A real producer thread


def test_real_producer_thread_delivers_every_sample_in_order() -> None:
    framer = make_framer()
    total = 32000  # two seconds of audio
    rng = np.random.default_rng(11)
    sizes = rng.integers(1, 500, size=total)

    def produce() -> None:
        index = 0
        for size in sizes:
            n = min(int(size), total - index)
            if n == 0:
                break
            index = push_block(framer, index, n)
            time.sleep(0.0002)

    thread = threading.Thread(target=produce)
    thread.start()
    frames: list[AudioFrame] = []
    deadline = time.monotonic() + 20
    while len(frames) < total // 160 and time.monotonic() < deadline:
        frame = framer.read_frame()
        if frame is None:
            time.sleep(0.0005)
        else:
            frames.append(frame)
    thread.join()

    assert len(frames) == total // 160
    assert [f.seq for f in frames] == list(range(total // 160))
    data = np.concatenate([f.data for f in frames])
    assert np.array_equal(data, ramp(0, total, 2))
    assert framer.ring.overruns == 0


# Arguments


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"capacity_s": 0.005}, "less than one"),
        ({"reanchor_threshold_ns": 0}, "reanchor_threshold_ns"),
        ({"reanchor_after": 0}, "reanchor_after"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(FormatError, match=match):
        make_framer(**kwargs)


def test_channel_map_must_match_format() -> None:
    with pytest.raises(FormatError, match="Channel map"):
        Framer(AudioFormat(16000, 2, "float32"), ChannelMap.uniform(1), "c")
