"""Simulated rooms: a robot that talks and listens in a reverberant room (DESIGN section 8).

Builds on :mod:`ral.bench.synthetic` voices but replaces the toy echo path with room
impulse responses from the image-source method (pyroomacoustics; J. B. Allen and D. A.
Berkley, "Image method for efficiently simulating small-room acoustics", JASA 1979). The
robot's speaker and its microphones share a head, so the echo is loud and early, while
the user stands a metre or two away. Echo and user paths are convolved separately, so a
test knows what each microphone heard from whom.

The capture for each echo mode is assembled from the same room:

- ``mics``: the raw microphones (``software_ref``, ``gate``, ``off``).
- ``with_loopback``: the mics plus an electrical loopback of the speaker feed on the mics'
  clock (``loopback``).
- ``hardware_array``: a processed channel standing in for an array's on-board AEC (echo
  attenuated by ``residual_db``) plus one raw mic (``hardware``).

pyroomacoustics is a test and benchmark dependency (``dev``), imported lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ral.bench.synthetic import EchoScene, Span, fft_convolve, voice
from ral.core.errors import MissingDependencyError

__all__ = ["RoomScene", "room_scene"]

Signal = npt.NDArray[np.float64]


@dataclass
class RoomScene:
    """One simulated session. ``echo`` and ``near`` are ``(samples, mics)``."""

    rate: int
    far: Signal
    echo: Signal
    near: Signal
    noise: Signal
    rt60_s: float

    @property
    def mics(self) -> Signal:
        """What each raw microphone captured."""
        out: Signal = self.echo + self.near + self.noise
        return out

    def as_echo_scene(self, mic: int = 0) -> EchoScene:
        """One microphone as an :class:`EchoScene` (for ERLE and the offline runner)."""
        return EchoScene(
            self.rate, self.far, self.echo[:, mic], self.near[:, mic], self.noise[:, mic]
        )

    def with_loopback(self, delay_ms: float = 1.0) -> tuple[Signal, list[str]]:
        """Mics plus the speaker feed as a ``reference`` channel ``delay_ms`` late."""
        d = round(delay_ms * self.rate / 1000)
        loop = np.concatenate([np.zeros(d), self.far])[: self.far.shape[0]]
        roles = ["mic"] * self.echo.shape[1] + ["reference"]
        return np.column_stack([self.mics, loop]), roles

    def hardware_array(self, residual_db: float = -45.0) -> tuple[Signal, list[str]]:
        """A processed channel (echo reduced by ``residual_db``) plus the first raw mic.

        The default of -45 dB is an assumption for on-board AEC with residual echo
        suppression, not a measured value; ``ral probe`` and ``ral bench`` measure real arrays.
        """
        processed = self.echo[:, 0] * 10 ** (residual_db / 20) + self.near[:, 0] + self.noise[:, 0]
        return np.column_stack([processed, self.mics[:, 0]]), ["processed", "mic"]


def room_scene(
    *,
    seconds: float = 20.0,
    rate: int = 16_000,
    room_m: Sequence[float] = (5.0, 4.0, 2.8),
    rt60_s: float = 0.35,
    robot_m: Sequence[float] = (2.0, 2.0, 1.2),
    user_m: Sequence[float] = (3.4, 2.6, 1.6),
    mic_spacing_m: float = 0.06,
    speaker_to_mic_m: float = 0.10,
    echo_db: float = -20.0,
    near_db: float = -32.0,
    noise_db: float = -65.0,
    far_spans: Sequence[Span] | None = None,
    near_spans: Sequence[Span] = (),
    seed: int = 1,
) -> RoomScene:
    """Simulate a robot (speaker plus two mics) and a user in a shoebox room.

    Args:
        rt60_s: Reverberation time (0.2 s is a furnished office, 0.6 s a bare hall).
        speaker_to_mic_m: Distance from the robot's speaker to its mics (below the mics).
        echo_db, near_db: RMS levels at the first mic while the robot (echo) or the user
            (near) speaks. The room sets the shape and reverberation of each path; these set
            the levels, because image-source amplitudes are not calibrated to a real
            speaker's loudness.

    Raises:
        MissingDependencyError: If pyroomacoustics is not installed.
    """
    try:
        import pyroomacoustics as pra
    except ImportError:
        raise MissingDependencyError("pyroomacoustics", "dev", "Room simulation") from None
    n = int(seconds * rate)
    absorption, max_order = pra.inverse_sabine(rt60_s, list(room_m))
    room = pra.ShoeBox(
        list(room_m), fs=rate, materials=pra.Material(absorption), max_order=max_order
    )
    rx, ry, rz = robot_m
    room.add_source([rx, ry, rz - speaker_to_mic_m])  # speaker, under the mics
    room.add_source(list(user_m))
    mics = np.array(
        [[rx - mic_spacing_m / 2, ry, rz], [rx + mic_spacing_m / 2, ry, rz]], dtype=np.float64
    ).T
    room.add_microphone_array(mics)
    room.compute_rir()

    t = np.arange(n) / rate

    def spans_mask(spans: Sequence[Span]) -> Signal:
        m = np.zeros(n)
        for a, b in spans:
            m[(t >= a) & (t < b)] = 1.0
        return m

    far = voice(n, rate, 120.0, seed) * 0.1  # -20 dBFS speaker feed
    if far_spans is not None:
        far = far * spans_mask(far_spans)
    near_src = voice(n, rate, 210.0, seed + 1) * spans_mask(near_spans)
    n_mics = mics.shape[1]
    echo = np.zeros((n, n_mics))
    near = np.zeros((n, n_mics))
    for m in range(n_mics):
        echo[:, m] = fft_convolve(far, np.asarray(room.rir[m][0], np.float64))
        near[:, m] = fft_convolve(near_src, np.asarray(room.rir[m][1], np.float64))
    echo *= _gain_to(echo[:, 0], far != 0, echo_db)
    near *= _gain_to(near[:, 0], near_src != 0, near_db)
    noise = np.random.default_rng(seed + 3).standard_normal((n, n_mics)) * 10 ** (noise_db / 20)
    return RoomScene(rate, far, echo, near, noise, rt60_s)


def _gain_to(x: Signal, active: npt.NDArray[np.bool_], level_db: float) -> float:
    if not np.any(active):
        return 1.0
    rms = float(np.sqrt(np.mean(x[active] ** 2)))
    return float(10 ** (level_db / 20) / rms) if rms > 0 else 1.0
