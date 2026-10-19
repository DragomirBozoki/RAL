from collections.abc import Iterable

from ral.devices.doctor import DoctorReport, Level
from ral.devices.doctor.host import HostScanner


def linux(paths: Iterable[str], accessible: bool = True, runtime: str = "/run/user/1000"):
    present = set(paths)
    return HostScanner(
        system="Linux",
        platform_name="Linux-6.8-aarch64",
        env={"XDG_RUNTIME_DIR": runtime} if runtime else {},
        exists=lambda p: p in present,
        accessible=lambda p: accessible,
        clock_name="monotonic",
        clock_resolution_ns=1,
    )


def scan(scanner: HostScanner) -> list[tuple[Level, str, str | None]]:
    report = DoctorReport()
    scanner.scan(report)
    assert all(f.area == "host" for f in report.findings)
    return [(f.level, f.message, f.hint) for f in report.findings]


def test_reports_platform_python_and_fine_clock() -> None:
    findings = scan(linux(["/dev/snd"]))
    assert findings[0][0] is Level.INFO
    assert findings[0][1].startswith("Linux-6.8-aarch64, Python 3.")
    assert findings[1] == (Level.OK, "RAL clock: time.monotonic, resolution 1 ns", None)


def test_coarse_clock_is_a_warning() -> None:
    scanner = HostScanner(system="Windows", clock_name="monotonic", clock_resolution_ns=15_625_000)
    level, message, _ = scan(scanner)[1]
    assert level is Level.WARN
    assert "15.625 ms" in message


def test_windows_skips_linux_checks() -> None:
    scanner = HostScanner(system="Windows", clock_name="perf_counter", clock_resolution_ns=100)
    findings = scan(scanner)
    assert len(findings) == 2
    assert findings[1][1] == "RAL clock: QueryPerformanceCounter, resolution 100 ns"


def test_missing_dev_snd_points_to_ros2_and_containers() -> None:
    level, message, hint = scan(linux([]))[2]
    assert level is Level.WARN
    assert "/dev/snd is missing" in message
    assert hint is not None
    assert "ROS 2" in hint
    assert "--device /dev/snd" in hint


def test_unreadable_dev_snd_suggests_audio_group() -> None:
    level, _, hint = scan(linux(["/dev/snd"], accessible=False))[2]
    assert level is Level.ERROR
    assert hint is not None
    assert "usermod -aG audio" in hint


def test_sound_servers() -> None:
    both = ["/dev/snd", "/run/user/1000/pipewire-0", "/run/user/1000/pulse/native"]
    assert scan(linux(both))[3][1] == "PipeWire is running with its PulseAudio service."
    assert scan(linux(["/dev/snd", "/run/user/1000/pipewire-0"]))[3][1] == "PipeWire is running."
    assert (
        scan(linux(["/dev/snd", "/run/user/1000/pulse/native"]))[3][1] == "PulseAudio is running."
    )
    level, message, _ = scan(linux(["/dev/snd"], runtime=""))[3]
    assert level is Level.INFO
    assert "uses ALSA directly" in message
