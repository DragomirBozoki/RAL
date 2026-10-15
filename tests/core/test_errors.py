import pytest

from ral.core import (
    DeviceError,
    DeviceNotFoundError,
    FormatError,
    MissingDependencyError,
    RalError,
)


@pytest.mark.parametrize(
    ("cls", "builtin"),
    [
        (FormatError, ValueError),
        (DeviceError, RalError),
        (DeviceNotFoundError, LookupError),
        (DeviceNotFoundError, DeviceError),
        (MissingDependencyError, ImportError),
    ],
)
def test_errors_subclass_ral_error_and_their_builtin(
    cls: type[Exception], builtin: type[Exception]
) -> None:
    assert issubclass(cls, RalError)
    assert issubclass(cls, builtin)


def test_missing_dependency_message_says_what_and_how_to_fix() -> None:
    err = MissingDependencyError("sounddevice", "audio", "The PortAudio backend")
    assert err.package == "sounddevice"
    assert err.extra == "audio"
    assert err.install_hint == 'pip install "ral-audio[audio]"'
    assert str(err) == (
        "The PortAudio backend needs the optional package 'sounddevice', which is not "
        'installed. Install it with: pip install "ral-audio[audio]"'
    )


def test_missing_dependency_can_be_caught_as_import_error() -> None:
    with pytest.raises(ImportError, match="sounddevice"):
        raise MissingDependencyError("sounddevice", "audio", "The PortAudio backend")
