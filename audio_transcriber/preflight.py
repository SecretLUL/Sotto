"""Checks made before a recording or an upload starts.

Everything below used to come to light only after the recording: no API key for
the cloud engine, a 3 GB model that still had to be downloaded, an output folder
that cannot be written, a disk that filled up half way through a meeting. The
recorded tracks survive such a failure (see capture.find_unfinished), but nobody
wants to learn at the end of a meeting that there is no transcript yet.

Each finding is an ERROR (the run would fail: do not start), a WARNING (it might
- ask first) or a NOTE (it works, it just takes a little longer).
"""

import os
import shutil
from dataclasses import dataclass

from . import paths
from .transcribe import binaries

ERROR, WARNING, NOTE = "error", "warning", "note"
_ORDER = {ERROR: 0, WARNING: 1, NOTE: 2}

# One hour of recording takes about this much disk while it is made and
# processed: two raw tracks (mono, 16 bit, typically 48 kHz: 0.35 GB each), the
# 16 kHz copies made for recognition (0.12 GB each) and the saved mixdown (mono
# at the rate of the raw tracks: 0.35 GB).
BYTES_PER_RECORDED_HOUR = int(1.2 * 2 ** 30)

# Below this a recording is not worth starting (about twenty minutes) ...
MIN_FREE_BYTES = 300 * 10 ** 6
# ... and below room for this many hours the user is told how long it lasts.
WARN_BELOW_HOURS = 2.0

# A download needs a little more than the file: the partial copy is verified
# before it replaces anything.
DOWNLOAD_HEADROOM = 1.05


@dataclass(frozen=True)
class Finding:
    level: str
    text: str


def check(settings, recording=True):
    """What is wrong, or worth knowing, before this run starts.

    `settings` is the copy the run is going to use. `recording` is False for an
    upload or a recovered recording: they need no room for a long recording and
    have no live text. Errors come first.
    """
    out_dir = settings.get_output_dir()
    findings = (_engine(settings, recording)
                + _output_folder(out_dir)
                + (_recording_space(out_dir) if recording else []))
    return sorted(findings, key=lambda finding: _ORDER[finding.level])


# ----------------------------------------------------------------------
def _engine(settings, recording):
    cloud = settings.uses_cloud()
    if cloud and not (settings.api_key or "").strip():
        return [Finding(ERROR,
                        "ElevenLabs is chosen, but no API key is set. Enter it on "
                        "the Settings tab, set the ELEVENLABS_API_KEY environment "
                        "variable, or choose a local model.")]

    # The models this run will load. The closing pass uses the chosen one (not
    # with the cloud); the live text and the preview, whichever of them runs,
    # use the small model - for a cloud run as well, the preview is local.
    wanted = [] if cloud else [settings.model_name()]
    if recording and ((settings.live_transcribe and not cloud) or settings.live_preview):
        live = settings.live_model_name()
        if live not in wanted:
            wanted.append(live)
    missing = [name for name in wanted if not binaries.model_present(name)]
    if not missing:
        return []

    problem = no_room_for_models(missing)
    if problem:
        return [Finding(ERROR, problem)]

    sizes = {name: binaries.MODEL_SIZE_BYTES.get(name, 0) for name in missing}
    when = ("It is downloaded in the background while you record."
            if recording else
            "It is downloaded first; the transcription starts afterwards.")
    return [Finding(NOTE, f"The model '{name}' ({_size(sizes[name])}) is not on "
                          f"this computer yet. {when}")
            for name in missing]


def no_room_for_models(names):
    """Why the models `names` do not fit on the disk, or None if they do.

    Also asked by the Settings tab before it downloads a model on request.
    """
    needed = sum(binaries.MODEL_SIZE_BYTES.get(name, 0) for name in names)
    free = _free_bytes(paths.BIN_DIR)
    if free is None or not needed or free >= needed * DOWNLOAD_HEADROOM:
        return None
    listed = " and ".join(f"'{name}'" for name in names)
    drive = os.path.splitdrive(paths.BIN_DIR)[0] or paths.BIN_DIR
    return (f"Not enough free space for the model {listed}: it needs "
            f"{_size(needed)} on {drive}, and {_size(free)} is free.")


def _output_folder(out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        return [Finding(ERROR, f"The output folder {out_dir} cannot be created "
                               f"({exc}). Choose another one in the settings.")]
    if not paths.is_writable(out_dir):
        return [Finding(ERROR, f"Cannot write to the output folder {out_dir}. "
                               f"Choose another one in the settings.")]
    return []


def _recording_space(out_dir):
    """Room for the recording: the tighter of the two folders it goes through."""
    free = [value for value in (_free_bytes(paths.TMP_DIR), _free_bytes(out_dir))
            if value is not None]
    if not free:
        return []
    room = min(free)
    if room < MIN_FREE_BYTES:
        return [Finding(ERROR, f"Only {_size(room)} of disk space is free - not "
                               f"enough for a recording. Free some up or choose "
                               f"another output folder.")]
    hours = room / BYTES_PER_RECORDED_HOUR
    if hours < WARN_BELOW_HOURS:
        return [Finding(WARNING, f"Only {_size(room)} of disk space is free - "
                                 f"room for about {_duration(hours)} of "
                                 f"recording.")]
    return []


# ----------------------------------------------------------------------
def _free_bytes(folder):
    """Free bytes on the volume `folder` is (or would be) on; None if unknown."""
    probe = os.path.abspath(folder)
    while not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return int(shutil.disk_usage(probe).free)
    except (OSError, ValueError):
        return None


def _size(num_bytes):
    if num_bytes >= 10 ** 9:
        return f"{num_bytes / 10 ** 9:.1f} GB"
    return f"{num_bytes / 10 ** 6:.0f} MB"


def _duration(hours):
    minutes = int(round(hours * 60))
    whole, minutes = divmod(minutes, 60)
    return f"{whole} h {minutes} min" if whole else f"{minutes} min"
