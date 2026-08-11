"""Central path resolution.

Fixes M1: the previous version used os.getcwd() everywhere. Launched from a
shortcut or a different working directory, settings.json and output/ ended up
somewhere else and bin/ counted as "missing" -> a 3 GB model re-download.
Every path now hangs off the script directory.
"""

import os
import re

# .../Audio-Transcriber/audio_transcriber/paths.py  ->  .../Audio-Transcriber
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BIN_DIR = os.path.join(APP_DIR, "bin")
OUT_DIR = os.path.join(APP_DIR, "output")
TMP_DIR = os.path.join(OUT_DIR, ".tmp")
CFG_PATH = os.path.join(APP_DIR, "settings.json")
LOG_PATH = os.path.join(APP_DIR, "recorder.log")

WHISPER_EXE = os.path.join(BIN_DIR, "whisper-cli.exe")

# Everything a single run writes into the output directory under its base name.
# Both files are overwritten without asking unless the caller checks first.
OUTPUT_EXTENSIONS = (".wav", ".txt")

# A base name that already carries a counter: 'my_meeting_2' -> ('my_meeting', 2)
_NUMBERED_RE = re.compile(r"^(?P<stem>.+)_(?P<number>\d+)$")


def ensure_dirs():
    """Create the working directories (idempotent)."""
    for directory in (BIN_DIR, OUT_DIR, TMP_DIR):
        os.makedirs(directory, exist_ok=True)


def model_path(model_name):
    """Path to the ggml model file for e.g. 'small' or 'large-v3-turbo'."""
    return os.path.join(BIN_DIR, f"ggml-{model_name}.bin")


def safe_output_name(user_input, default="my_meeting"):
    """Turn user input into a safe base name without any path component.

    Blocks path traversal ('..\\..\\windows\\x') and empty names.

    Both separators are stripped on every platform. os.path.basename() follows
    the host rules, so on Linux and macOS a backslash is an ordinary character
    and a name written on Windows sanitised to something else entirely - the
    same settings.json produced a different file name depending on where it
    ran. The result is still safe either way, just not the same.
    """
    name = (user_input or "").strip()
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = os.path.splitext(name)[0]
    # Strip characters Windows does not allow in file names
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    name = name.strip(" .")
    return name or default


def existing_outputs(out_dir, base_name, extensions=OUTPUT_EXTENSIONS):
    """File names in out_dir that a run with this base name would overwrite.

    Returns them in the order of OUTPUT_EXTENSIONS, empty when nothing
    collides. Case sensitivity is left to the platform: os.path.exists() maps
    'Meeting.wav' onto 'meeting.wav' on Windows, which is exactly the file that
    would be clobbered there.
    """
    found = []
    for extension in extensions:
        file_name = f"{base_name}{extension}"
        if os.path.exists(os.path.join(out_dir, file_name)):
            found.append(file_name)
    return found


def next_free_name(out_dir, base_name, extensions=OUTPUT_EXTENSIONS):
    """First '<base>_<n>' variant of base_name that collides with nothing.

    A name that already ends in a counter continues counting from there, so
    repeated runs produce my_meeting_2, my_meeting_3, ... instead of
    my_meeting_2_2. The loop terminates: every candidate is a different name
    and a directory only holds finitely many files.
    """
    match = _NUMBERED_RE.match(base_name)
    if match:
        stem = match.group("stem")
        number = int(match.group("number")) + 1
    else:
        stem, number = base_name, 2

    while True:
        candidate = f"{stem}_{number}"
        if not existing_outputs(out_dir, candidate, extensions):
            return candidate
        number += 1
