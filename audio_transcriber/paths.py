"""Central path resolution.

Fixes M1: the previous version used os.getcwd() everywhere. Launched from a
shortcut or a different working directory, settings.json and output/ ended up
somewhere else and bin/ counted as "missing" -> a 3 GB model re-download.
Every path now hangs off the script directory.
"""

import os
import re
import uuid

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

# Extensions that may come with a typed name or an uploaded file and must not
# end up in the base name, because the app appends its own '.wav' / '.txt'.
# A list on purpose: os.path.splitext() removes ANY trailing '.something', so
# 'Meeting 03.10.2026' became 'Meeting 03.10' and 'v1.2 review' became 'v1'.
_STRIPPED_EXTENSIONS = frozenset({
    ".wav", ".txt", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4",
    ".webm", ".opus", ".aiff", ".aif", ".m4b", ".amr", ".caf",
})

# Names Microsoft documents as devices, with or without an extension ('NUL.txt'
# is equivalent to NUL). Whether a given Windows build still honours that
# differs - 11 build 26200 writes an ordinary file - so this is a precaution
# for the builds that do.
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{device}{digit}" for device in ("COM", "LPT")
       for digit in "123456789¹²³"})


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

    Pass the name as it was typed or as the file is called - extension
    included. Exactly one known audio/transcript extension is removed; any
    other dots belong to the name.
    """
    name = (user_input or "").strip()
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem, extension = os.path.splitext(name)
    if extension.lower() in _STRIPPED_EXTENSIONS:
        name = stem
    # Strip characters Windows does not allow in file names
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    name = name.strip(" .")
    # Reserved device names count by their first segment: 'aux.v2' is AUX too.
    first, dot, rest = name.partition(".")
    if first.rstrip().upper() in _RESERVED_NAMES:
        name = f"{first}_{dot}{rest}"
    return name or default


def scratch_name(*labels, suffix=".wav"):
    """Unique, ASCII-only file name for a temporary file whisper-cli must open.

    whisper-cli.exe receives its arguments in the ANSI code page. On a western
    Windows a Turkish or Arabic recording name arrives mangled ('toplantı'
    becomes 'toplanti', Arabic becomes '???') and whisper stops with exit code
    2, "input file not found" - so nothing the user typed may end up in such a
    name. The labels are program constants ('mic', 'live', ...); a random token
    keeps two runs from ever sharing a file.
    """
    parts = [re.sub(r"[^A-Za-z0-9]+", "", str(label)) for label in labels]
    parts = [part for part in parts if part]
    parts.append(uuid.uuid4().hex[:10])
    return "-".join(parts) + suffix


def ansi_safe_path(path, windows=None, short_path=None):
    """`path` in a form whisper-cli.exe can open, given its ANSI arguments.

    scratch_name() keeps user text out of file names, but the directories are
    not ours to choose: an install folder or user profile with a Turkish or
    Arabic name breaks the model path and the input path alike (exit code 2,
    "input file not found"). The 8.3 short form of an existing path is pure
    ASCII, so it is used instead. A volume without short names, or a path that
    does not exist, leaves the path unchanged - the call then fails as it
    always did.

    `windows` and `short_path` exist so the logic can be tested anywhere.
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows or path.isascii():
        return path
    short_path = short_path or _windows_short_path
    try:
        short = short_path(path)
    except OSError:
        short = None
    return short if short and short.isascii() else path


def _windows_short_path(path):
    """The 8.3 form of an existing path, or None if there is none."""
    import ctypes
    from ctypes import wintypes

    get_short = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD

    needed = get_short(path, None, 0)
    if not needed:
        return None
    buffer = ctypes.create_unicode_buffer(needed)
    written = get_short(path, buffer, needed)
    return buffer.value if written else None


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
