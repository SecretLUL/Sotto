"""Central path resolution.

Fixes M1: the previous version used os.getcwd() everywhere. Launched from a
shortcut or a different working directory, settings.json and output/ ended up
somewhere else and bin/ counted as "missing" -> a 3 GB model re-download.
Every path now hangs off one data directory that does not depend on the cwd.

Where that directory is (resolve_data_dir):
  * $AUDIO_TRANSCRIBER_HOME, if set - an explicit choice always wins.
  * From source: the repository, as before.
  * A packaged (PyInstaller) build: next to the executable, so "extract
    anywhere and run" keeps working. __file__ is no use there - in the onedir
    layout it points into the bundled _internal folder, which is where bin/
    (the multi-GB models), output/ and settings.json used to end up: gone with
    every update of the app folder, and not writable wherever the app is
    installed read-only.
  * If the folder next to the executable is not writable, and always on macOS
    (writing into a .app breaks its signature, and a quarantined app is run
    from a read-only copy): the per-user data folder.
"""

import os
import re
import sys
import uuid

# The name of the per-user data folder, from before the app was called Sotto.
# Renaming it would leave every user's models and settings behind.
APP_NAME = "AudioTranscriber"

# Names the data directory explicitly - portable installs, tests, CI.
HOME_ENV = "AUDIO_TRANSCRIBER_HOME"

# .../Audio-Transcriber/audio_transcriber/paths.py  ->  .../Audio-Transcriber
# In a packaged build this is PyInstaller's _internal folder - not a place for
# data, but where an older packaged build left some (see migrate_legacy_data).
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_writable(directory):
    """True if a file can really be created there.

    os.access() is useless on Windows: it only looks at the read-only attribute
    and reports C:\\Program Files as writable.

    Exactly one attempt, deliberately not tempfile: on Windows tempfile takes a
    PermissionError for "that name exists" whenever os.access() claims the
    folder is writable, and retries up to 10 000 times - in Program Files that
    is minutes of a frozen start-up, in the one place this check exists for.
    """
    probe = os.path.join(directory, f".write-test-{uuid.uuid4().hex}")
    # O_TEMPORARY (Windows) deletes the file when it is closed.
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_TEMPORARY", 0)
    try:
        os.close(os.open(probe, flags))
    except OSError:
        return False
    try:
        os.remove(probe)               # POSIX; on Windows it is already gone
    except OSError:
        pass
    return True


def _user_data_dir(platform, env, home):
    if platform == "win32":
        base = (env.get("LOCALAPPDATA") or env.get("APPDATA")
                or os.path.join(home, "AppData", "Local"))
    elif platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    else:
        base = env.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return os.path.join(base, APP_NAME)


def resolve_data_dir(source_dir=None, frozen=None, executable=None,
                     platform=None, env=None, home=None, writable=None):
    """Where settings.json, bin/ and output/ live (see the module docstring).

    The arguments exist so the decision can be tested for every platform; the
    defaults describe this process. They are looked up at call time, not bound
    when the module is imported.
    """
    source_dir = SOURCE_DIR if source_dir is None else source_dir
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    executable = sys.executable if executable is None else executable
    platform = sys.platform if platform is None else platform
    env = os.environ if env is None else env
    home = os.path.expanduser("~") if home is None else home
    writable = is_writable if writable is None else writable

    override = (env.get(HOME_ENV) or "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if not frozen:
        return source_dir

    if platform != "darwin":
        beside_executable = os.path.dirname(os.path.abspath(executable))
        if writable(beside_executable):
            return beside_executable
    return _user_data_dir(platform, env, home)


DATA_DIR = resolve_data_dir()
APP_DIR = DATA_DIR            # what this module called it before packaged builds

BIN_DIR = os.path.join(DATA_DIR, "bin")
OUT_DIR = os.path.join(DATA_DIR, "output")
TMP_DIR = os.path.join(OUT_DIR, ".tmp")
CFG_PATH = os.path.join(DATA_DIR, "settings.json")
LOG_PATH = os.path.join(DATA_DIR, "recorder.log")

# Where a whisper-cli that was put into bin/ is expected. Only Windows gets one
# downloaded there (transcribe/binaries.py).
WHISPER_EXE = os.path.join(BIN_DIR, "whisper-cli.exe" if os.name == "nt"
                           else "whisper-cli")

# What a packaged build used to keep inside PyInstaller's _internal folder.
LEGACY_ITEMS = ("settings.json", "bin", "output")

# Everything a single run writes into the output directory under its base name.
# Both files are overwritten without asking unless the caller checks first.
OUTPUT_EXTENSIONS = (".wav", ".txt")

# What "Keep raw tracks" adds for a recording: the unmixed microphone and system
# tracks at their native sample rate, for a DAW or an editor.
RAW_TRACK_EXTENSIONS = (".mic.wav", ".sys.wav")


def raw_track_path(out_dir, base_name, kind):
    """Where a kept raw track goes: <out_dir>/<base_name>.mic.wav or .sys.wav."""
    return os.path.join(out_dir, f"{base_name}.{kind}.wav")


def output_extensions(keep_raw_tracks=False):
    """Every file a recording run writes under its base name.

    Uploads write the first two only; a recording with "Keep raw tracks" on
    writes the raw tracks as well, and the overwrite check has to know.
    """
    if keep_raw_tracks:
        return OUTPUT_EXTENSIONS + RAW_TRACK_EXTENSIONS
    return OUTPUT_EXTENSIONS

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
# for the builds that do. COM and LPT come with 1-9 and with the superscript
# digits (the last three characters below), which Windows reserves as well.
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{device}{digit}" for device in ("COM", "LPT")
       for digit in "123456789¹²³"})


def ensure_dirs():
    """Create the working directories (idempotent)."""
    for directory in (BIN_DIR, OUT_DIR, TMP_DIR):
        os.makedirs(directory, exist_ok=True)


def migrate_legacy_data(source_dir=None, data_dir=None, frozen=None):
    """Move what an older packaged build left in _internal to the data directory.

    Those builds kept bin/ (the multi-GB models), output/ and settings.json
    inside PyInstaller's _internal folder. A version that looks next to the
    executable instead would make all of it look lost - including a 3 GB model
    that would be downloaded again - so it is moved over, once, before anything
    reads or creates files there.

    Moving is a rename: instant even for the models, and it cannot leave a
    half-copied model behind. Where a rename is not possible (another drive)
    the data simply stays where it is. Nothing is ever overwritten; an empty
    placeholder directory is replaced. Does nothing outside a packaged build.

    Returns the names that were moved.
    """
    source_dir = SOURCE_DIR if source_dir is None else source_dir
    data_dir = DATA_DIR if data_dir is None else data_dir
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if not frozen or _same_directory(source_dir, data_dir):
        return []

    moved = []
    for name in LEGACY_ITEMS:
        old = os.path.join(source_dir, name)
        new = os.path.join(data_dir, name)
        if not os.path.exists(old):
            continue
        try:
            if os.path.exists(new):
                if not _is_empty_dir(new):
                    continue
                os.rmdir(new)
            os.makedirs(data_dir, exist_ok=True)
            os.rename(old, new)
        except OSError:
            continue
        moved.append(name)
    return moved


def _same_directory(first, second):
    return (os.path.normcase(os.path.realpath(first))
            == os.path.normcase(os.path.realpath(second)))


def _is_empty_dir(path):
    return os.path.isdir(path) and not os.listdir(path)


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
