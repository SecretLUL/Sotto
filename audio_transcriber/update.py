"""Updating a packaged build from the GitHub releases.

The way through, start to end:
  1. check_for_update() asks GitHub for the newest release and compares its
     tag with the one this build was made for (version.current()).
  2. prepare() downloads the archive for this system and SHA256SUMS.txt from
     that release, refuses an archive whose checksum does not match, and
     unpacks it into the workspace: <install folder>/.update/staged/Sotto.
  3. start_install() runs the NEW executable from there as the installer
     (--apply-update), and the app quits.
  4. The installer waits for the old process to end, copies the new program
     files next to the old ones, swaps them in by renaming, starts Sotto again
     if asked to, and leaves a result.json for it.
  5. The new version reads that result at its start (take_result()) and clears
     the workspace (cleanup()).

Why a separate process: Windows does not let a running program replace its
own .exe and the DLLs in _internal, and it is the new version's installer
that knows how its files are laid out. The command line of step 3 is the
contract between an old version and the next one - keep accepting it.

Only the program files are replaced: the entries at the top of the release
archive (Sotto.exe or Sotto, and _internal). A portable install keeps its
models, recordings and settings in the same folder (paths.resolve_data_dir),
and none of that is in an archive - one that tried to bring a bin/ or a
settings.json along is refused. The swap is renames within one folder, so
it takes a moment, and a failure half way puts every old file back.

Where the app cannot replace itself - run from source, on macOS (an app
bundle, unsigned), from a folder it cannot write to - it still says that a
new version is out, and offers the release page instead.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import paths, version
from .events import UpdateChecked, UpdateFetch, UpdateFetchEnded
from .transcribe import binaries

REPOSITORY = "SecretLUL/Sotto"
LATEST_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases/latest"
SUMS_NAME = "SHA256SUMS.txt"
CHECK_TIMEOUT_S = 15

# The workspace in the install folder, and what it holds.
WORKSPACE_NAME = ".update"
DOWNLOAD_DIR = "download"      # the archive and the checksums
STAGED_DIR = "staged"          # the unpacked archive; the installer runs from here
INCOMING_DIR = "incoming"      # the installer's copy, renamed into place from here
BACKUP_DIR = "backup"          # the old program files, until the new version starts
RESULT_FILE = "result.json"
LOG_FILE = "update.log"

# What the data directory keeps next to the program. Never part of an archive.
DATA_NAMES = frozenset(name.lower() for name in (
    *paths.LEGACY_ITEMS, os.path.basename(paths.LOG_PATH), WORKSPACE_NAME))

# The installer's command line: the contract between two versions (see above).
HELPER_FLAG = "--apply-update"

# How long the installer waits for the old process to end.
EXIT_TIMEOUT_S = 60.0
# Renames are tried again for a while: a virus scanner that is looking at a
# freshly written file holds it open for a moment.
RENAME_ATTEMPTS = 50
RENAME_DELAY_S = 0.2

# A download unpacks to about two and a half times its size, and the installer
# copies that once more; this leaves room for both and the archive.
ROOM_FACTOR = 6

# prepare() and cleanup() both clear the workspace.
_WORKSPACE_LOCK = threading.Lock()


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int = 0


@dataclass(frozen=True)
class Release:
    tag: str
    page_url: str
    archive: Asset = None      # the download for this system; None if it has none
    sums: Asset = None


# ----------------------------------------------------------------------
# Is there something new?
# ----------------------------------------------------------------------
def fetch_latest(timeout=CHECK_TIMEOUT_S, urlopen=None):
    """GitHub's description of the newest release (drafts and pre-releases
    are not among them). Raises UpdateError."""
    urlopen = urlopen or urllib.request.urlopen
    request = urllib.request.Request(LATEST_URL, headers={
        "User-Agent": f"Sotto/{version.current() or 'dev'}",
        "Accept": "application/vnd.github+json",
    })
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read(1 << 20).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("No release has been published yet.") from exc
        if exc.code in (403, 429):
            raise UpdateError("GitHub is limiting requests from this network "
                              "right now; try again later.") from exc
        raise UpdateError(f"GitHub responded with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise UpdateError(f"Could not reach GitHub ({exc.reason}).") from exc
    except (OSError, ValueError) as exc:
        raise UpdateError(f"Could not read GitHub's answer ({exc}).") from exc
    if not isinstance(data, dict):
        raise UpdateError("GitHub's answer has an unexpected format.")
    return data


def release_from_json(data, system=None, machine=None):
    """The Release in GitHub's answer, or None if it is not one to install."""
    tag = str(data.get("tag_name") or "").strip()
    if data.get("draft") or data.get("prerelease") or version.parse(tag) is None:
        return None
    assets = {}
    for item in data.get("assets") or ():
        if not isinstance(item, dict):
            continue
        name, url = str(item.get("name") or ""), str(item.get("browser_download_url") or "")
        if name and url.startswith("https://"):
            try:
                size = max(0, int(item.get("size") or 0))
            except (TypeError, ValueError):
                size = 0
            assets[name] = Asset(name, url, size)
    page = str(data.get("html_url") or "")
    if not page.startswith("https://github.com/"):
        page = RELEASES_PAGE
    return Release(tag, page,
                   archive=assets.get(version.archive_name(tag, system, machine)),
                   sums=assets.get(SUMS_NAME))


def check_for_update(installed, fetch=None, system=None, machine=None):
    """The newest release if it is newer than `installed`, else None."""
    release = release_from_json((fetch or fetch_latest)(), system, machine)
    if release is None or not version.is_newer(release.tag, installed):
        return None
    return release


# ----------------------------------------------------------------------
# Can this build replace itself?
# ----------------------------------------------------------------------
def executable_name(system=None):
    system = sys.platform if system is None else system
    return f"{version.APP_NAME}.exe" if system == "win32" else version.APP_NAME


def install_dir(executable=None):
    return os.path.dirname(os.path.abspath(sys.executable if executable is None
                                           else executable))


def workspace(target=None):
    return os.path.join(install_dir() if target is None else target, WORKSPACE_NAME)


def install_problem(frozen=None, system=None, executable=None, writable=None):
    """Why this build cannot install an update itself, or None if it can."""
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    system = sys.platform if system is None else system
    executable = sys.executable if executable is None else executable
    writable = paths.is_writable if writable is None else writable

    if not frozen:
        return "Sotto runs from its source code here; update it with git pull."
    if system == "darwin":
        return ("On macOS Sotto cannot replace itself yet: download the new "
                "version and put it in place of the old one.")
    if os.path.basename(executable) != executable_name(system):
        return (f"The program file is no longer called {executable_name(system)}, "
                f"so the update would not know what to replace.")
    if not writable(install_dir(executable)):
        return ("Sotto's folder is read-only for your account. Download the new "
                "version, or move Sotto to a folder you can write to.")
    return None


# ----------------------------------------------------------------------
# Download, verify, unpack
# ----------------------------------------------------------------------
def prepare(release, workspace_dir, download=None, cancelled=None, on_bytes=None,
            log=None, system=None, disk_usage=shutil.disk_usage):
    """Fetch `release` for this system and unpack it, ready to install.

    Returns the unpacked app folder (the one with the executable in it).
    Raises UpdateError, or binaries.DownloadError for a failed or cancelled
    download.
    """
    download = download or binaries.download
    log = log or (lambda _message: None)
    if release.archive is None:
        _suffix, slug = version.platform_tag(system)
        raise UpdateError(f"{release.tag} has no download for this system ({slug}).")
    if release.sums is None:
        raise UpdateError(f"{release.tag} comes without checksums, and an "
                          f"unverified download is not installed.")

    with _WORKSPACE_LOCK:
        for name in (DOWNLOAD_DIR, STAGED_DIR, INCOMING_DIR, BACKUP_DIR):
            _remove(os.path.join(workspace_dir, name))
        download_dir = os.path.join(workspace_dir, DOWNLOAD_DIR)
        os.makedirs(download_dir)

        needed = release.archive.size * ROOM_FACTOR
        free = disk_usage(workspace_dir).free
        if needed and free < needed:
            raise UpdateError(
                f"There is not enough free space for the update: it needs about "
                f"{needed / (1 << 20):.0f} MB, and {free / (1 << 20):.0f} MB are free.")

        log("Fetching the checksums…")
        sums_path = download(release.sums.url, os.path.join(download_dir, SUMS_NAME),
                             "Checksums", cancelled=cancelled)
        expected = read_sums(sums_path).get(release.archive.name)
        if expected is None:
            raise UpdateError(f"{SUMS_NAME} lists no checksum for "
                              f"{release.archive.name}.")

        log(f"Downloading Sotto {release.tag}…")
        archive = download(release.archive.url,
                           os.path.join(download_dir, release.archive.name),
                           f"Sotto {release.tag}", cancelled=cancelled,
                           on_bytes=on_bytes)
        log("Verifying the download…")
        digest = binaries.sha256_of(archive)
        if digest != expected:
            raise UpdateError(
                f"The download does not match its checksum (expected "
                f"{expected[:16]}…, got {digest[:16]}…), so it was not installed.")

        log("Unpacking…")
        staged = os.path.join(workspace_dir, STAGED_DIR)
        extract(archive, staged)
        app = _app_folder(staged, system)
        _remove(download_dir)              # the archive is not needed any more
        return app


def read_sums(path):
    """{file name: sha256} from a sha256sum listing ('<hash>  <name>' or '*<name>')."""
    sums = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and len(parts[0]) == 64:
                sums[parts[1].lstrip("*").strip()] = parts[0].lower()
    return sums


def extract(archive, dest):
    """Unpack a release archive into `dest`, refusing paths that leave it."""
    os.makedirs(dest, exist_ok=True)
    if archive.endswith(".zip"):
        binaries.safe_extract(archive, dest)
        return
    if not hasattr(tarfile, "data_filter"):
        raise UpdateError("This Python cannot unpack the archive safely.")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            # "data": nothing outside dest, no devices, no setuid - but the
            # executable keeps its executable bit.
            tar.extractall(dest, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise UpdateError(f"The archive could not be unpacked ({exc}).") from exc


def _app_folder(staged, system=None):
    """The one Sotto folder an archive unpacks to, checked."""
    entries = os.listdir(staged)
    app = os.path.join(staged, version.APP_NAME)
    if entries != [version.APP_NAME] or not os.path.isdir(app):
        raise UpdateError(f"The download does not unpack to a single "
                          f"{version.APP_NAME} folder.")
    check_program_names(os.listdir(app), executable_name(system))
    return app


def check_program_names(names, exe_name):
    """Refuse a set of program files that would replace data, or lacks the program."""
    clashing = sorted(name for name in names if name.lower() in DATA_NAMES)
    if clashing:
        raise UpdateError(f"The download contains {', '.join(clashing)}, which "
                          f"would replace your own data - it was not installed.")
    if exe_name not in names:
        raise UpdateError(f"The download contains no {exe_name}.")


# ----------------------------------------------------------------------
# Install: the old app starts the installer, the installer swaps the files
# ----------------------------------------------------------------------
def helper_command(staged_app, target, pid, relaunch, system=None):
    command = [os.path.join(staged_app, executable_name(system)), HELPER_FLAG,
               "--source", staged_app, "--target", target, "--wait-pid", str(pid)]
    if relaunch:
        command.append("--relaunch")
    return command


def start_install(staged_app, target=None, relaunch=True, pid=None, popen=None):
    """Start the new version's installer. The caller quits right after."""
    target = install_dir() if target is None else target
    pid = os.getpid() if pid is None else pid
    _launch(helper_command(staged_app, target, pid, relaunch), cwd=staged_app,
            popen=popen)


def apply_from_command_line(argv):
    """The installer: main.py hands it the command line when HELPER_FLAG is on it.

    Never raises and never shows a window: what happened goes to result.json,
    which the next start of Sotto reads, and to update.log.
    """
    parser = argparse.ArgumentParser(prog=version.APP_NAME, add_help=False)
    parser.add_argument(HELPER_FLAG, action="store_true")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--relaunch", action="store_true")
    try:
        args, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        return 2

    workspace_dir = workspace(args.target)
    tag = version.current() or "the new version"
    try:
        os.makedirs(workspace_dir, exist_ok=True)
    except OSError:
        return 1
    _log(workspace_dir, f"Installing {tag} into {args.target}")

    if args.wait_pid and not wait_for_exit(args.wait_pid, EXIT_TIMEOUT_S):
        _log(workspace_dir, f"Sotto (process {args.wait_pid}) is still running "
                            f"after {EXIT_TIMEOUT_S:.0f} s; trying anyway")
    try:
        install(args.source, args.target, workspace_dir)
    except Exception as exc:
        _log(workspace_dir, f"Failed: {exc}")
        write_result(workspace_dir, ok=False, tag=tag, error=str(exc))
    else:
        _log(workspace_dir, "Installed")
        write_result(workspace_dir, ok=True, tag=tag)

    if args.relaunch:
        try:
            _launch([os.path.join(args.target, executable_name())], cwd=args.target)
        except OSError as exc:
            _log(workspace_dir, f"Could not start Sotto again: {exc}")
    return 0


def install(source, target, workspace_dir, exe_name=None):
    """Replace the program files in `target` with those in `source`.

    Copies first, into the workspace - the slow part, while nothing has been
    touched yet - then moves each old entry aside and the new one into its
    place. A failure on the way moves everything back. Anything in `target`
    that is not in `source` (the data) is left alone.
    """
    exe_name = exe_name or executable_name()
    names = os.listdir(source)
    check_program_names(names, exe_name)
    # The executable last: until it is swapped, the old one is still complete.
    names.sort(key=lambda name: (name == exe_name, name))

    incoming = os.path.join(workspace_dir, INCOMING_DIR)
    backup = os.path.join(workspace_dir, BACKUP_DIR)
    for folder in (incoming, backup):
        _remove(folder)
        os.makedirs(folder)
    for name in names:
        _copy(os.path.join(source, name), os.path.join(incoming, name))

    swapped = []
    try:
        for name in names:
            current = os.path.join(target, name)
            had_old = os.path.lexists(current)
            if had_old:
                _retry(os.rename, current, os.path.join(backup, name))
            try:
                _retry(os.rename, os.path.join(incoming, name), current)
            except BaseException:
                if had_old:
                    _retry(os.rename, os.path.join(backup, name), current)
                raise
            swapped.append((name, had_old))
    except BaseException as exc:
        try:
            for name, had_old in reversed(swapped):
                _retry(os.rename, os.path.join(target, name),
                       os.path.join(incoming, name))
                if had_old:
                    _retry(os.rename, os.path.join(backup, name),
                           os.path.join(target, name))
        except OSError as undo:
            raise UpdateError(
                f"{exc} - and the old version could not be put back ({undo}); "
                f"its files are in {backup}.") from exc
        raise


def write_result(workspace_dir, ok, tag, error=""):
    try:
        with open(os.path.join(workspace_dir, RESULT_FILE), "w",
                  encoding="utf-8") as handle:
            json.dump({"ok": bool(ok), "tag": tag, "error": error}, handle)
    except OSError:
        pass


def take_result(workspace_dir=None):
    """What the last installer left behind ({'ok', 'tag', 'error'}), once."""
    workspace_dir = workspace() if workspace_dir is None else workspace_dir
    path = os.path.join(workspace_dir, RESULT_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, ValueError):
        return None
    try:
        os.remove(path)
    except OSError:
        pass
    if not isinstance(result, dict):
        return None
    return {"ok": bool(result.get("ok")), "tag": str(result.get("tag") or ""),
            "error": str(result.get("error") or "")}


def cleanup(workspace_dir=None, attempts=None):
    """Clear what an update left in the workspace; the log stays.

    Called at the start. The installer may still be on its way out, holding
    the staged files, so each folder is tried for a while.
    """
    workspace_dir = workspace() if workspace_dir is None else workspace_dir
    with _WORKSPACE_LOCK:
        for name in (DOWNLOAD_DIR, STAGED_DIR, INCOMING_DIR, BACKUP_DIR):
            try:
                _remove(os.path.join(workspace_dir, name), attempts=attempts)
            except OSError:
                pass


def wait_for_exit(pid, timeout):
    """True once process `pid` has ended, False if it still runs after `timeout`."""
    if os.name == "nt":
        return _wait_for_exit_windows(pid, timeout)
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass                       # it exists, it is just not ours
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _wait_for_exit_windows(pid, timeout):
    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return True                    # no such process (any more)
    try:
        return kernel32.WaitForSingleObject(handle, int(timeout * 1000)) == 0
    finally:
        kernel32.CloseHandle(handle)


def child_environment(env=None, frozen=None):
    """The environment for a program started from this one, without the bundle's.

    A PyInstaller build changes its own environment on the way up: Linux gets
    LD_LIBRARY_PATH pointed into _internal, Tk gets TCL_LIBRARY, and the
    bootloader leaves _PYI_ variables that tell a child it is part of the same
    application. Handed on unchanged, the new version would start with the old
    one's libraries - from a folder the installer has just moved away.
    """
    env = dict(os.environ if env is None else env)
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if not frozen:
        return env
    for key in list(env):
        if key.startswith("_PYI_") or key in ("_MEIPASS2", "TCL_LIBRARY", "TK_LIBRARY"):
            del env[key]
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        original = env.pop(key + "_ORIG", None)
        if original is not None:
            env[key] = original
        else:
            env.pop(key, None)
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return env


def _launch(command, cwd, popen=None):
    """Start `command` on its own: no console, not tied to this process."""
    popen = popen or subprocess.Popen
    options = dict(cwd=cwd, env=child_environment(), stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   close_fds=True)
    if os.name == "nt":
        options["creationflags"] = (subprocess.DETACHED_PROCESS
                                    | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        options["start_new_session"] = True
    return popen(command, **options)


def _log(workspace_dir, message):
    try:
        with open(os.path.join(workspace_dir, LOG_FILE), "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")
    except OSError:
        pass


def _retry(function, *args, attempts=None):
    attempts = RENAME_ATTEMPTS if attempts is None else attempts
    for attempt in range(max(1, attempts)):
        try:
            return function(*args)
        except OSError:
            if attempt >= attempts - 1:
                raise
            time.sleep(RENAME_DELAY_S)


def _copy(source, dest):
    if os.path.isdir(source) and not os.path.islink(source):
        shutil.copytree(source, dest, symlinks=True)
    else:
        shutil.copy2(source, dest, follow_symlinks=False)


def _remove(path, attempts=None):
    """Delete a file or folder, if it is there; read-only files included."""
    if not os.path.lexists(path):
        return

    def remove():
        if os.path.isdir(path) and not os.path.islink(path):
            # onerror before Python 3.12, which deprecates it for onexc.
            handler = "onexc" if sys.version_info >= (3, 12) else "onerror"
            shutil.rmtree(path, **{handler: _make_writable_and_retry})
        else:
            os.remove(path)

    _retry(remove, attempts=attempts)


def _make_writable_and_retry(function, path, _exc):
    """Windows refuses to delete a read-only file; clear the flag and try again."""
    import stat
    os.chmod(path, stat.S_IWRITE)
    function(path)


# ----------------------------------------------------------------------
# Workers for the window
# ----------------------------------------------------------------------
class UpdateCheck:
    """Asks GitHub in the background; posts exactly one UpdateChecked."""

    def __init__(self, bridge, installed, fetch=None):
        self.bridge = bridge
        self.installed = installed
        self._fetch = fetch
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="update-check",
                                        daemon=True)
        self._thread.start()
        return self._thread

    def _run(self):
        try:
            release = check_for_update(self.installed, fetch=self._fetch)
        except Exception as exc:
            self.bridge.post(UpdateChecked(error=str(exc)))
            return
        self.bridge.post(UpdateChecked(release=release))


class UpdateDownload:
    """Downloads and unpacks a release in the background - "Update now".

    Reports through UpdateFetch events (throttled) and exactly one
    UpdateFetchEnded, which names the unpacked app folder on success.
    """

    REPORT_INTERVAL_S = 0.1

    def __init__(self, bridge, release, workspace_dir, prepare_fn=None):
        self.bridge = bridge
        self.release = release
        self.workspace_dir = workspace_dir
        self._prepare = prepare_fn or prepare
        self._cancelled = threading.Event()
        self._thread = None
        self._step = ""
        self._started = 0.0
        self._last_report = 0.0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="update-download",
                                        daemon=True)
        self._thread.start()
        return self._thread

    def cancel(self):
        """Ends the download between two blocks; UpdateFetchEnded follows."""
        self._cancelled.set()

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    def _run(self):
        tag = self.release.tag
        try:
            staged = self._prepare(self.release, self.workspace_dir,
                                   cancelled=self._cancelled.is_set,
                                   on_bytes=self._report, log=self._on_step)
        except Exception as exc:
            if self._cancelled.is_set():
                self.bridge.post(UpdateFetchEnded(tag, cancelled=True))
            else:
                self.bridge.post(UpdateFetchEnded(tag, error=str(exc)))
            return
        if self._cancelled.is_set():
            self.bridge.post(UpdateFetchEnded(tag, cancelled=True))
            return
        self.bridge.post(UpdateFetchEnded(tag, staged=staged))

    def _on_step(self, message):
        self._step = message
        self._started = time.monotonic()
        self.bridge.post(UpdateFetch(message))

    def _report(self, done, total):
        total = total or (self.release.archive.size if self.release.archive else 0)
        now = time.monotonic()
        if now - self._last_report < self.REPORT_INTERVAL_S and done != total:
            return
        self._last_report = now
        rate = done / max(1e-6, now - self._started)
        self.bridge.post(UpdateFetch(self._step, done=done, total=total, rate=rate))
