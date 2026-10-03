"""Fetching whisper.cpp binaries and models.

Fixes H8 and M2:
  * Downloads went straight to the target file. If the 3.1 GB download of
    ggml-large-v3.bin broke off, a partial file was left behind that
    os.path.exists() waved through on the next start -> whisper loaded a
    corrupt model. Downloads now go to <target>.part, the length is verified
    and only then is the file renamed atomically.
  * urlopen() without a timeout could hang indefinitely.
  * zip_ref.extractall() checked no paths (zip slip).
"""

import hashlib
import http.client
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile

from ..paths import BIN_DIR, WHISPER_EXE, model_path

USER_AGENT = "AudioTranscriber/2.0 (+local)"
CONNECT_TIMEOUT = 30

_IS_WINDOWS = os.name == "nt"

# The only prebuilt whisper.cpp this app knows is the Windows (Vulkan) one below,
# a third-party build of the upstream project. It is executed after download,
# so it is pinned to the exact archive: GitHub's digest for that release asset
# (created 2026-06-20, not modified since).
WHISPER_ZIP_URL = ("https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/"
                   "download/v1.8.4/whisper-v1.8.4-windows-vulkan-x64.zip")

# The model repositories are pinned to the revision the hashes below belong to.
# A branch name moves; a revision keeps serving exactly these files, so a
# checksum failure means a damaged or tampered download, not "upstream changed".
WHISPER_MODELS_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"  # ggerganov/whisper.cpp
VAD_REVISION = "9ffd54a1e1ee413ddf265af9913beaf518d1639b"             # ggml-org/whisper-vad
MODEL_URL_TEMPLATE = ("https://huggingface.co/ggerganov/whisper.cpp/resolve/"
                      + WHISPER_MODELS_REVISION + "/ggml-{name}.bin")
VAD_MODEL_URL = ("https://huggingface.co/ggml-org/whisper-vad/resolve/"
                 + VAD_REVISION + "/ggml-silero-v5.1.2.bin")
VAD_MODEL_PATH = os.path.join(BIN_DIR, "ggml-silero-v5.1.2.bin")

# Files kept from the release archive.
ESSENTIAL = {
    "whisper-cli.exe", "whisper.dll", "ggml.dll", "ggml-base.dll",
    "ggml-cpu.dll", "ggml-vulkan.dll", "SDL2.dll",
}

# SHA-256 of everything that is downloaded, keyed by the file name it is saved
# under. The model hashes are the LFS object ids Hugging Face publishes for the
# pinned revision; they also match the files an earlier version of this app had
# downloaded. A file whose name is not listed is only checked for its length.
EXPECTED_SHA256 = {
    "whisper-vulkan.zip":
        "e0d20a0f92e31b98adc0faf71172efc810b701e6391a9d858ca045bff26f77cd",
    "ggml-tiny.bin":
        "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
    "ggml-base.bin":
        "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
    "ggml-small.bin":
        "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    "ggml-medium.bin":
        "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
    "ggml-large-v3-turbo.bin":
        "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    "ggml-large-v3.bin":
        "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
    "ggml-silero-v5.1.2.bin":
        "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
}

# Size of every model file at the pinned revision. Only used to tell the user
# how much a download is and to see whether the disk has room for it; the hash
# above is what proves a download right.
MODEL_SIZE_BYTES = {
    "tiny": 77_691_713,
    "base": 147_951_465,
    "small": 487_601_967,
    "medium": 1_533_763_059,
    "large-v3-turbo": 1_624_555_275,
    "large-v3": 3_095_033_483,
}

NO_WHISPER_HINT = (
    "The local engine needs whisper.cpp's command-line program (whisper-cli), "
    "which this app only downloads automatically on Windows. Install it - "
    "macOS: 'brew install whisper-cpp'; Linux: build "
    "https://github.com/ggml-org/whisper.cpp or use your distribution's "
    "package - so that 'whisper-cli' is on the PATH, or copy it to "
    "{bin_dir}. Or choose the ElevenLabs engine in the settings.")


class DownloadError(RuntimeError):
    pass


# One lock per file that is fetched. The live preview and the closing pass can
# both ask for the same model within seconds of each other; the second one used
# to delete the first one's half-written .part file (on Windows: a raw
# PermissionError after the meeting) or download everything again.
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path):
    key = os.path.normcase(os.path.abspath(path))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


# ----------------------------------------------------------------------
def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest_path, description="File", progress=None,
             timeout=CONNECT_TIMEOUT, cancelled=None):
    """Download atomically to dest_path.

    progress: callable(text) for progress messages (may be None).
    cancelled: callable() -> bool, asked between blocks; True ends the download
    with a DownloadError and removes the partial file.
    Raises DownloadError.
    """
    tmp_path = dest_path + ".part"
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError as exc:
            # On Windows that means another download of this very file is
            # writing to it - say so instead of leaking a bare PermissionError.
            raise DownloadError(
                f"{description}: a partial download is in use by something "
                f"else ({exc}).") from exc

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1 << 20
            started = last_report = time.monotonic()

            with open(tmp_path, "wb") as out:
                while True:
                    if cancelled is not None and cancelled():
                        raise DownloadError(f"{description}: download cancelled.")
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out.write(buffer)
                    downloaded += len(buffer)

                    now = time.monotonic()
                    if progress and (now - last_report > 0.4 or downloaded == total):
                        last_report = now
                        elapsed = max(1e-6, now - started)
                        speed = downloaded / elapsed / (1 << 20)
                        if total:
                            progress(f"{description}: {downloaded / total * 100:5.1f} % "
                                     f"({downloaded / (1 << 20):.1f}/{total / (1 << 20):.1f} MB) "
                                     f"at {speed:.1f} MB/s")
                        else:
                            progress(f"{description}: {downloaded / (1 << 20):.1f} MB "
                                     f"at {speed:.1f} MB/s")

        if total and os.path.getsize(tmp_path) != total:
            raise DownloadError(
                f"{description} downloaded incompletely "
                f"({os.path.getsize(tmp_path)} of {total} bytes).")

        name = os.path.basename(dest_path)
        if name in EXPECTED_SHA256:
            digest = sha256_of(tmp_path)
            if digest != EXPECTED_SHA256[name]:
                raise DownloadError(
                    f"Checksum mismatch for {name} "
                    f"(expected {EXPECTED_SHA256[name][:16]}…, "
                    f"got {digest[:16]}…).")

        os.replace(tmp_path, dest_path)          # atomic
        return dest_path

    except urllib.error.HTTPError as exc:
        raise DownloadError(f"{description}: server responded with "
                            f"HTTP {exc.code} ({exc.reason}).") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"{description}: could not connect "
                            f"({exc.reason}).") from exc
    except http.client.HTTPException as exc:
        # IncompleteRead and friends: the connection broke mid-transfer. Not an
        # OSError, so it used to escape as a bare exception.
        raise DownloadError(f"{description}: the connection was lost "
                            f"({exc}).") from exc
    except OSError as exc:
        raise DownloadError(f"{description}: network or disk error "
                            f"({exc}).") from exc
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def safe_extract(zip_path, dest_dir):
    """Extract an archive, rejecting path traversal (zip slip)."""
    dest_real = os.path.realpath(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = os.path.realpath(os.path.join(dest_real, member.filename))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise DownloadError(
                    f"Archive contains a path outside the target directory: "
                    f"{member.filename!r} - extraction aborted.")
        archive.extractall(dest_real)


# ----------------------------------------------------------------------
def find_whisper_executable(which=shutil.which, windows=None):
    """The whisper.cpp command-line program that is available, or None.

    Windows: the one in bin/ - fetched on demand by ensure_whisper_binary().
    Elsewhere nothing is fetched (the only prebuilt build known here is a
    Windows one), so: bin/whisper-cli if the user put one there, else
    'whisper-cli' (Homebrew, a source build) or 'whisper-cpp' on the PATH.
    """
    windows = _IS_WINDOWS if windows is None else windows
    if os.path.exists(WHISPER_EXE):
        return WHISPER_EXE
    if windows:
        return None
    for name in ("whisper-cli", "whisper-cpp"):
        found = which(name)
        if found:
            return found
    return None


def local_engine_problem(which=shutil.which, windows=None):
    """Why the local engine cannot run right now, or None if it can.

    Windows can always fetch what is missing; elsewhere a missing whisper-cli
    is the user's to install - and worth saying before a recording rather than
    after it.
    """
    windows = _IS_WINDOWS if windows is None else windows
    if windows or find_whisper_executable(which=which, windows=windows):
        return None
    return NO_WHISPER_HINT.format(bin_dir=BIN_DIR)


def ensure_whisper_binary(progress=None, log=None, cancelled=None):
    """Make sure whisper-cli (whisper-cli.exe on Windows) is available."""
    found = find_whisper_executable()
    if found:
        return found
    if not _IS_WINDOWS:
        raise DownloadError(NO_WHISPER_HINT.format(bin_dir=BIN_DIR))

    # Check, lock, check again: whoever waited here finds it done afterwards.
    with _lock_for(WHISPER_EXE):
        if os.path.exists(WHISPER_EXE):
            return WHISPER_EXE

        os.makedirs(BIN_DIR, exist_ok=True)
        zip_path = os.path.join(BIN_DIR, "whisper-vulkan.zip")
        if log:
            log("Downloading whisper.cpp…\n")
        download(WHISPER_ZIP_URL, zip_path, "whisper.cpp", progress,
                 cancelled=cancelled)

        try:
            if log:
                log("Extracting archive…\n")
            safe_extract(zip_path, BIN_DIR)
        finally:
            try:
                os.remove(zip_path)
            except OSError:
                pass

        # Remove the example programs shipped in the release
        for item in os.listdir(BIN_DIR):
            if item.endswith(".exe") and item not in ESSENTIAL:
                try:
                    os.remove(os.path.join(BIN_DIR, item))
                except OSError:
                    pass

        if not os.path.exists(WHISPER_EXE):
            raise DownloadError("whisper-cli.exe was not contained in the "
                                "downloaded archive.")
        if log:
            log("whisper.cpp is ready.\n")
        return WHISPER_EXE


def _usable(path, minimum):
    return os.path.exists(path) and os.path.getsize(path) > minimum


def model_present(name):
    """Whether the model is on this computer, ready to use."""
    return _usable(model_path(name), 1 << 20)


def ensure_model(name, progress=None, log=None, cancelled=None):
    """Make sure the ggml model is present."""
    path = model_path(name)
    if _usable(path, 1 << 20):
        return path

    with _lock_for(path):
        if _usable(path, 1 << 20):
            return path
        if os.path.exists(path):
            # File exists but is obviously too small to be usable
            os.remove(path)
        if log:
            log(f"Downloading model '{name}'…\n")
        return download(MODEL_URL_TEMPLATE.format(name=name), path,
                        f"Whisper model '{name}'", progress, cancelled=cancelled)


def ensure_vad_model(progress=None, log=None, cancelled=None):
    """Silero VAD for whisper.cpp (0.9 MB).

    Covers M11: hallucinations during pauses are prevented at the source
    instead of being filtered out afterwards via RMS thresholds. Off by
    default - see the note in whispercpp.py.
    """
    if _usable(VAD_MODEL_PATH, 100_000):
        return VAD_MODEL_PATH

    with _lock_for(VAD_MODEL_PATH):
        if _usable(VAD_MODEL_PATH, 100_000):
            return VAD_MODEL_PATH
        if log:
            log("Downloading voice activity model (VAD)…\n")
        return download(VAD_MODEL_URL, VAD_MODEL_PATH, "VAD model", progress,
                        cancelled=cancelled)
