"""Automated build script for Sotto.

Builds a standalone executable using PyInstaller, prunes heavy unused packages
from the local environment, and packages the result into a release archive.

Runs on Windows, Linux and macOS - the CI calls this same script on all three
so the archive naming lives in exactly one place.

Version resolution, highest priority first:
    1. python build_release.py v1.2.3
    2. RELEASE_VERSION=v1.2.3 python build_release.py   (set by the CI from the tag)
    3. the git tag pointing at HEAD
    4. v0.0.0-dev
"""

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

from audio_transcriber import version as app_version

# The executable, the archives and the folder they unpack to. Builds from before
# Sotto were called AudioTranscriber; a portable install of one keeps its models,
# recordings and settings.json in that folder, and Sotto moves them over on its
# first start (paths.migrate_legacy_data).
APP_NAME = app_version.APP_NAME
FALLBACK_VERSION = "v0.0.0-dev"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(ROOT_DIR, "dist")
BUILD_DIR = os.path.join(ROOT_DIR, "build")
APP_DIST_DIR = os.path.join(DIST_DIR, APP_NAME)
MAC_APP_DIR = os.path.join(DIST_DIR, f"{APP_NAME}.app")

# Heavy packages from the local Python environment that the app does not use.
# NOTE: Pillow (PIL) must NOT be listed here - ui/icons.py imports it at module
# level to render the vector icons, so excluding it produces a build that dies
# with "No module named 'PIL'" on the very first import.
UNNEEDED_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "pandas",
    "pandas.libs",
    "sklearn",
    "matplotlib",
    "pyarrow",
    "pyarrow.libs",
    "jedi",
    "grpc",
    "IPython",
    "notebook",
    "numba",
    "numba.libs",
    "llvmlite",
    "llvmlite.libs",
]


# ----------------------------------------------------------------------
def resolve_version():
    """Version for the archive name - see the module docstring for the order."""
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if args:
        return _normalize_version(args[0])

    from_env = os.environ.get("RELEASE_VERSION", "").strip()
    if from_env:
        return _normalize_version(from_env)

    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=10)
        if tag.returncode == 0 and tag.stdout.strip():
            return _normalize_version(tag.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass

    print(f"No version tag found - falling back to {FALLBACK_VERSION}")
    return FALLBACK_VERSION


def _normalize_version(value):
    """'1.2.3' and 'v1.2.3' both become 'v1.2.3'."""
    value = value.strip()
    return value if value.startswith("v") else f"v{value}"


def platform_tag():
    """(archive suffix, platform slug) for the current OS.

    Defined in the app (version.py): the updater looks for its download under
    the very name this build gives it.
    """
    return app_version.platform_tag()


# ----------------------------------------------------------------------
def clean():
    """Clean build artifacts."""
    print("Cleaning build directories...")
    for directory in (DIST_DIR, BUILD_DIR):
        if os.path.exists(directory):
            shutil.rmtree(directory, ignore_errors=True)
    spec_file = os.path.join(ROOT_DIR, f"{APP_NAME}.spec")
    if os.path.exists(spec_file):
        os.remove(spec_file)


def write_icon():
    """The executable's icon, drawn like the window's (ui/icons.py).

    Windows takes an .ico with every size in it; on macOS PyInstaller turns a
    PNG into the .icns itself (with Pillow). Linux executables have no icon.
    Kept apart from build/Sotto, which PyInstaller --clean empties.
    """
    if sys.platform not in ("win32", "darwin"):
        return None
    from audio_transcriber.ui import icons

    folder = os.path.join(BUILD_DIR, "icon")
    os.makedirs(folder, exist_ok=True)
    if sys.platform == "win32":
        path = os.path.join(folder, f"{APP_NAME}.ico")
        icons.render_icon_image("app_logo", 256).save(
            path, format="ICO",
            sizes=[(size, size) for size in (16, 20, 24, 32, 40, 48, 64, 128, 256)])
    else:
        path = os.path.join(folder, f"{APP_NAME}.png")
        icons.render_icon_image("app_logo", 1024).save(path, format="PNG")
    return path


def write_release_file(version):
    """The version file the app reads (version.current()), for the bundle.

    It is how a build knows which release it is - and so whether a newer one
    is out. Kept apart from build/Sotto, which PyInstaller --clean empties.
    """
    folder = os.path.join(BUILD_DIR, "release")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, os.path.basename(app_version.RELEASE_FILE))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(version + "\n")
    return path


def build_exe(icon=None, release_file=None):
    """Run PyInstaller to build the standalone executable directory."""
    print("Building executable with PyInstaller...")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name", APP_NAME,
        "--windowed",  # No console window
        "--onedir",    # Directory mode for fast launch & robust DLL loading
        "--noconfirm",
        "--clean",
        "--collect-all", "soundfile",
        "--hidden-import", "scipy.signal",
    ]
    if icon:
        cmd += ["--icon", icon]
    if release_file:
        # Next to the package's modules, where version.py looks for it.
        cmd += ["--add-data", f"{release_file}{os.pathsep}audio_transcriber"]

    # Only Windows has the WASAPI loopback fork; elsewhere plain PyAudio is
    # used and ui/app.py falls back to it at import time.
    if sys.platform == "win32":
        cmd += ["--collect-all", "pyaudiowpatch"]

    for mod in UNNEEDED_PACKAGES:
        cmd.extend(["--exclude-module", mod])

    cmd.append("main.py")

    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print("ERROR: PyInstaller build failed!")
        sys.exit(1)

    print(f"Executable built at: {APP_DIST_DIR}")


def prune_unneeded():
    """Prune leftover heavy directories collected by PyInstaller hooks."""
    internal_dir = os.path.join(APP_DIST_DIR, "_internal")
    if not os.path.exists(internal_dir):
        return

    print("Pruning unneeded heavy packages from _internal...")
    for item in os.listdir(internal_dir):
        if any(item.startswith(pkg) for pkg in UNNEEDED_PACKAGES):
            target = os.path.join(internal_dir, item)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
                print(f"  - Pruned directory: {item}")
            elif os.path.isfile(target):
                os.remove(target)
                print(f"  - Pruned file: {item}")


def verify_bundle():
    """Fail loudly if a module the app imports at startup is missing.

    An excluded dependency only shows up when a user double-clicks the binary,
    which is far too late - Pillow was silently pruned this way and every
    released Windows build died with "No module named 'PIL'".
    """
    internal_dir = os.path.join(APP_DIST_DIR, "_internal")
    if not os.path.exists(internal_dir):
        return

    entries = os.listdir(internal_dir)
    required = {
        "PIL": "Pillow (ui/icons.py renders the vector icons with it)",
        "soundfile": "soundfile (WAV reading/writing)",
    }

    missing = [f"{name} - {why}" for name, why in required.items()
               if not any(entry.split(".")[0] == name or entry.startswith(name + "-")
                          for entry in entries)]
    release = os.path.join(internal_dir, "audio_transcriber",
                           os.path.basename(app_version.RELEASE_FILE))
    if not os.path.isfile(release):
        missing.append("audio_transcriber/RELEASE - the version, without which "
                       "the build cannot tell whether an update is out")
    if missing:
        print("ERROR: the bundle is missing modules the app imports at startup:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)

    print("Bundle verified: all startup imports are present.")


def create_archive(archive_path, suffix):
    """Package the built application directory into the release archive."""
    print(f"Creating archive: {archive_path}...")

    # macOS --windowed produces a .app bundle next to the plain directory.
    source_dir = MAC_APP_DIR if os.path.isdir(MAC_APP_DIR) else APP_DIST_DIR
    arc_root = os.path.basename(source_dir)

    if suffix == "tar.gz":
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source_dir, arcname=arc_root)
    else:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _dirs, files in os.walk(source_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.join(
                        arc_root, os.path.relpath(full_path, source_dir))
                    zip_file.write(full_path, rel_path)

    size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"Release package created: {os.path.basename(archive_path)} ({size_mb:.1f} MB)")


def main():
    version = resolve_version()
    suffix, slug = platform_tag()
    archive_name = f"{APP_NAME}-{version}-{slug}.{suffix}"
    archive_path = os.path.join(DIST_DIR, archive_name)

    print(f"--- Building {APP_NAME} {version} for {slug} ---")
    clean()
    build_exe(icon=write_icon(), release_file=write_release_file(version))
    prune_unneeded()
    verify_bundle()
    create_archive(archive_path, suffix)
    print("\n--- RELEASE BUILD COMPLETE ---")
    print(f"Archive ready for GitHub Release: {archive_path}")


if __name__ == "__main__":
    main()
