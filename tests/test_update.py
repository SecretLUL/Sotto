"""Updating a packaged build: finding a newer release, fetching it safely, and
swapping the program files without touching the data next to them."""

import hashlib
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from unittest.mock import patch

import build_release
from audio_transcriber import update, version
from audio_transcriber.events import UpdateChecked, UpdateFetch, UpdateFetchEnded
from audio_transcriber.transcribe import binaries

WAIT = 5.0
TAG = "v2.1.0"


def _release_json(tag=TAG, names=None, **extra):
    names = names if names is not None else [
        f"Sotto-{tag}-windows-x64.zip", "SHA256SUMS.txt"]
    data = {
        "tag_name": tag,
        "html_url": f"https://github.com/SecretLUL/Sotto/releases/tag/{tag}",
        "draft": False, "prerelease": False,
        "assets": [{"name": name, "size": 1000,
                    "browser_download_url":
                        f"https://github.com/SecretLUL/Sotto/releases/download/{tag}/{name}"}
                   for name in names],
    }
    data.update(extra)
    return data


def _write_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class Bridge:
    def __init__(self):
        self.events = queue.Queue()

    def post(self, event):
        self.events.put(event)

    def all(self):
        items = []
        while not self.events.empty():
            items.append(self.events.get())
        return items


# ----------------------------------------------------------------------
class TestVersion(unittest.TestCase):
    def test_what_counts_as_newer(self):
        cases = [
            ("v2.0.2", "v2.0.1", True),
            ("v2.1.0", "v2.0.9", True),
            ("v10.0.0", "v9.9.9", True),         # numbers, not text
            ("v2.0.1", "v2.0.1", False),
            ("v2.0.0", "v2.0.1", False),
            ("v2.1.0", "v2.1.0-rc1", True),      # the release after its pre-release
            ("v2.1.0-rc1", "v2.1.0", False),
            ("v2.1.0-rc2", "v2.1.0-rc1", True),
            ("2.0.2", "v2.0.1", True),           # the "v" is optional
            ("latest", "v2.0.1", False),         # not a version: never newer
            ("v2.0.2", "nonsense", False),
        ]
        for candidate, installed, expected in cases:
            with self.subTest(candidate=candidate, installed=installed):
                self.assertEqual(version.is_newer(candidate, installed), expected)

    def test_the_version_comes_from_the_release_file(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        path = os.path.join(folder, "RELEASE")
        self.assertIsNone(version.current(path), "from source: no file, no version")
        for written, expected in (("v2.0.1\n", "v2.0.1"), ("2.0.1", "v2.0.1"),
                                  ("v2.1.0-rc1", "v2.1.0-rc1"),
                                  ("v0.0.0-dev", None),     # a hand-made build
                                  ("garbage", None)):
            with self.subTest(written=written):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(written)
                self.assertEqual(version.current(path), expected)

    def test_the_archive_is_named_as_the_updater_looks_for_it(self):
        self.assertEqual(version.archive_name("v2.0.2"),
                         "Sotto-v2.0.2-windows-x64.zip")

    def test_the_build_writes_the_version_where_the_app_reads_it(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        with patch.object(build_release, "BUILD_DIR", folder):
            path = build_release.write_release_file("v2.1.0")
        self.assertEqual(os.path.basename(path),
                         os.path.basename(version.RELEASE_FILE))
        self.assertEqual(version.current(path), "v2.1.0")


# ----------------------------------------------------------------------
class TestFindingARelease(unittest.TestCase):
    def test_it_picks_the_windows_download(self):
        release = update.release_from_json(_release_json())
        self.assertEqual(release.tag, TAG)
        self.assertEqual(release.archive.name, f"Sotto-{TAG}-windows-x64.zip")
        self.assertEqual(release.sums.name, "SHA256SUMS.txt")

    def test_a_release_without_a_windows_download_says_so(self):
        release = update.release_from_json(
            _release_json(names=["SHA256SUMS.txt"]))
        self.assertIsNone(release.archive)

    def test_drafts_pre_releases_and_odd_tags_are_not_offered(self):
        for data in (_release_json(draft=True), _release_json(prerelease=True),
                     _release_json(tag="nightly")):
            with self.subTest(data=data["tag_name"]):
                self.assertIsNone(update.release_from_json(data))

    def test_only_https_downloads_and_github_pages_are_taken(self):
        data = _release_json(html_url="javascript:alert(1)")
        data["assets"][0]["browser_download_url"] = "http://example.com/x.zip"
        release = update.release_from_json(data)
        self.assertIsNone(release.archive)
        self.assertEqual(release.page_url, update.RELEASES_PAGE)

    def test_only_a_newer_release_is_an_update(self):
        fetch = lambda: _release_json()
        self.assertEqual(update.check_for_update("v2.0.2", fetch=fetch).tag, TAG)
        self.assertIsNone(update.check_for_update(TAG, fetch=fetch))
        self.assertIsNone(update.check_for_update("v3.0.0", fetch=fetch))

    def test_githubs_answer_is_read(self):
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        seen = []

        def urlopen(request, timeout):
            seen.append(request)
            return Response(json.dumps(_release_json()).encode())

        self.assertEqual(update.fetch_latest(urlopen=urlopen)["tag_name"], TAG)
        self.assertEqual(seen[0].full_url, update.LATEST_URL)

    def test_network_trouble_becomes_a_sentence(self):
        def failing(error):
            def urlopen(request, timeout):
                raise error
            return urlopen

        cases = (
            (urllib.error.HTTPError(update.LATEST_URL, 404, "Not Found", {}, None),
             "No release"),
            (urllib.error.HTTPError(update.LATEST_URL, 403, "Forbidden", {}, None),
             "limiting"),
            (urllib.error.URLError("no route"), "Could not reach GitHub"),
        )
        for error, words in cases:
            with self.subTest(error=error):
                with self.assertRaises(update.UpdateError) as caught:
                    update.fetch_latest(urlopen=failing(error))
                self.assertIn(words, str(caught.exception))


# ----------------------------------------------------------------------
class TestWhereItCanInstallItself(unittest.TestCase):
    def problem(self, **overrides):
        options = dict(frozen=True,
                       executable=os.path.join("C:\\", "Apps", "Sotto", "Sotto.exe"),
                       writable=lambda _folder: True)
        options.update(overrides)
        return update.install_problem(**options)

    def test_a_packaged_build_in_a_folder_of_its_own_can(self):
        self.assertIsNone(self.problem())

    def test_where_it_cannot_it_says_why(self):
        self.assertIn("git pull", self.problem(frozen=False))
        self.assertIn("Sotto.exe", self.problem(
            executable=os.path.join("C:\\", "Apps", "Renamed.exe")))
        self.assertIn("read-only", self.problem(writable=lambda _folder: False))


# ----------------------------------------------------------------------
class FakeServer:
    """Stands in for binaries.download: serves local files by URL."""

    def __init__(self):
        self.files = {}
        self.requests = []

    def add(self, asset, path):
        self.files[asset.url] = path

    def download(self, url, dest_path, description="File", progress=None,
                 timeout=0, cancelled=None, on_bytes=None):
        self.requests.append(url)
        if cancelled is not None and cancelled():
            raise binaries.DownloadError(f"{description}: download cancelled.")
        shutil.copyfile(self.files[url], dest_path)
        if on_bytes is not None:
            size = os.path.getsize(dest_path)
            on_bytes(size // 2, size)
            on_bytes(size, size)
        return dest_path


class WorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.workspace = os.path.join(self.root, "Sotto", update.WORKSPACE_NAME)
        os.makedirs(self.workspace)
        self.server = FakeServer()
        self.files = os.path.join(self.root, "served")
        os.makedirs(self.files)

    def publish(self, files, sums=None):
        """A release whose archive holds `files`."""
        name = version.archive_name(TAG)
        path = os.path.join(self.files, name)
        _write_zip(path, files)
        sums_path = os.path.join(self.files, update.SUMS_NAME)
        with open(sums_path, "w", encoding="utf-8") as handle:
            handle.write(sums if sums is not None
                         else f"{_sha256(path)}  {name}\n{'0' * 64}  other.zip\n")
        release = update.Release(
            TAG, update.RELEASES_PAGE,
            archive=update.Asset(name, f"https://example.test/{name}",
                                 os.path.getsize(path)),
            sums=update.Asset(update.SUMS_NAME, "https://example.test/sums"))
        self.server.add(release.archive, path)
        self.server.add(release.sums, sums_path)
        return release

    def prepare(self, release, **options):
        return update.prepare(release, self.workspace,
                              download=self.server.download, **options)


class TestPrepare(WorkspaceTestCase):
    WINDOWS_BUILD = {"Sotto/Sotto.exe": b"new exe",
                     "Sotto/_internal/python312.dll": b"new dll"}

    def test_it_downloads_verifies_and_unpacks(self):
        release = self.publish(self.WINDOWS_BUILD)
        steps = []
        staged = self.prepare(release, log=steps.append)
        with open(os.path.join(staged, "Sotto.exe"), "rb") as handle:
            self.assertEqual(handle.read(), b"new exe")
        self.assertEqual(sorted(os.listdir(staged)), ["Sotto.exe", "_internal"])
        self.assertFalse(os.path.exists(os.path.join(self.workspace, update.DOWNLOAD_DIR)),
                         "the archive is removed once it is unpacked")
        self.assertEqual(self.server.requests[0], release.sums.url,
                         "the checksums come first")
        self.assertIn(f"Downloading Sotto {TAG}…", steps)

    def test_a_download_that_does_not_match_its_checksum_is_refused(self):
        release = self.publish(self.WINDOWS_BUILD,
                               sums=f"{'a' * 64}  {version.archive_name(TAG)}\n")
        with self.assertRaises(update.UpdateError) as caught:
            self.prepare(release)
        self.assertIn("checksum", str(caught.exception))
        self.assertFalse(os.path.exists(os.path.join(self.workspace, update.STAGED_DIR)))

    def test_a_download_without_a_checksum_is_not_installed(self):
        release = self.publish(self.WINDOWS_BUILD, sums=f"{'a' * 64}  other.zip\n")
        with self.assertRaises(update.UpdateError):
            self.prepare(release)
        no_sums = update.Release(TAG, update.RELEASES_PAGE, archive=release.archive)
        with self.assertRaises(update.UpdateError):
            self.prepare(no_sums)
        self.assertEqual(len(self.server.requests), 1,
                         "without SHA256SUMS.txt nothing is even downloaded")

    def test_a_release_without_a_windows_build_is_refused(self):
        release = update.Release(TAG, update.RELEASES_PAGE,
                                 sums=update.Asset("SHA256SUMS.txt", "https://x"))
        with self.assertRaises(update.UpdateError) as caught:
            self.prepare(release)
        self.assertIn("windows-x64", str(caught.exception))

    def test_an_archive_that_would_replace_the_users_data_is_refused(self):
        for data in ("Sotto/bin/ggml-small.bin", "Sotto/settings.json",
                     "Sotto/output/meeting.wav", "Sotto/Output/x.txt"):
            with self.subTest(data=data):
                release = self.publish({**self.WINDOWS_BUILD, data: b"x"})
                with self.assertRaises(update.UpdateError) as caught:
                    self.prepare(release)
                self.assertIn("your own data", str(caught.exception))

    def test_an_archive_that_reaches_outside_its_folder_is_refused(self):
        release = self.publish({**self.WINDOWS_BUILD, "../escaped.txt": b"x"})
        with self.assertRaises((update.UpdateError, binaries.DownloadError)):
            self.prepare(release)
        self.assertFalse(os.path.exists(os.path.join(self.workspace, "escaped.txt")))

    def test_an_archive_of_some_other_shape_is_refused(self):
        for files in ({"Other/Sotto.exe": b"x"},
                      {**self.WINDOWS_BUILD, "README.txt": b"x"},
                      {"Sotto/_internal/python312.dll": b"no exe"}):
            with self.subTest(files=sorted(files)):
                with self.assertRaises(update.UpdateError):
                    self.prepare(self.publish(files))

    def test_it_needs_room_on_the_disk(self):
        class Usage:
            free = 10
        release = self.publish(self.WINDOWS_BUILD)
        with self.assertRaises(update.UpdateError) as caught:
            self.prepare(release, disk_usage=lambda _path: Usage())
        self.assertIn("free space", str(caught.exception))

    def test_leftovers_of_an_earlier_attempt_are_cleared_first(self):
        stale = os.path.join(self.workspace, update.STAGED_DIR, "Sotto", "old.txt")
        os.makedirs(os.path.dirname(stale))
        open(stale, "w").close()
        staged = self.prepare(self.publish(self.WINDOWS_BUILD))
        self.assertNotIn("old.txt", os.listdir(staged))


# ----------------------------------------------------------------------
class TestInstall(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.target = os.path.dirname(self.workspace)
        self.write(self.target, "Sotto.exe", "old exe")
        self.write(self.target, "_internal/python312.dll", "old dll")
        self.write(self.target, "_internal/gone.pyd", "only in the old version")
        # What a portable install keeps next to the program.
        self.write(self.target, "bin/ggml-small.bin", "model")
        self.write(self.target, "output/meeting.txt", "transcript")
        self.write(self.target, "settings.json", "{}")
        self.source = os.path.join(self.root, "staged", "Sotto")
        self.write(self.source, "Sotto.exe", "new exe")
        self.write(self.source, "_internal/python312.dll", "new dll")
        self.write(self.source, "_internal/new.pyd", "only in the new version")
        patcher = patch.object(update, "RENAME_ATTEMPTS", 1)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def write(folder, relative, text):
        path = os.path.join(folder, *relative.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def read(self, relative):
        with open(os.path.join(self.target, *relative.split("/")), encoding="utf-8") as handle:
            return handle.read()

    def test_the_program_is_replaced_and_the_data_left_alone(self):
        update.install(self.source, self.target, self.workspace, exe_name="Sotto.exe")
        self.assertEqual(self.read("Sotto.exe"), "new exe")
        self.assertEqual(self.read("_internal/python312.dll"), "new dll")
        self.assertEqual(sorted(os.listdir(os.path.join(self.target, "_internal"))),
                         ["new.pyd", "python312.dll"],
                         "_internal is the new one as a whole, not a mix")
        self.assertEqual(self.read("bin/ggml-small.bin"), "model")
        self.assertEqual(self.read("output/meeting.txt"), "transcript")
        self.assertEqual(self.read("settings.json"), "{}")
        self.assertTrue(os.path.isdir(self.source), "the source is copied, not moved")

    def test_a_failure_half_way_puts_the_old_version_back(self):
        real_rename = os.rename
        calls = []

        def rename(source, dest):
            calls.append((source, dest))
            if len(calls) == 3:          # _internal is in, Sotto.exe moving aside
                raise PermissionError("in use")
            real_rename(source, dest)

        with patch("os.rename", side_effect=rename):
            with self.assertRaises(PermissionError):
                update.install(self.source, self.target, self.workspace,
                               exe_name="Sotto.exe")
        self.assertEqual(self.read("Sotto.exe"), "old exe")
        self.assertEqual(sorted(os.listdir(os.path.join(self.target, "_internal"))),
                         ["gone.pyd", "python312.dll"])
        self.assertEqual(self.read("bin/ggml-small.bin"), "model")

    def test_program_files_that_would_replace_data_are_refused(self):
        self.write(self.source, "settings.json", "{\"evil\": true}")
        with self.assertRaises(update.UpdateError):
            update.install(self.source, self.target, self.workspace,
                           exe_name="Sotto.exe")
        self.assertEqual(self.read("settings.json"), "{}")
        self.assertEqual(self.read("Sotto.exe"), "old exe")


class TestInstaller(TestInstall):
    """The installer process: main.py hands it the command line."""

    def run_installer(self, *extra):
        launched = []
        with patch.object(update, "_launch",
                          side_effect=lambda command, cwd: launched.append(command)), \
             patch.object(version, "current", return_value=TAG):
            code = update.apply_from_command_line(
                ["--apply-update", "--source", self.source, "--target", self.target,
                 *extra])
        self.assertEqual(code, 0)
        return launched

    def test_it_installs_and_leaves_word_for_the_next_start(self):
        launched = self.run_installer()
        self.assertEqual(self.read("_internal/python312.dll"), "new dll")
        self.assertEqual(launched, [], "closing Sotto installs without starting it again")
        self.assertEqual(update.take_result(self.workspace),
                         {"ok": True, "tag": TAG, "error": ""})
        self.assertIsNone(update.take_result(self.workspace), "said once")
        with open(os.path.join(self.workspace, update.LOG_FILE), encoding="utf-8") as log:
            self.assertIn("Installed", log.read())

    def test_restart_now_starts_the_new_version(self):
        launched = self.run_installer("--relaunch")
        self.assertEqual(launched,
                         [[os.path.join(self.target, update.EXE_NAME)]])

    def test_a_failed_install_is_reported_and_the_old_version_restarted(self):
        shutil.rmtree(os.path.join(self.source, "_internal"))
        self.write(self.source, "bin/x", "a data folder in the download")
        launched = self.run_installer("--relaunch")
        result = update.take_result(self.workspace)
        self.assertFalse(result["ok"])
        self.assertIn("bin", result["error"])
        self.assertEqual(self.read("_internal/python312.dll"), "old dll")
        self.assertEqual(len(launched), 1)

    def test_the_old_version_starts_it_with_the_agreed_command_line(self):
        started = []
        update.start_install(self.source, target=self.target, relaunch=True, pid=4321,
                             popen=lambda command, **options: started.append(
                                 (command, options)))
        command, options = started[0]
        self.assertEqual(command, [os.path.join(self.source, update.EXE_NAME),
                                   "--apply-update", "--source", self.source,
                                   "--target", self.target, "--wait-pid", "4321",
                                   "--relaunch"])
        self.assertEqual(options["cwd"], self.source,
                         "not from the folder whose files it replaces")


class TestCleanup(WorkspaceTestCase):
    def test_what_an_update_left_is_cleared_but_the_log_stays(self):
        for name in (update.DOWNLOAD_DIR, update.STAGED_DIR, update.INCOMING_DIR,
                     update.BACKUP_DIR):
            os.makedirs(os.path.join(self.workspace, name, "Sotto"))
        update._log(self.workspace, "something happened")
        update.cleanup(self.workspace)
        self.assertEqual(os.listdir(self.workspace), [update.LOG_FILE])

    def test_read_only_files_go_too(self):
        path = os.path.join(self.workspace, update.BACKUP_DIR, "Sotto.exe")
        os.makedirs(os.path.dirname(path))
        open(path, "w").close()
        os.chmod(path, 0o444)
        update.cleanup(self.workspace)
        self.assertFalse(os.path.exists(path))


# ----------------------------------------------------------------------
class TestProcesses(unittest.TestCase):
    def test_waiting_for_a_process_that_has_ended(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()
        self.assertTrue(update.wait_for_exit(process.pid, 2.0))

    def test_waiting_for_one_that_keeps_running_gives_up(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertFalse(update.wait_for_exit(process.pid, 0.3))
        finally:
            process.kill()
            process.wait()

    def test_the_new_version_does_not_inherit_the_old_bundles_environment(self):
        env = {"PATH": "x", "_PYI_APPLICATION_HOME_DIR": "old", "_MEIPASS2": "old",
               "TCL_LIBRARY": "old/_tcl_data", "TK_LIBRARY": "old/_tk_data"}
        cleaned = update.child_environment(env, frozen=True)
        self.assertEqual(cleaned, {"PATH": "x", "PYINSTALLER_RESET_ENVIRONMENT": "1"})
        self.assertEqual(update.child_environment(env, frozen=False), env)


# ----------------------------------------------------------------------
class TestWorkers(WorkspaceTestCase):
    def test_the_check_reports_a_newer_release(self):
        bridge = Bridge()
        worker = update.UpdateCheck(bridge, "v2.0.2", fetch=lambda: _release_json())
        worker.start().join(WAIT)
        (event,) = bridge.all()
        self.assertEqual(event.release.tag, TAG)

    def test_the_check_reports_nothing_new_and_trouble_alike(self):
        bridge = Bridge()
        update.UpdateCheck(bridge, TAG, fetch=lambda: _release_json()).start().join(WAIT)

        def offline():
            raise update.UpdateError("Could not reach GitHub (offline).")

        update.UpdateCheck(bridge, TAG, fetch=offline).start().join(WAIT)
        self.assertEqual(bridge.all(), [UpdateChecked(),
                                        UpdateChecked(error="Could not reach GitHub (offline).")])

    def download(self, prepare_fn):
        bridge = Bridge()
        release = self.publish(TestPrepare.WINDOWS_BUILD)
        worker = update.UpdateDownload(bridge, release, self.workspace,
                                       prepare_fn=prepare_fn)
        thread = worker.start()
        thread.join(WAIT)
        self.assertFalse(thread.is_alive())
        return worker, bridge.all()

    def test_the_download_reports_its_way_and_where_the_result_is(self):
        def prepare_fn(release, workspace_dir, **options):
            return update.prepare(release, workspace_dir,
                                  download=self.server.download, **options)

        _worker, events = self.download(prepare_fn)
        steps = [event.step for event in events if isinstance(event, UpdateFetch)]
        self.assertIn(f"Downloading Sotto {TAG}…", steps)
        self.assertTrue(any(event.total for event in events
                            if isinstance(event, UpdateFetch)))
        last = events[-1]
        self.assertIsInstance(last, UpdateFetchEnded)
        self.assertTrue(os.path.isfile(os.path.join(last.staged, "Sotto.exe")))

    def test_a_failed_download_says_why(self):
        def prepare_fn(release, workspace_dir, **options):
            raise binaries.DownloadError("server responded with HTTP 500.")

        _worker, events = self.download(prepare_fn)
        self.assertEqual(events[-1], UpdateFetchEnded(
            TAG, error="server responded with HTTP 500."))

    def test_a_cancelled_download_is_just_cancelled(self):
        holder = {}

        def prepare_fn(release, workspace_dir, cancelled=None, **options):
            holder["worker"].cancel()
            if cancelled():
                raise binaries.DownloadError("download cancelled.")

        bridge = Bridge()
        release = self.publish(TestPrepare.WINDOWS_BUILD)
        worker = holder["worker"] = update.UpdateDownload(
            bridge, release, self.workspace, prepare_fn=prepare_fn)
        worker.start().join(WAIT)
        self.assertEqual(bridge.all(), [UpdateFetchEnded(TAG, cancelled=True)])


# ----------------------------------------------------------------------
try:
    import tkinter as tk
    _HAS_TK = True
except Exception:                                   # pragma: no cover
    _HAS_TK = False


def _can_open_window():
    if not _HAS_TK:
        return False
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestUpdateInTheWindow(unittest.TestCase):
    RELEASE = update.Release(
        TAG, "https://github.com/SecretLUL/Sotto/releases/tag/v2.1.0",
        archive=update.Asset("Sotto-v2.1.0-windows-x64.zip", "https://x/a.zip", 100),
        sums=update.Asset("SHA256SUMS.txt", "https://x/sums"))

    def setUp(self):
        from audio_transcriber import config
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons
        from audio_transcriber.ui.app import RecorderApp

        icons._ICON_CACHE.clear()
        self.patches = [
            patch.object(config, "load", return_value=(config.Settings(), [])),
            patch.object(AudioEngine, "configure", return_value=[]),
            # This build is a release that can install itself.
            patch.object(version, "current", return_value="v2.0.2"),
            patch.object(update, "install_problem", return_value=None),
            patch.object(update, "start_install"),
        ]
        for patcher in self.patches:
            patcher.start()
        self.start_install = update.start_install
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.closed = False

    def tearDown(self):
        from audio_transcriber.ui import icons
        if not self.closed:
            with patch("audio_transcriber.ui.app.messagebox") as box:
                box.askyesno.return_value = True
                self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        icons._ICON_CACHE.clear()

    def deliver(self, event):
        self.app.bridge.post(event)
        self.app.bridge.drain_now()
        self.root.update_idletasks()

    def banner_shown(self):
        return bool(self.app.update_banner.winfo_manager())

    def shown_buttons(self):
        return [button["text"] for button in
                self.app.update_action_btn.master.pack_slaves()]

    def test_no_banner_while_there_is_nothing_new(self):
        self.assertFalse(self.banner_shown())
        self.deliver(UpdateChecked())
        self.assertFalse(self.banner_shown())
        self.assertIn("newest", self.app.update_info["text"])

    def test_a_new_version_brings_the_banner_up(self):
        self.deliver(UpdateChecked(release=self.RELEASE))
        self.assertTrue(self.banner_shown())
        self.assertIn(TAG, self.app.update_title["text"])
        self.assertIn("v2.0.2", self.app.update_detail["text"])
        self.assertEqual(self.shown_buttons(), ["What's new", "Update now", "Later"])

    def test_later_puts_it_away_for_this_session(self):
        self.deliver(UpdateChecked(release=self.RELEASE))
        self.app.update_later_btn.invoke()
        self.assertFalse(self.banner_shown())

    def test_where_it_cannot_install_itself_it_offers_the_download_page(self):
        self.deliver(UpdateChecked(release=self.RELEASE))
        with patch.object(update, "install_problem", return_value="Read-only folder."), \
             patch("audio_transcriber.ui.app.webbrowser.open") as browser:
            self.app._refresh_update_banner()
            self.assertEqual(self.app.update_action_btn["text"], "Download")
            self.assertIn("Read-only folder.", self.app.update_detail["text"])
            self.app.update_action_btn.invoke()
        browser.assert_called_once_with(self.RELEASE.page_url)

    def test_update_now_downloads_then_offers_the_restart(self):
        self.deliver(UpdateChecked(release=self.RELEASE))
        with patch.object(update, "UpdateDownload") as download:
            download.return_value.cancelled = False
            self.app.update_action_btn.invoke()
        download.return_value.start.assert_called_once()
        self.assertEqual(self.app.update_action_btn["text"], "Cancel")
        self.assertEqual(self.shown_buttons(), ["Cancel"])

        self.deliver(UpdateFetch(f"Downloading Sotto {TAG}…", done=50, total=100,
                                 rate=10.0))
        self.assertEqual(self.app.update_percent["text"], "50 %")

        self.deliver(UpdateFetchEnded(TAG, staged="C:/staged/Sotto"))
        self.assertEqual(self.app.update_action_btn["text"], "Restart now")
        self.assertIn("ready", self.app.update_title["text"])

    def test_a_failed_download_can_be_tried_again(self):
        self.deliver(UpdateChecked(release=self.RELEASE))
        self.app._update_download = object()
        self.deliver(UpdateFetchEnded(TAG, error="The connection was lost."))
        self.assertEqual(self.app.update_action_btn["text"], "Try again")
        self.assertEqual(self.app.update_detail["text"], "The connection was lost.")

    def test_restart_now_hands_over_to_the_installer_and_closes(self):
        self.deliver(UpdateChecked(release=self.RELEASE))
        self.deliver(UpdateFetchEnded(TAG, staged="C:/staged/Sotto"))
        with patch.object(self.app, "on_close") as close:
            self.app.update_action_btn.invoke()
        self.start_install.assert_called_once_with("C:/staged/Sotto", relaunch=True)
        close.assert_called_once()

    def test_no_restart_in_the_middle_of_a_recording(self):
        self.deliver(UpdateChecked(release=self.RELEASE))
        self.deliver(UpdateFetchEnded(TAG, staged="C:/staged/Sotto"))
        with patch.object(self.app, "_busy_with_work", return_value=True), \
             patch("audio_transcriber.ui.app.messagebox") as box:
            self.app.update_action_btn.invoke()
        box.showinfo.assert_called_once()
        self.start_install.assert_not_called()

    def test_closing_with_a_downloaded_update_installs_it(self):
        self.deliver(UpdateChecked(release=self.RELEASE))
        self.deliver(UpdateFetchEnded(TAG, staged="C:/staged/Sotto"))
        self.app.on_close()
        self.closed = True
        self.start_install.assert_called_once_with("C:/staged/Sotto", relaunch=False)

    def test_closing_without_one_installs_nothing(self):
        self.app.on_close()
        self.closed = True
        self.start_install.assert_not_called()

    def test_the_start_reports_the_last_update_and_asks_github(self):
        with patch.object(update, "take_result",
                          return_value={"ok": True, "tag": "v2.0.2", "error": ""}), \
             patch.object(update, "cleanup"), \
             patch.object(update, "UpdateCheck") as check:
            self.app.start_update_check()
        check.assert_called_once_with(self.app.bridge, "v2.0.2")
        self.assertIn("updated to v2.0.2",
                      self.app.transcript.text.get("1.0", tk.END))

    def test_the_switch_keeps_the_start_from_asking(self):
        self.app.update_auto_var.set(False)
        self.app._sync_settings_from_ui()
        self.assertFalse(self.app.settings.check_for_updates)
        with patch.object(update, "take_result", return_value=None), \
             patch.object(update, "cleanup"), \
             patch.object(update, "UpdateCheck") as check:
            self.app.start_update_check()
        check.assert_not_called()


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestFromSource(unittest.TestCase):
    def test_a_run_from_source_never_asks(self):
        from audio_transcriber import config
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui.app import RecorderApp

        with patch.object(config, "load", return_value=(config.Settings(), [])), \
             patch.object(AudioEngine, "configure", return_value=[]), \
             patch.object(version, "current", return_value=None), \
             patch.object(update, "UpdateCheck") as check:
            root = tk.Tk()
            root.withdraw()
            app = RecorderApp(root)
            try:
                app.start_update_check()
                app.check_for_updates(manual=True)
                check.assert_not_called()
                self.assertEqual(str(app.update_check_btn["state"]), "disabled")
                self.assertIn("git pull", app.update_info["text"])
            finally:
                app.on_close()


if __name__ == "__main__":
    unittest.main()
