"""Tests for the download and extraction layer (audit findings H8 and M2)."""

import http.client
import http.server
import os
import shutil
import tempfile
import threading
import time
import unittest
import zipfile
from unittest.mock import patch

from audio_transcriber.transcribe import binaries

PAYLOAD = b"x" * (256 * 1024)

# Larger than ensure_model()'s "obviously too small" threshold of 1 MiB, and
# served slowly enough that two downloaders really overlap.
BIG = b"m" * (1280 * 1024)


class _Handler(http.server.BaseHTTPRequestHandler):
    """A server with deliberately broken responses."""

    hits = []

    def log_message(self, *args):
        pass

    def _send(self, body, length=None, status=200):
        self.send_response(status)
        self.send_header("Content-Length",
                         str(length if length is not None else len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_slowly(self, body, chunk=64 * 1024, pause=0.03):
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            for start in range(0, len(body), chunk):
                self.wfile.write(body[start:start + chunk])
                self.wfile.flush()
                time.sleep(pause)
        except OSError:
            pass                     # the client gave up (cancelled)

    def do_GET(self):
        _Handler.hits.append(self.path)
        if self.path.startswith("/big"):
            self._send_slowly(BIG)
        elif self.path == "/ok":
            self._send(PAYLOAD)
        elif self.path == "/truncated":
            # Announces the full length but delivers only half of it
            self._send(PAYLOAD[:len(PAYLOAD) // 2], length=len(PAYLOAD))
        elif self.path == "/missing":
            self._send(b"not found", status=404)
        else:
            self._send(b"", status=400)


class _Server:
    def __enter__(self):
        _Handler.hits = []
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_successful_download(self):
        dest = os.path.join(self.dir, "model.bin")
        seen = []
        with _Server() as base:
            binaries.download(f"{base}/ok", dest, "Test file", progress=seen.append)
        self.assertEqual(os.path.getsize(dest), len(PAYLOAD))
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_truncated_download_leaves_no_file(self):
        """Regression H8: the previous version wrote straight to the target
        file. An abort left a partial file behind that passed as a valid model
        on the next start."""
        dest = os.path.join(self.dir, "model.bin")
        with _Server() as base:
            with self.assertRaises(binaries.DownloadError) as ctx:
                binaries.download(f"{base}/truncated", dest, "Test file")
        self.assertIn("incompletely", str(ctx.exception))
        self.assertFalse(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_http_error_is_reported(self):
        dest = os.path.join(self.dir, "model.bin")
        with _Server() as base:
            with self.assertRaises(binaries.DownloadError) as ctx:
                binaries.download(f"{base}/missing", dest, "Test file")
        self.assertIn("404", str(ctx.exception))
        self.assertFalse(os.path.exists(dest))

    def test_unreachable_host_is_reported(self):
        dest = os.path.join(self.dir, "model.bin")
        with self.assertRaises(binaries.DownloadError):
            binaries.download("http://127.0.0.1:1/nothing", dest, "Test file",
                              timeout=2)

    def test_stale_part_file_is_replaced(self):
        dest = os.path.join(self.dir, "model.bin")
        with open(dest + ".part", "wb") as handle:
            handle.write(b"garbage")
        with _Server() as base:
            binaries.download(f"{base}/ok", dest, "Test file")
        self.assertEqual(os.path.getsize(dest), len(PAYLOAD))


class TestConcurrentAndCancelledDownloads(unittest.TestCase):
    """The live preview and the closing pass can ask for the same model within
    seconds of each other, and the preview is stopped on the GUI thread."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_two_threads_asking_for_one_model_download_it_once(self):
        """Regression: the second downloader deleted the first one's .part file
        (a bare PermissionError on Windows) or fetched everything again."""
        target = os.path.join(self.dir, "ggml-tiny.bin")
        outcomes = []
        barrier = threading.Barrier(2)

        def ask():
            barrier.wait()
            try:
                outcomes.append(binaries.ensure_model("tiny"))
            except Exception as exc:                    # noqa: BLE001
                outcomes.append(exc)

        with _Server() as base:
            # Fake bytes under a real model name: the pinned hash must not apply.
            with patch.dict(binaries.EXPECTED_SHA256, {}, clear=True), \
                    patch.object(binaries, "model_path", lambda _name: target), \
                    patch.object(binaries, "MODEL_URL_TEMPLATE", base + "/big-{name}"):
                threads = [threading.Thread(target=ask) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=30)

        self.assertEqual(outcomes, [target, target], outcomes)
        self.assertEqual(os.path.getsize(target), len(BIG))
        self.assertEqual(len([h for h in _Handler.hits if h.startswith("/big")]), 1,
                         "the model was fetched more than once")
        self.assertFalse(os.path.exists(target + ".part"))

    def test_a_download_can_be_cancelled_between_blocks(self):
        dest = os.path.join(self.dir, "model.bin")
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()
        with _Server() as base:
            with self.assertRaises(binaries.DownloadError) as ctx:
                binaries.download(f"{base}/big", dest, "Test model",
                                  cancelled=stop.is_set)
        self.assertIn("cancelled", str(ctx.exception))
        self.assertFalse(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_a_connection_lost_mid_transfer_is_a_download_error(self):
        """IncompleteRead is not an OSError and used to escape as it was."""
        dest = os.path.join(self.dir, "model.bin")
        with patch("urllib.request.urlopen",
                   side_effect=http.client.IncompleteRead(b"partial", 100)):
            with self.assertRaises(binaries.DownloadError) as ctx:
                binaries.download("http://example.invalid/x", dest, "Test file")
        self.assertIn("connection was lost", str(ctx.exception))

    @unittest.skipUnless(os.name == "nt",
                         "only Windows refuses to delete an open file")
    def test_a_partial_file_in_use_is_a_download_error_not_a_permission_error(self):
        dest = os.path.join(self.dir, "model.bin")
        with open(dest + ".part", "wb") as holder:
            holder.write(b"another download is writing this")
            with self.assertRaises(binaries.DownloadError) as ctx:
                binaries.download("http://127.0.0.1:1/x", dest, "Test file",
                                  timeout=2)
        self.assertIn("in use", str(ctx.exception))

    def test_the_model_is_not_fetched_again_when_it_is_already_there(self):
        target = os.path.join(self.dir, "ggml-tiny.bin")
        with open(target, "wb") as handle:
            handle.write(b"x" * (2 << 20))
        with patch.object(binaries, "model_path", lambda _name: target), \
                patch.object(binaries, "download") as download:
            self.assertEqual(binaries.ensure_model("tiny"), target)
        download.assert_not_called()


class TestPinnedDownloads(unittest.TestCase):
    """What is downloaded - and in one case executed - is pinned and verified."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_every_model_the_app_offers_has_a_pinned_hash(self):
        from audio_transcriber import config
        names = [name for _label, name in config.MODEL_CHOICES if name]
        for name in names:
            with self.subTest(model=name):
                self.assertIn(f"ggml-{name}.bin", binaries.EXPECTED_SHA256)

    def test_the_hashes_look_like_sha256(self):
        self.assertIn("whisper-vulkan.zip", binaries.EXPECTED_SHA256)
        self.assertIn("ggml-silero-v5.1.2.bin", binaries.EXPECTED_SHA256)
        for name, digest in binaries.EXPECTED_SHA256.items():
            with self.subTest(file=name):
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_the_urls_name_a_revision_not_a_moving_branch(self):
        for url in (binaries.MODEL_URL_TEMPLATE, binaries.VAD_MODEL_URL):
            with self.subTest(url=url):
                self.assertRegex(url, r"/resolve/[0-9a-f]{40}/")
                self.assertNotIn("/resolve/main/", url)

    def test_a_download_with_the_right_checksum_is_accepted(self):
        import hashlib
        dest = os.path.join(self.dir, "model.bin")
        good = hashlib.sha256(PAYLOAD).hexdigest()
        with patch.dict(binaries.EXPECTED_SHA256, {"model.bin": good}):
            with _Server() as base:
                binaries.download(f"{base}/ok", dest, "Test file")
        self.assertEqual(os.path.getsize(dest), len(PAYLOAD))

    def test_a_download_with_a_wrong_checksum_is_rejected_and_leaves_nothing(self):
        """The checksum check existed but the table was empty, so a tampered or
        damaged model - or the executable fetched for Windows - passed."""
        dest = os.path.join(self.dir, "model.bin")
        with patch.dict(binaries.EXPECTED_SHA256, {"model.bin": "0" * 64}):
            with _Server() as base:
                with self.assertRaises(binaries.DownloadError) as ctx:
                    binaries.download(f"{base}/ok", dest, "Test file")
        self.assertIn("Checksum mismatch", str(ctx.exception))
        self.assertFalse(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))


class TestFindingWhisper(unittest.TestCase):
    """Only Windows gets a whisper-cli downloaded; elsewhere it has to exist."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.exe = os.path.join(self.dir, "whisper-cli")
        self.patches = [patch.object(binaries, "WHISPER_EXE", self.exe),
                        patch.object(binaries, "BIN_DIR", self.dir)]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _touch_exe(self):
        with open(self.exe, "wb"):
            pass

    @staticmethod
    def _which(*available):
        return lambda name: f"/usr/bin/{name}" if name in available else None

    def test_the_program_in_bin_wins_everywhere(self):
        self._touch_exe()
        for windows in (True, False):
            with self.subTest(windows=windows):
                self.assertEqual(
                    binaries.find_whisper_executable(
                        which=self._which("whisper-cli"), windows=windows),
                    self.exe)

    def test_windows_does_not_look_on_the_path(self):
        self.assertIsNone(binaries.find_whisper_executable(
            which=self._which("whisper-cli"), windows=True))

    def test_elsewhere_the_path_is_searched(self):
        self.assertEqual(
            binaries.find_whisper_executable(which=self._which("whisper-cli"),
                                             windows=False),
            "/usr/bin/whisper-cli")
        self.assertEqual(
            binaries.find_whisper_executable(which=self._which("whisper-cpp"),
                                             windows=False),
            "/usr/bin/whisper-cpp")
        self.assertIsNone(binaries.find_whisper_executable(
            which=self._which(), windows=False))

    def test_a_missing_program_is_reported_before_recording_not_after(self):
        self.assertIsNone(binaries.local_engine_problem(which=self._which(),
                                                        windows=True))
        self.assertIsNone(binaries.local_engine_problem(
            which=self._which("whisper-cli"), windows=False))
        problem = binaries.local_engine_problem(which=self._which(), windows=False)
        self.assertIn("brew install whisper-cpp", problem)
        self.assertIn(self.dir, problem)
        self.assertIn("ElevenLabs", problem)

    def test_nothing_is_downloaded_where_only_a_windows_build_exists(self):
        """It used to fetch the Windows archive on Linux and macOS, 'succeed',
        and then fail to start the .exe."""
        with patch.object(binaries, "_IS_WINDOWS", False), \
                patch.object(binaries, "find_whisper_executable", return_value=None), \
                patch.object(binaries, "download") as download:
            with self.assertRaises(binaries.DownloadError) as ctx:
                binaries.ensure_whisper_binary()
        download.assert_not_called()
        self.assertIn("brew install whisper-cpp", str(ctx.exception))


class TestSafeExtract(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_normal_archive(self):
        zip_path = os.path.join(self.dir, "good.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("whisper-cli.exe", b"MZ")
            archive.writestr("subfolder/whisper.dll", b"MZ")
        target = os.path.join(self.dir, "bin")
        os.makedirs(target)
        binaries.safe_extract(zip_path, target)
        self.assertTrue(os.path.exists(os.path.join(target, "whisper-cli.exe")))

    def test_path_traversal_is_rejected(self):
        """Regression M2 (zip slip): extractall() of the previous version
        would have placed this file outside the target directory."""
        zip_path = os.path.join(self.dir, "evil.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("harmless.txt", b"ok")
            archive.writestr("../../escaped.txt", b"pwned")
        target = os.path.join(self.dir, "bin")
        os.makedirs(target)

        with self.assertRaises(binaries.DownloadError) as ctx:
            binaries.safe_extract(zip_path, target)
        self.assertIn("outside", str(ctx.exception))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "..", "escaped.txt")))

    def test_absolute_path_is_rejected(self):
        zip_path = os.path.join(self.dir, "absolute.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            info = zipfile.ZipInfo("C:/Windows/Temp/evil.txt")
            archive.writestr(info, b"pwned")
        target = os.path.join(self.dir, "bin")
        os.makedirs(target)
        # Depending on normalisation either rejected or kept inside the target
        try:
            binaries.safe_extract(zip_path, target)
        except binaries.DownloadError:
            return
        for root, _dirs, files in os.walk(target):
            for name in files:
                self.assertTrue(os.path.realpath(os.path.join(root, name))
                                .startswith(os.path.realpath(target)))


if __name__ == "__main__":
    unittest.main()
