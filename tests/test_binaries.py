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
            with patch.object(binaries, "model_path", lambda _name: target), \
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
