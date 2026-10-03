"""Tests for the ElevenLabs backend (audit findings M8, H8, K2) and for how its
request behaves: cancelling, retrying, the FLAC upload and the timeouts."""

import http.server
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_transcriber.transcribe import elevenlabs
from audio_transcriber.transcribe.base import TranscriptionError

CAPTURED = {}
BEHAVIOUR = {"mode": "ok"}
STALL = threading.Event()           # releases a "stall" request

WORDS_RESPONSE = {
    "text": "Hello world. How are you?",
    "words": [
        {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.4, "end": 0.45, "type": "spacing"},
        {"text": "world.", "start": 0.45, "end": 0.9, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.9, "end": 1.0, "type": "spacing"},
        {"text": "How", "start": 1.0, "end": 1.2, "type": "word", "speaker_id": "speaker_1"},
        {"text": " ", "start": 1.2, "end": 1.25, "type": "spacing"},
        {"text": "are", "start": 1.25, "end": 1.5, "type": "word", "speaker_id": "speaker_1"},
        {"text": " ", "start": 1.5, "end": 1.55, "type": "spacing"},
        {"text": "you?", "start": 1.55, "end": 2.1, "type": "word", "speaker_id": "speaker_1"},
    ],
}


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _answer(self, status, payload, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        CAPTURED["content_length"] = length
        CAPTURED["actual_length"] = len(body)
        CAPTURED["content_type"] = self.headers.get("Content-Type", "")
        CAPTURED["api_key"] = self.headers.get("xi-api-key", "")
        CAPTURED["body"] = body
        CAPTURED["requests"] = CAPTURED.get("requests", 0) + 1
        mode = BEHAVIOUR["mode"]

        if mode == "stall":                    # the server is "transcribing"
            STALL.wait(30)
            return
        if mode == "hang_up":                  # takes the upload, never answers
            self.close_connection = True
            return
        if mode == "flaky" and CAPTURED["requests"] <= BEHAVIOUR["failures"]:
            self._answer(BEHAVIOUR["status"], b'{"detail": "try again later"}')
            return
        if mode == "always_503":
            self._answer(503, b'{"detail": "service unavailable"}')
            return
        if mode == "model_error" and b"scribe_v2" in body:
            payload = json.dumps({"detail": {"message": "model_id not found"}}).encode()
            self._answer(422, payload)
            return
        if mode == "unauthorized":
            self._answer(401, b'{"detail":"invalid api key"}')
            return

        self._answer(200, json.dumps(WORDS_RESPONSE).encode())


class ElevenLabsTestCase(unittest.TestCase):
    def setUp(self):
        CAPTURED.clear()
        BEHAVIOUR.clear()
        BEHAVIOUR["mode"] = "ok"
        STALL.clear()
        self.dir = tempfile.mkdtemp()
        self.wav = os.path.join(self.dir, "recording.wav")
        sf.write(self.wav, np.zeros(16000 * 3, dtype=np.float32), 16000,
                 subtype="PCM_16")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        # A short poll interval: shutdown() waits for one, which was half a
        # second per test.
        threading.Thread(target=self.httpd.serve_forever, args=(0.01,),
                         daemon=True).start()
        self._original_url = elevenlabs.API_URL
        elevenlabs.API_URL = f"http://127.0.0.1:{self.httpd.server_port}/v1/speech-to-text"
        # Retrying is the point of some tests - without the real pauses.
        self._delays = patch.object(elevenlabs, "RETRY_DELAYS", (0.0, 0.0), create=True)
        self._delays.start()

    def tearDown(self):
        self._delays.stop()
        STALL.set()
        elevenlabs.API_URL = self._original_url
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def backend(self, **options):
        return elevenlabs.ElevenLabsBackend(api_key=options.pop("api_key", "sk_test"),
                                            **options)


class TestChainedBody(unittest.TestCase):
    def test_streams_prefix_file_suffix_exactly(self):
        """Regression M8: the previous version assembled the whole body as a
        bytearray in RAM (roughly 700 MB peak for an hour of audio)."""
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "data.bin")
            payload = bytes(range(256)) * 400
            with open(path, "wb") as handle:
                handle.write(payload)

            handle = open(path, "rb")
            body = elevenlabs._ChainedBody([b"PREFIX", handle, b"SUFFIX"])
            chunks = []
            while True:
                chunk = body.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
            self.assertEqual(b"".join(chunks), b"PREFIX" + payload + b"SUFFIX")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_small_reads(self):
        body = elevenlabs._ChainedBody([b"abc", b"de"])
        self.assertEqual(body.read(2), b"ab")
        self.assertEqual(body.read(2), b"c")
        self.assertEqual(body.read(10), b"de")
        self.assertEqual(body.read(10), b"")

    def test_the_hooks_see_the_upload_and_its_end_exactly_once(self):
        seen, ended = [], []
        body = elevenlabs._ChainedBody([b"abc", b"de"], on_progress=seen.append,
                                       on_end=lambda: ended.append(True))
        for _ in range(5):
            body.read(10)
        self.assertEqual(seen, [3, 5])
        self.assertEqual(ended, [True])


class TestRequest(ElevenLabsTestCase):
    def test_multipart_is_well_formed(self):
        self.backend(model_id="scribe_v2").transcribe(self.wav, language="de")

        self.assertEqual(CAPTURED["api_key"], "sk_test")
        self.assertEqual(CAPTURED["content_length"], CAPTURED["actual_length"])
        self.assertIn("multipart/form-data; boundary=", CAPTURED["content_type"])

        body = CAPTURED["body"]
        self.assertIn(b'name="model_id"', body)
        self.assertIn(b"scribe_v2", body)
        self.assertIn(b'name="diarize"', body)
        self.assertIn(b'name="language_code"', body)
        self.assertIn(b"\r\nde\r\n", body)
        self.assertIn(b'filename="recording.flac"', body)
        self.assertTrue(body.rstrip().endswith(b"--"))

    def test_boundary_is_random(self):
        backend = self.backend()
        backend.transcribe(self.wav)
        first = CAPTURED["content_type"]
        backend.transcribe(self.wav)
        self.assertNotEqual(first, CAPTURED["content_type"])

    def test_language_auto_is_omitted(self):
        self.backend().transcribe(self.wav, language="auto")
        self.assertNotIn(b'name="language_code"', CAPTURED["body"])


class TestUploadFormat(ElevenLabsTestCase):
    """FLAC is lossless and documented as supported; an hour of 16 kHz speech is
    115 MB as WAV."""

    def test_the_upload_is_flac_and_smaller_than_the_wav(self):
        wav_size = os.path.getsize(self.wav)
        self.backend().transcribe(self.wav)

        body = CAPTURED["body"]
        self.assertIn(b'filename="recording.flac"', body)
        self.assertIn(b"Content-Type: audio/flac", body)
        self.assertIn(b"fLaC", body)                    # the FLAC stream marker
        self.assertLess(CAPTURED["content_length"], wav_size)
        self.assertEqual(os.listdir(self.dir), ["recording.wav"],
                         "the temporary FLAC must be removed again")

    def test_the_conversion_is_lossless(self):
        samples = (np.random.default_rng(5).normal(0, 0.1, 32000) * 32767).astype(np.int16)
        wav = os.path.join(self.dir, "noise.wav")
        flac = os.path.join(self.dir, "noise.flac")
        sf.write(wav, samples, 16000, subtype="PCM_16")

        elevenlabs._to_flac(wav, flac)

        again, rate = sf.read(flac, dtype="int16")
        self.assertEqual(rate, 16000)
        np.testing.assert_array_equal(again, samples)

    def test_a_failed_conversion_falls_back_to_the_wav(self):
        with patch.object(elevenlabs, "_to_flac", side_effect=RuntimeError("no flac")):
            self.backend().transcribe(self.wav)
        self.assertIn(b'filename="recording.wav"', CAPTURED["body"])
        self.assertIn(b"Content-Type: audio/wav", CAPTURED["body"])
        self.assertEqual(os.listdir(self.dir), ["recording.wav"],
                         "no half-written FLAC may stay behind")

    def test_upload_progress_runs_up_to_one_hundred_percent(self):
        seen = []
        self.backend().transcribe(self.wav, progress=seen.append)
        self.assertTrue(seen)
        self.assertIn("100.0 %", seen[-1])
        self.assertIn("recording.flac", seen[-1])


class TestResponseParsing(ElevenLabsTestCase):
    def test_words_are_joined_without_double_spaces(self):
        """Regression M8: ' '.join() across all tokens produced double spaces
        and detached punctuation."""
        segments = self.backend().transcribe(self.wav)
        texts = [segment.text for segment in segments]
        self.assertEqual(texts, ["Hello world.", "How are you?"])
        for text in texts:
            self.assertNotIn("  ", text)
            self.assertNotIn(" .", text)
            self.assertNotIn(" ?", text)

    def test_timestamps_and_speaker_hints(self):
        segments = self.backend().transcribe(self.wav)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 0.9)
        self.assertAlmostEqual(segments[1].start, 1.0)
        self.assertEqual(segments[0].speaker_hint, "speaker_0")
        self.assertEqual(segments[1].speaker_hint, "speaker_1")


class TestErrorHandling(ElevenLabsTestCase):
    def test_missing_key_is_reported_clearly(self):
        with self.assertRaises(TranscriptionError) as ctx:
            self.backend(api_key="").transcribe(self.wav)
        self.assertIn("API key", str(ctx.exception))

    def test_unauthorized_gives_actionable_message(self):
        """Regression K2: in the previous version every error message vanished
        into a NameError and the user saw nothing at all."""
        BEHAVIOUR["mode"] = "unauthorized"
        with self.assertRaises(TranscriptionError) as ctx:
            self.backend(api_key="sk_wrong").transcribe(self.wav)
        message = str(ctx.exception)
        self.assertIn("401", message)
        self.assertIn("rejected", message)

    def test_model_fallback_to_scribe_v1(self):
        """If the API rejects scribe_v2, scribe_v1 is tried automatically."""
        BEHAVIOUR["mode"] = "model_error"
        backend = self.backend(model_id="scribe_v2")
        segments = backend.transcribe(self.wav)
        self.assertTrue(segments)
        self.assertEqual(backend.model_id, "scribe_v1")
        self.assertIn(b"scribe_v1", CAPTURED["body"])

    def test_missing_file(self):
        with self.assertRaises(TranscriptionError):
            self.backend().transcribe(os.path.join(self.dir, "does-not-exist.wav"))


class TestRetries(ElevenLabsTestCase):
    """A transient failure should not cost the user the whole upload again and
    again - and a lost answer must NOT be retried, because the file may have
    been processed (and billed) already."""

    def test_rate_limits_are_retried(self):
        BEHAVIOUR.update(mode="flaky", failures=2, status=429)
        logs = []
        segments = self.backend().transcribe(self.wav, log=logs.append)

        self.assertTrue(segments)
        self.assertEqual(CAPTURED["requests"], 3)
        self.assertIn("Trying again", " ".join(logs))

    def test_server_errors_are_retried_then_reported(self):
        BEHAVIOUR["mode"] = "always_503"
        with self.assertRaises(TranscriptionError) as ctx:
            self.backend().transcribe(self.wav)
        self.assertIn("503", str(ctx.exception))
        self.assertEqual(CAPTURED["requests"], 3)

    def test_a_rejected_key_is_not_retried(self):
        BEHAVIOUR["mode"] = "unauthorized"
        with self.assertRaises(TranscriptionError):
            self.backend().transcribe(self.wav)
        self.assertEqual(CAPTURED["requests"], 1)

    def test_an_unreachable_server_is_retried_then_reported(self):
        # Refused at once, whatever the platform - a real closed port takes
        # Windows two seconds per attempt.
        logs = []
        with patch("http.client.HTTPConnection.connect",
                   side_effect=ConnectionRefusedError(10061, "refused")):
            with self.assertRaises(TranscriptionError) as ctx:
                self.backend().transcribe(self.wav, log=logs.append)
        self.assertIn("Could not reach ElevenLabs", str(ctx.exception))
        self.assertEqual(" ".join(logs).count("Trying again"), 2)

    def test_a_lost_answer_after_the_upload_is_not_retried(self):
        BEHAVIOUR["mode"] = "hang_up"
        with self.assertRaises(TranscriptionError) as ctx:
            self.backend().transcribe(self.wav)
        message = str(ctx.exception)
        self.assertIn("did not answer after the upload", message)
        self.assertIn("billed", message)
        self.assertEqual(CAPTURED["requests"], 1)

    def test_the_wait_for_the_answer_scales_with_the_audio(self):
        long_file = SimpleNamespace(duration=3 * 3600.0)
        short_file = SimpleNamespace(duration=60.0)
        with patch.object(elevenlabs.sf, "info", return_value=long_file):
            self.assertEqual(elevenlabs._response_timeout("x"), 5400.0)
        with patch.object(elevenlabs.sf, "info", return_value=short_file):
            self.assertEqual(elevenlabs._response_timeout("x"),
                             elevenlabs.RESPONSE_TIMEOUT)


class TestCancel(ElevenLabsTestCase):
    """cancel() used to take effect only once the answer had arrived - after the
    upload and the minutes the server needs, which is when it matters."""

    def _run_in_thread(self, backend):
        outcome = {}

        def work():
            started = time.monotonic()
            try:
                backend.transcribe(self.wav)
            except TranscriptionError as exc:
                outcome["error"] = str(exc)
            outcome["seconds"] = time.monotonic() - started

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        return thread, outcome

    @staticmethod
    def _wait_for(predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_cancel_ends_the_wait_for_the_server_at_once(self):
        BEHAVIOUR["mode"] = "stall"
        backend = self.backend()
        thread, outcome = self._run_in_thread(backend)
        self.assertTrue(self._wait_for(lambda: CAPTURED.get("requests", 0) >= 1),
                        "the upload never completed")

        backend.cancel()
        thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive(), "cancel() did not end the request")
        self.assertIn("cancelled", outcome["error"])
        self.assertEqual(CAPTURED["requests"], 1, "a cancelled request is not retried")

    def test_cancel_before_the_request_sends_nothing(self):
        backend = self.backend()
        backend.cancel()
        with self.assertRaises(TranscriptionError) as ctx:
            backend.transcribe(self.wav)
        self.assertIn("cancelled", str(ctx.exception))
        self.assertEqual(CAPTURED.get("requests", 0), 0)

    def test_cancel_during_the_pause_between_attempts(self):
        BEHAVIOUR["mode"] = "always_503"
        backend = self.backend()
        with patch.object(elevenlabs, "RETRY_DELAYS", (30.0, 30.0)):
            thread, outcome = self._run_in_thread(backend)
            self.assertTrue(self._wait_for(lambda: CAPTURED.get("requests", 0) >= 1))
            time.sleep(0.2)                      # now it is waiting to retry
            backend.cancel()
            thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive(), "the pause was not interrupted")
        self.assertIn("cancelled", outcome["error"])
        self.assertEqual(CAPTURED["requests"], 1)


if __name__ == "__main__":
    unittest.main()
