"""What the closing pass of a recording and the transcription of an uploaded
file have in common.

Both are workers in a thread of their own, and both owe the window the same
things: a cancel that reaches the backend that is busy, scratch files that are
gone when the run ends however it ends, and a last line of defence so that
anything unexpected becomes an error message instead of a locked interface.
They used to be two copies of that code; these tests hold for both.
"""

import os
import queue
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from audio_transcriber import config, pipeline
from audio_transcriber.audio import capture
from audio_transcriber.events import Failed, Finished
from audio_transcriber.transcribe.base import Segment, TranscriptionError

RATE = 16000
WAIT = 10.0


class Bridge:
    def __init__(self):
        self.events = queue.Queue()

    def post(self, event):
        self.events.put(event)

    def post_exception(self, prefix, exc):
        self.events.put(Failed(message=f"{prefix}: {exc}"))

    def all(self):
        items = []
        while not self.events.empty():
            items.append(self.events.get())
        return items

    def failure(self):
        return next((event for event in self.all() if isinstance(event, Failed)), None)


class Backend:
    """Records what it is asked; can fail, or block until it is cancelled."""

    def __init__(self, block=False, error=None):
        self.block = block
        self.error = error
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.calls = []
        self.scratch_seen = []        # (path, existed while being recognised)

    def transcribe(self, path, language="de", log=None, track="", progress=None):
        self.calls.append(track)
        self.scratch_seen.append((path, os.path.exists(path)))
        self.started.set()
        if self.error is not None:
            raise self.error
        if self.block:
            if not self.cancelled.wait(WAIT):
                raise AssertionError("the backend was never cancelled")
            raise TranscriptionError("Transcription cancelled.")
        return [Segment(start=0.5, end=1.5, text="spoken", track=track)]

    def cancel(self):
        self.cancelled.set()


class WorkerContract:
    """Shared cases. A subclass says how to build and start its worker."""

    title = ""            # heading of the error for something unexpected
    thread_name = ""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "out")
        self.tmp = os.path.join(self.dir, "tmp")
        os.makedirs(self.tmp)
        patcher = patch.object(pipeline, "TMP_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.bridge = Bridge()
        self.settings = config.Settings(output_dir=self.out, language="en")

    def _scratch_left(self):
        return [name for name in os.listdir(self.tmp) if name.endswith(".asr.wav")]

    def _run(self, backend):
        thread = self.start(self.build(backend))
        thread.join(WAIT)
        self.assertFalse(thread.is_alive(), "the run did not end")
        return thread

    # ------------------------------------------------------------------
    def test_a_cancel_reaches_the_backend_that_is_busy(self):
        backend = Backend(block=True)
        worker = self.build(backend)
        thread = self.start(worker)
        self.assertTrue(backend.started.wait(WAIT), "the backend was never reached")

        worker.cancel()

        thread.join(WAIT)
        self.assertFalse(thread.is_alive())
        self.assertTrue(backend.cancelled.is_set())
        self.assertIsNotNone(self.bridge.failure())

    def test_a_run_cancelled_before_it_got_going_recognises_nothing(self):
        backend = Backend()
        worker = self.build(backend)
        worker.cancel()
        self.start(worker).join(WAIT)

        self.assertEqual(backend.calls, [])
        failure = self.bridge.failure()
        self.assertIsNotNone(failure)
        self.assertEqual(failure.message, "Processing cancelled.")

    def test_a_recognition_error_is_reported_as_it_is(self):
        self._run(Backend(error=TranscriptionError("The model is missing.")))
        self.assertEqual(self.bridge.failure().message, "The model is missing.")

    def test_something_unexpected_is_reported_under_the_workers_own_heading(self):
        self._run(Backend(error=RuntimeError("disk on fire")))
        self.assertEqual(self.bridge.failure().message,
                         f"{self.title}: disk on fire")

    def test_scratch_files_exist_during_the_run_and_are_gone_after_success(self):
        backend = Backend()
        self._run(backend)
        self.assertTrue(backend.scratch_seen)
        self.assertTrue(all(existed for _path, existed in backend.scratch_seen))
        self.assertEqual(self._scratch_left(), [])
        self.assertTrue(any(isinstance(event, Finished) for event in self.bridge.all()))

    def test_scratch_files_are_gone_after_a_failure(self):
        backend = Backend(error=RuntimeError("boom"))
        self._run(backend)
        self.assertTrue(backend.scratch_seen, "nothing was prepared, nothing to clean")
        self.assertEqual(self._scratch_left(), [])

    def test_scratch_files_are_gone_after_a_cancel(self):
        backend = Backend(block=True)
        worker = self.build(backend)
        thread = self.start(worker)
        self.assertTrue(backend.started.wait(WAIT))
        worker.cancel()
        thread.join(WAIT)
        self.assertEqual(self._scratch_left(), [])

    def test_the_thread_is_named_for_what_it_does(self):
        thread = self._run(Backend())
        self.assertEqual(thread.name, self.thread_name)

    def test_it_is_a_daemon_so_it_never_keeps_the_program_alive(self):
        self.assertTrue(self._run(Backend()).daemon)

    def test_the_worker_uses_the_settings_it_was_given(self):
        seen = []

        def factory(settings):
            seen.append(settings)
            return Backend()

        thread = self.start(self.build(Backend(), factory=factory))
        thread.join(WAIT)
        self.assertTrue(seen)
        self.assertTrue(all(settings is self.settings for settings in seen))


class TestFinalizerAsWorker(WorkerContract, unittest.TestCase):
    title = "Unexpected error during processing"
    thread_name = "finalize"

    def _recording(self):
        rng = np.random.default_rng(2)
        result = capture.RecordingResult()
        for kind in ("mic", "sys"):
            path = os.path.join(self.tmp, f"take.{kind}.raw.wav")
            sf.write(path, rng.normal(0, 0.2, RATE * 3).astype(np.float32), RATE,
                     subtype="PCM_16")
            setattr(result, kind, capture.TrackResult(path=path, rate=RATE,
                                                      frames=RATE * 3))
        return result

    def build(self, backend, factory=None, live=None):
        return pipeline.Finalizer(self.bridge, self.settings,
                                  backend_factory=factory or (lambda s: backend),
                                  live=live)

    def start(self, worker):
        return worker.run_async(self._recording(), "take")

    def test_a_cancel_also_stops_the_live_transcription_of_that_recording(self):
        live = MagicMock()
        live.finish.return_value = ({}, {})
        worker = self.build(Backend(), live=live)
        worker.cancel()
        live.cancel.assert_called_once()

    def test_a_cancel_between_the_two_tracks_stops_before_the_second(self):
        backend = Backend()
        worker = self.build(backend)
        original = backend.transcribe

        def cancel_after_the_first(*args, **kwargs):
            segments = original(*args, **kwargs)
            worker.cancel()
            return segments

        backend.transcribe = cancel_after_the_first
        self.start(worker).join(WAIT)

        self.assertEqual(len(backend.calls), 1, "the second track must not start")
        self.assertEqual(self.bridge.failure().message, "Processing cancelled.")


class TestFileFinalizerAsWorker(WorkerContract, unittest.TestCase):
    title = "Unexpected error during file processing"
    thread_name = "file-finalize"

    def _audio_file(self):
        path = os.path.join(self.dir, "talk.wav")
        t = np.arange(RATE * 2) / RATE
        sf.write(path, (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), RATE,
                 subtype="PCM_16")
        return path

    def build(self, backend, factory=None):
        return pipeline.FileFinalizer(self.bridge, self.settings,
                                      backend_factory=factory or (lambda s: backend))

    def start(self, worker):
        return worker.run_async(self._audio_file(), base_name="talk")


if __name__ == "__main__":
    unittest.main()
