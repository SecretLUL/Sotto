"""Fetching the model of the closing pass while the recording runs.

A model that is not on the computer yet used to be downloaded after the
meeting - with large-v3 that is 3 GB of waiting at the moment the transcript is
wanted. Recording takes long enough to do it in the meantime.
"""

import queue
import threading
import unittest

from audio_transcriber import config, pipeline
from audio_transcriber.events import Failed, Log, Progress
from audio_transcriber.transcribe import binaries

WAIT = 5.0


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


class Backend:
    def __init__(self, prepare=None):
        self._prepare = prepare
        self.prepared = threading.Event()
        self.cancelled = threading.Event()

    def prepare(self, progress=None, log=None):
        self.prepared.set()
        if self._prepare is not None:
            self._prepare(self, progress, log)

    def cancel(self):
        self.cancelled.set()


class TestModelPrefetch(unittest.TestCase):
    def setUp(self):
        self.bridge = Bridge()
        self.settings = config.Settings(model="large-v3")

    def prefetch(self, backend, settings=None, built=None):
        def factory(given):
            if built is not None:
                built.append(given)
            return backend
        return pipeline.ModelPrefetch(self.bridge, settings or self.settings,
                                      backend_factory=factory)

    def test_it_prepares_the_backend_of_the_closing_pass_in_the_background(self):
        backend, built = Backend(), []
        thread = self.prefetch(backend, built=built).start()
        thread.join(WAIT)
        self.assertTrue(backend.prepared.is_set())
        self.assertEqual(built, [self.settings])
        self.assertTrue(thread.daemon)

    def test_it_does_not_wait_for_the_download(self):
        release = threading.Event()
        backend = Backend(prepare=lambda *_: release.wait(WAIT))
        prefetch = self.prefetch(backend)
        thread = prefetch.start()                      # returns at once
        self.assertTrue(backend.prepared.wait(WAIT))
        self.assertTrue(thread.is_alive(), "the download is still running")
        release.set()
        thread.join(WAIT)
        self.assertFalse(thread.is_alive())

    def test_progress_and_log_go_to_the_window(self):
        def prepare(_backend, progress, log):
            log("Downloading model 'large-v3'…\n")
            progress("Whisper model 'large-v3':  10.0 %")
        self.prefetch(Backend(prepare=prepare)).start().join(WAIT)
        events = self.bridge.all()
        self.assertEqual([type(event) for event in events], [Log, Progress])
        self.assertIn("large-v3", events[0].text)
        self.assertIn("10.0 %", events[1].text)

    def test_the_cloud_has_nothing_to_fetch(self):
        built = []
        backend = Backend()
        prefetch = self.prefetch(backend, config.Settings(model=config.CLOUD_MODEL),
                                 built=built)
        self.assertIsNone(prefetch.start())
        self.assertEqual(built, [])
        self.assertFalse(backend.prepared.is_set())

    def test_a_failure_is_a_note_never_an_error(self):
        """The closing pass tries again and reports a real failure where it
        matters; a failed download in the background must not look like one."""
        def prepare(*_):
            raise binaries.DownloadError("Whisper model: could not connect (no network).")
        self.prefetch(Backend(prepare=prepare)).start().join(WAIT)

        events = self.bridge.all()
        self.assertFalse(any(isinstance(event, Failed) for event in events))
        notes = [event.text for event in events if isinstance(event, Log)]
        self.assertEqual(len(notes), 1)
        self.assertIn("no network", notes[0])
        self.assertIn("again when the recording ends", notes[0])

    def test_cancelling_stops_the_download(self):
        started = threading.Event()

        def prepare(backend, *_):
            started.set()
            backend.cancelled.wait(WAIT)
            raise binaries.DownloadError("download cancelled.")

        backend = Backend(prepare=prepare)
        prefetch = self.prefetch(backend)
        thread = prefetch.start()
        self.assertTrue(started.wait(WAIT))

        prefetch.cancel()
        thread.join(WAIT)

        self.assertTrue(backend.cancelled.is_set())
        self.assertFalse(thread.is_alive())
        self.assertEqual([event for event in self.bridge.all() if isinstance(event, Log)],
                         [], "a cancel is not worth a line")

    def test_a_cancel_before_the_thread_got_going_prevents_the_download(self):
        backend = Backend()
        prefetch = self.prefetch(backend)
        prefetch.cancel()
        thread = prefetch.start()
        thread.join(WAIT)
        self.assertFalse(backend.prepared.is_set())

    def test_cancelling_something_that_never_started_is_harmless(self):
        self.prefetch(Backend()).cancel()

    def test_a_backend_without_a_prepare_step_is_left_alone(self):
        class Plain:
            def cancel(self):
                pass

        self.prefetch(Plain()).start().join(WAIT)
        self.assertEqual(self.bridge.all(), [])

    def test_it_asks_for_the_model_of_the_closing_pass_not_the_live_one(self):
        """build_backend() with live=False: large-v3, not the small live model."""
        seen = []

        class Recording(Backend):
            pass

        prefetch = pipeline.ModelPrefetch(self.bridge, self.settings)
        original = pipeline.build_backend
        try:
            pipeline.build_backend = lambda settings, **kw: (seen.append((settings, kw)),
                                                              Recording())[1]
            prefetch.start().join(WAIT)
        finally:
            pipeline.build_backend = original
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1].get("live", False), False)
        self.assertEqual(seen[0][0].model_name(), "large-v3")


if __name__ == "__main__":
    unittest.main()
