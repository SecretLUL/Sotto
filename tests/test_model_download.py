"""Downloading a model on request - the Download button on the Settings tab.

A model used to come only with the first recording that needed it, and nothing
showed whether it was there.
"""

import queue
import threading
import unittest

from audio_transcriber import pipeline
from audio_transcriber.events import ModelFetch, ModelFetchEnded
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


class TestModelDownload(unittest.TestCase):
    def setUp(self):
        self.bridge = Bridge()
        self.fetched = []

    def download(self, fetch_model=None, fetch_engine=None, engine_needed=False):
        def model(name, log=None, cancelled=None, on_bytes=None):
            self.fetched.append(name)
            log(f"Downloading model '{name}'…\n")
            for done in (0, 50, 100):
                on_bytes(done, 100)

        def engine(log=None, cancelled=None, on_bytes=None):
            self.fetched.append("whisper.cpp")
            log("Downloading whisper.cpp…\n")
            on_bytes(10, 10)
            log("Extracting archive…\n")

        return pipeline.ModelDownload(self.bridge, "small",
                                      fetch_engine=fetch_engine or engine,
                                      fetch_model=fetch_model or model,
                                      engine_needed=lambda: engine_needed)

    def run_to_end(self, download):
        thread = download.start()
        thread.join(WAIT)
        self.assertFalse(thread.is_alive())
        self.assertTrue(thread.daemon)
        return self.bridge.all()

    def test_it_fetches_the_model_and_says_how_far_it_got(self):
        events = self.run_to_end(self.download())
        self.assertEqual(self.fetched, ["small"])
        progress = [event for event in events if isinstance(event, ModelFetch)]
        self.assertEqual(progress[0].step, "Downloading model 'small'…")
        last = progress[-1]
        self.assertEqual((last.done, last.total), (100, 100),
                         "the last block is always reported, however quick")
        self.assertEqual(events[-1], ModelFetchEnded("small"))

    def test_whisper_cpp_comes_first_where_it_is_missing(self):
        events = self.run_to_end(self.download(engine_needed=True))
        self.assertEqual(self.fetched, ["whisper.cpp", "small"])
        steps = [event.step for event in events if isinstance(event, ModelFetch)]
        self.assertIn("Extracting archive…", steps)

    def test_reports_are_thinned_out(self):
        """A 3 GB model is 3000 blocks; the window needs ten reports a second."""
        def model(name, log=None, cancelled=None, on_bytes=None):
            for done in range(1, 3001):
                on_bytes(done, 3000)

        events = self.run_to_end(self.download(fetch_model=model))
        progress = [event for event in events if isinstance(event, ModelFetch)]
        self.assertLess(len(progress), 100)
        self.assertEqual(progress[-1].done, 3000)

    def test_the_known_size_stands_in_when_the_server_does_not_say(self):
        def model(name, log=None, cancelled=None, on_bytes=None):
            on_bytes(10, 0)

        events = self.run_to_end(self.download(fetch_model=model))
        progress = [event for event in events if isinstance(event, ModelFetch)]
        self.assertEqual(progress[-1].total, binaries.MODEL_SIZE_BYTES["small"])

    def test_a_failure_ends_with_the_reason(self):
        def model(*_args, **_kwargs):
            raise binaries.DownloadError("Whisper model: could not connect.")

        events = self.run_to_end(self.download(fetch_model=model))
        self.assertEqual(events[-1], ModelFetchEnded(
            "small", error="Whisper model: could not connect."))

    def test_cancelling_ends_the_download_and_is_no_failure(self):
        started = threading.Event()

        def model(name, log=None, cancelled=None, on_bytes=None):
            started.set()
            for _ in range(500):
                if cancelled():
                    raise binaries.DownloadError("download cancelled.")
                threading.Event().wait(0.01)

        download = self.download(fetch_model=model)
        thread = download.start()
        self.assertTrue(started.wait(WAIT))
        download.cancel()
        thread.join(WAIT)
        self.assertFalse(thread.is_alive())
        self.assertTrue(download.cancelled)
        self.assertEqual(self.bridge.all()[-1],
                         ModelFetchEnded("small", cancelled=True))


if __name__ == "__main__":
    unittest.main()
