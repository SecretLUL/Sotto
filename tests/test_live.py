"""Live transcription while recording, and the live preview it replaced.

Two things are covered here:
  * The preview regression: `len(mic or [])` raises ValueError on a numpy
    array with more than one element, and the preview loop caught every
    exception - so the feature silently did nothing at all.
  * LiveTranscriber: chunking, the timeline it hands to the closing pass, and
    that the finalizer transcribes only what is left instead of everything.
"""

import os
import shutil
import tempfile
import threading
import time
import unittest
from collections import namedtuple
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_transcriber import pipeline
from audio_transcriber.audio import dsp
from audio_transcriber.config import Settings
from audio_transcriber.events import Failed, Finished, LivePreview
from audio_transcriber.transcribe.base import Segment

RATE = 48000


def tone(seconds, rate=RATE, freq=220.0, level=0.3):
    """An uninterrupted tone, loud enough to pass the silence checks."""
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False, dtype=np.float32)
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speech(seconds, rate=RATE, freq=220.0, level=0.3):
    """A tone with a short pause at the end of every second.

    The pauses give the chunk splitter something to aim at - the way real
    speech does.
    """
    tone_ = tone(seconds, rate, freq, level)
    for second in range(int(seconds)):
        gap = int((second + 0.9) * rate)
        tone_[gap:gap + int(0.1 * rate)] = 0.0
    return tone_


def wait_for(predicate, timeout=20.0):
    """Give the worker thread time to catch up, the way a recording does."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class FakeBridge:
    def __init__(self):
        self.events = []

    def post(self, event):
        self.events.append(event)

    def post_exception(self, message, exc):
        self.events.append(Failed(message=f"{message}: {exc}"))

    def of_type(self, event_type):
        return [event for event in self.events if isinstance(event, event_type)]


class FakeEngine:
    """Only the tap side of AudioEngine, driven by the test."""

    def __init__(self):
        self.tap = None
        self.written = {"mic": 0, "sys": 0}

    def set_tap(self, callback):
        self.tap = callback

    def feed(self, kind, samples, rate=RATE, position=None):
        at = self.written[kind] if position is None else position
        self.written[kind] = at + len(samples)
        if self.tap is not None:
            self.tap(kind, samples, rate, at)


Call = namedtuple("Call", "path track duration")


class FakeBackend:
    """Returns one segment per call, timed inside the chunk.

    The duration is read while the file still exists - both the live chunks
    and the closing pass delete their work files afterwards.
    """

    def __init__(self, text="chunk"):
        self.text = text
        self.calls = []
        self.cancelled = False

    def prepare(self, progress=None, log=None):
        return "exe", "model", None

    def transcribe(self, wav_path, language="de", log=None, track="", progress=None):
        self.calls.append(Call(wav_path, track, sf.info(wav_path).duration))
        return [Segment(start=1.0, end=2.0,
                        text=f"{self.text} {len(self.calls)}", track=track)]

    def cancel(self):
        self.cancelled = True


# ----------------------------------------------------------------------
class TestLiveMixdown(unittest.TestCase):
    """The preview regression, stated as a unit test."""

    def test_arrays_longer_than_one_sample_do_not_raise(self):
        mic = speech(3.0, rate=dsp.TARGET_RATE)
        system = speech(3.0, rate=dsp.TARGET_RATE, freq=440.0)
        result = pipeline.live_mixdown(mic, system)
        self.assertIsNotNone(result)
        mono, mic_out, sys_out = result
        self.assertEqual(len(mono), len(mic))
        self.assertEqual(len(mic_out), len(sys_out))

    def test_one_missing_track_is_padded(self):
        mic = speech(3.0, rate=dsp.TARGET_RATE)
        mono, mic_out, sys_out = pipeline.live_mixdown(mic, None)
        self.assertEqual(len(sys_out), len(mic_out))
        self.assertEqual(float(np.max(np.abs(sys_out))), 0.0)
        self.assertGreater(float(np.max(np.abs(mono))), 0.0)

    def test_too_little_material_is_skipped(self):
        short = speech(0.5, rate=dsp.TARGET_RATE)
        self.assertIsNone(pipeline.live_mixdown(short, short))
        self.assertIsNone(pipeline.live_mixdown(None, None))

    def test_tracks_of_different_length_align(self):
        mic = speech(3.0, rate=dsp.TARGET_RATE)
        system = speech(1.0, rate=dsp.TARGET_RATE, freq=440.0)
        _mono, mic_out, sys_out = pipeline.live_mixdown(mic, system)
        self.assertEqual(len(mic_out), len(sys_out))


class TestQuietestSplit(unittest.TestCase):
    def test_cut_lands_in_the_pause(self):
        audio = np.concatenate([tone(2.0), np.zeros(int(0.4 * RATE), np.float32),
                                tone(1.0)])
        cut = dsp.quietest_split(audio, RATE, int(1.5 * RATE), int(2.8 * RATE))
        self.assertGreaterEqual(cut, int(2.0 * RATE))
        self.assertLessEqual(cut, int(2.4 * RATE))

    def test_range_too_short_falls_back_to_the_end(self):
        self.assertEqual(dsp.quietest_split(tone(1.0), RATE, 0, 10), 10)


# ----------------------------------------------------------------------
class TestLiveTranscriber(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patcher = patch.object(pipeline, "TMP_DIR", self.tmp)
        self.patcher.start()
        self.bridge = FakeBridge()
        self.engine = FakeEngine()
        self.backend = FakeBackend()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _live(self, name="meeting", **kwargs):
        options = dict(chunk_s=2.0, preview=True)
        options.update(kwargs)
        live = pipeline.LiveTranscriber(
            self.bridge, Settings(), self.engine,
            backend_factory=lambda s: self.backend, **options)
        live.start(name)
        return live

    def test_chunks_are_transcribed_while_recording(self):
        live = self._live()
        self.engine.feed("mic", speech(5.0))
        self.assertTrue(wait_for(lambda: len(self.backend.calls) >= 2))
        covered, segments = live.finish(timeout=20.0)

        self.assertGreater(covered["mic"], 3.0)
        self.assertLessEqual(covered["mic"], 5.0)
        self.assertEqual(len(segments["mic"]), len(self.backend.calls))

    def test_segment_times_continue_across_chunks(self):
        live = self._live()
        self.engine.feed("mic", speech(6.0))
        self.assertTrue(wait_for(lambda: len(self.backend.calls) >= 2))
        _covered, segments = live.finish(timeout=20.0)

        starts = [segment.start for segment in segments["mic"]]
        self.assertEqual(starts, sorted(starts))
        self.assertAlmostEqual(starts[0], 1.0, delta=0.5)     # first chunk
        self.assertGreater(starts[1], 2.5)                    # shifted along

    def test_silent_chunks_skip_the_recogniser(self):
        live = self._live()
        self.engine.feed("sys", np.zeros(int(5 * RATE), dtype=np.float32))
        self.assertTrue(wait_for(lambda: live.covered_s["sys"] > 3.0))
        covered, segments = live.finish(timeout=20.0)

        self.assertEqual(self.backend.calls, [])
        self.assertGreater(covered["sys"], 3.0)               # still covered
        self.assertEqual(segments["sys"], [])

    def test_chunk_files_ignore_the_recording_name_and_are_removed(self):
        """whisper-cli cannot open a file named in Turkish or Arabic."""
        live = self._live(name="toplantı_şirket_اجتم")
        self.engine.feed("mic", speech(5.0))
        self.assertTrue(wait_for(lambda: self.backend.calls))
        live.finish(timeout=20.0)

        for call in self.backend.calls:
            self.assertTrue(os.path.basename(call.path).isascii(), call.path)
        self.assertEqual(os.listdir(self.tmp), [], "chunk files were left behind")

    def test_the_tail_stays_in_the_buffer(self):
        """Anything shorter than a chunk is left to the closing pass."""
        live = self._live()
        self.engine.feed("mic", speech(3.0))
        self.assertTrue(wait_for(lambda: self.backend.calls))
        covered, _segments = live.finish(timeout=20.0)
        self.assertLess(covered["mic"], 3.0)
        self.assertGreater(covered["mic"], 0.0)

    def test_a_backlog_stops_live_transcription(self):
        live = self._live(max_pending_s=1.0)
        self.engine.feed("mic", speech(3.0))
        live.finish(timeout=20.0)

        self.assertIsNone(self.engine.tap)                    # detached
        logs = " ".join(event.text for event in self.bridge.events
                        if hasattr(event, "text"))
        self.assertIn("slower than real time", logs)

    def test_attaching_after_the_first_block_gives_up(self):
        """covered_s only means anything if the tap heard the file from 0.

        Attaching late would make the closing pass skip a stretch nobody ever
        transcribed, so live transcription bows out instead.
        """
        live = self._live()
        self.engine.feed("mic", speech(3.0), position=5 * RATE)
        live.finish(timeout=20.0)

        self.assertIsNone(self.engine.tap)
        self.assertEqual(live.covered_s["mic"], 0.0)
        self.assertEqual(self.backend.calls, [])
        logs = " ".join(event.text for event in self.bridge.events
                        if hasattr(event, "text"))
        self.assertIn("attached after the recording", logs)

    def test_preview_events_carry_both_speakers(self):
        live = self._live()
        self.engine.feed("mic", speech(3.0))
        self.engine.feed("sys", speech(3.0, freq=440.0))
        self.assertTrue(wait_for(lambda: len(self.backend.calls) >= 2))
        live.finish(timeout=20.0)

        previews = self.bridge.of_type(LivePreview)
        self.assertTrue(previews)
        text = previews[-1].text
        self.assertIn("[You]", text)
        self.assertIn("[Participant]", text)

    def test_preview_can_be_switched_off(self):
        live = self._live(preview=False)
        self.engine.feed("mic", speech(3.0))
        self.assertTrue(wait_for(lambda: self.backend.calls))
        live.finish(timeout=20.0)
        self.assertEqual(self.bridge.of_type(LivePreview), [])

    def test_nothing_is_recorded_after_the_seal(self):
        live = self._live()
        self.engine.feed("mic", speech(3.0))
        self.assertTrue(wait_for(lambda: self.backend.calls))
        covered, _segments = live.finish(timeout=20.0)

        # A late chunk must not move the point the tail was cut at.
        live._advance("mic", 30.0, [Segment(start=0.0, end=1.0, text="late")])
        self.assertEqual(live.covered_s["mic"], covered["mic"])
        self.assertNotIn("late", [segment.text for segment in live.segments["mic"]])


# ----------------------------------------------------------------------
class TestLivePreviewStop(unittest.TestCase):
    """stop() runs on the GUI thread, so it must not wait for a download.

    It used to join the thread for up to five seconds: a frozen window whenever
    the preview was still fetching its model.
    """

    def test_stop_returns_at_once_even_if_the_thread_cannot_be_interrupted(self):
        entered = threading.Event()
        released = threading.Event()

        class StuckBackend:
            def prepare(self, progress=None, log=None):
                entered.set()
                released.wait(timeout=3.0)          # a download that ignores cancel
                raise RuntimeError("download cancelled")

            def cancel(self):
                pass

        bridge = FakeBridge()
        folder = tempfile.mkdtemp()
        try:
            with patch.object(pipeline, "TMP_DIR", folder):
                with patch.object(pipeline, "build_backend",
                                  lambda *args, **kwargs: StuckBackend()):
                    preview = pipeline.LivePreview(bridge, Settings(), FakeEngine())
                    preview.start()
                    self.assertTrue(entered.wait(5.0))

                    started = time.monotonic()
                    preview.stop()
                    elapsed = time.monotonic() - started
                    released.set()
                    time.sleep(0.3)         # let the thread end on its own
        finally:
            shutil.rmtree(folder, ignore_errors=True)

        self.assertLess(elapsed, 0.5, "stop() waited for the thread")
        logs = " ".join(e.text for e in bridge.events if hasattr(e, "text"))
        self.assertNotIn("unavailable", logs, "a cancel is not worth a log line")

    def test_a_stop_before_the_backend_exists_prevents_the_download(self):
        """stop() had no backend to cancel yet - the thread then started a
        download nobody could stop."""
        prepared = []

        class Backend:
            def prepare(self, progress=None, log=None):
                prepared.append(True)

            def cancel(self):
                pass

        folder = tempfile.mkdtemp()
        try:
            with patch.object(pipeline, "TMP_DIR", folder):
                with patch.object(pipeline, "build_backend",
                                  lambda *args, **kwargs: Backend()):
                    preview = pipeline.LivePreview(FakeBridge(), Settings(),
                                                   FakeEngine())
                    preview._stop.set()              # stop() arrived first
                    preview._loop()
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        self.assertEqual(prepared, [])


class TailBackend(FakeBackend):
    """Puts its segment at the end of whatever file it is handed.

    The closing pass prepends a few seconds of run-up before the point live
    transcription reached, and drops whatever is recognised inside it. A
    backend that always answers at second one would only ever land in that
    discarded stretch.
    """

    def transcribe(self, wav_path, language="de", log=None, track="", progress=None):
        duration = sf.info(wav_path).duration
        self.calls.append(Call(wav_path, track, duration))
        return [Segment(start=max(0.0, duration - 1.0), end=duration,
                        text=f"{self.text} {len(self.calls)}", track=track)]


class FakeLive:
    """Stands in for a LiveTranscriber that already did part of the work."""

    def __init__(self, covered, segments):
        self.covered = covered
        self.segments = segments
        self.finished = False
        self.cancelled = False

    def finish(self, timeout=180.0):
        self.finished = True
        return dict(self.covered), {kind: list(items)
                                    for kind, items in self.segments.items()}

    def cancel(self):
        self.cancelled = True


class TestFinalizerWithLiveResults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.out, exist_ok=True)
        self.patcher = patch.object(pipeline, "TMP_DIR", self.tmp)
        self.patcher.start()
        self.bridge = FakeBridge()
        self.backend = TailBackend(text="tail")

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # Longer than one whisper window, so the tail rule can actually bind: the
    # closing pass never gets less than LIVE_TAIL_WINDOW_S of audio.
    SECONDS = 60.0
    COVERED = 50.0

    def _recording(self, seconds=SECONDS, offset=0.0):
        from audio_transcriber.audio.capture import RecordingResult, TrackResult
        path = os.path.join(self.tmp, "meeting.mic.raw.wav")
        sf.write(path, speech(seconds, rate=dsp.TARGET_RATE), dsp.TARGET_RATE)
        return RecordingResult(
            mic=TrackResult(path=path, rate=dsp.TARGET_RATE,
                            frames=int(seconds * dsp.TARGET_RATE),
                            start_offset_s=offset, device_name="mic"))

    def _run(self, live):
        finalizer = pipeline.Finalizer(self.bridge, Settings(output_dir=self.out),
                                       backend_factory=lambda s: self.backend,
                                       live=live)
        thread = finalizer.run_async(self._recording(), "meeting")
        thread.join(timeout=30.0)
        failed = self.bridge.of_type(Failed)
        self.assertEqual(failed, [], f"unexpected failure: {failed}")
        return self.bridge.of_type(Finished)[0]

    def _live(self, segments=()):
        return FakeLive({"mic": self.COVERED}, {"mic": list(segments)})

    def test_live_lines_end_up_in_the_saved_transcript(self):
        live = self._live([Segment(start=1.0, end=2.0, text="said live",
                                   track="mic")])
        finished = self._run(live)
        self.assertTrue(live.finished)
        self.assertIn("said live", finished.text)
        with open(finished.txt_path, encoding="utf-8") as handle:
            self.assertIn("said live", handle.read())

    def test_only_the_tail_is_sent_to_the_recogniser(self):
        self._run(self._live([Segment(start=1.0, end=2.0, text="said live",
                                      track="mic")]))
        self.assertEqual(len(self.backend.calls), 1)
        # 10 s of tail, stretched to a full whisper window - anything shorter
        # is segmented differently from the same audio in a longer file.
        self.assertAlmostEqual(self.backend.calls[0].duration,
                               pipeline.LIVE_TAIL_WINDOW_S, delta=0.2)

    def test_tail_segments_are_moved_onto_the_full_timeline(self):
        finished = self._run(self._live())
        # The tail file starts at 30 s and the backend answers at its end.
        self.assertIn("[00:59]", finished.text)

    def test_what_the_run_up_recognises_is_thrown_away(self):
        """Those seconds are already in the live segments - once is enough."""
        self.backend = FakeBackend(text="run-up")     # answers at second 1
        finished = self._run(self._live([Segment(start=1.0, end=2.0,
                                                 text="said live", track="mic")]))

        self.assertEqual(len(self.backend.calls), 1)  # the run-up did run
        self.assertNotIn("run-up", finished.text)     # but is not kept
        self.assertIn("said live", finished.text)

    def test_a_segment_across_the_cut_is_kept(self):
        """Regression: it took the end of the recording with it.

        whisper merges the closing sentences into one segment that begins just
        before the cut. Judged by its start it counted as run-up and was
        dropped - together with everything it said after the cut, which nothing
        else had transcribed.
        """
        class StraddlingBackend(FakeBackend):
            def transcribe(self, wav_path, language="de", log=None, track="",
                           progress=None):
                duration = sf.info(wav_path).duration
                self.calls.append(Call(wav_path, track, duration))
                # Begins one second before the cut, runs to the end.
                return [Segment(start=duration - 11.0, end=duration,
                                text="goodbye then", track=track)]

        self.backend = StraddlingBackend()
        finished = self._run(self._live())
        self.assertIn("goodbye then", finished.text)

    def test_without_live_results_the_whole_track_runs(self):
        finished = self._run(None)
        self.assertAlmostEqual(self.backend.calls[0].duration, self.SECONDS,
                               delta=0.2)
        self.assertIn("tail 1", finished.text)

    def test_the_start_offset_shifts_live_and_tail_alike(self):
        """A track that started late moves on the common timeline."""
        live = FakeLive({"mic": 4.0},
                        {"mic": [Segment(start=1.0, end=2.0, text="early",
                                         track="mic")]})
        finalizer = pipeline.Finalizer(self.bridge, Settings(output_dir=self.out),
                                       backend_factory=lambda s: self.backend,
                                       live=live)
        thread = finalizer.run_async(self._recording(offset=2.0), "meeting")
        thread.join(timeout=30.0)

        finished = self.bridge.of_type(Finished)[0]
        self.assertIn("[00:03]", finished.text)      # live 1.0 s + 2.0 s offset
        # The tail is cut at 4.0 + 2.0 s and ends with the padded track at
        # 62.0 s - one second before that is where the backend answers.
        self.assertIn("[01:01]", finished.text)


if __name__ == "__main__":
    unittest.main()
