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

    def _live(self, **kwargs):
        options = dict(chunk_s=2.0, preview=True)
        options.update(kwargs)
        live = pipeline.LiveTranscriber(
            self.bridge, Settings(), self.engine,
            backend_factory=lambda s: self.backend, **options)
        live.start("meeting")
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
        self.backend = FakeBackend(text="tail")

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _recording(self, seconds=10.0, offset=0.0):
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

    def test_live_lines_end_up_in_the_saved_transcript(self):
        live = FakeLive({"mic": 8.0},
                        {"mic": [Segment(start=1.0, end=2.0, text="said live",
                                         track="mic")]})
        finished = self._run(live)
        self.assertTrue(live.finished)
        self.assertIn("said live", finished.text)
        with open(finished.txt_path, encoding="utf-8") as handle:
            self.assertIn("said live", handle.read())

    def test_only_the_tail_is_sent_to_the_recogniser(self):
        live = FakeLive({"mic": 8.0},
                        {"mic": [Segment(start=1.0, end=2.0, text="said live",
                                         track="mic")]})
        self._run(live)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertAlmostEqual(self.backend.calls[0].duration, 2.0,
                               delta=0.2)                       # 10 s - 8 s

    def test_tail_segments_are_moved_onto_the_full_timeline(self):
        live = FakeLive({"mic": 8.0}, {"mic": []})
        finished = self._run(live)
        # The backend reports 1.0 s inside the tail, which starts at 8.0 s.
        self.assertIn("[00:09]", finished.text)

    def test_without_live_results_the_whole_track_runs(self):
        finished = self._run(None)
        self.assertAlmostEqual(self.backend.calls[0].duration, 10.0, delta=0.2)
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
        # The tail starts at 4.0 + 2.0 s, so its 1.0 s segment lands at 7 s.
        self.assertIn("[00:07]", finished.text)


if __name__ == "__main__":
    unittest.main()
