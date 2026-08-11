"""Tests for the recording timeline (audit finding H3)."""

import itertools
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

from audio_transcriber.audio import capture, dsp

RATE = 48000
BLOCK = capture.BLOCK_FRAMES


class TestDriftCorrection(unittest.TestCase):
    """Both track timelines must stay tied to the system clock.

    In the previous version lost blocks (exception_on_overflow=False) were
    swallowed silently. Since both tracks were read independently they drifted
    apart - and the whole speaker attribution relied on the assumption that
    sample index equals time.
    """

    def test_no_correction_when_in_sync(self):
        elapsed = 100 * BLOCK / RATE
        written = 99 * BLOCK
        self.assertEqual(capture.drift_deficit(elapsed, RATE, written, BLOCK), 0)

    def test_dropout_produces_positive_deficit(self):
        # 500 ms have passed but only 100 ms were written
        deficit = capture.drift_deficit(0.5, RATE, int(0.1 * RATE), 0)
        self.assertGreater(deficit, 0)
        self.assertAlmostEqual(deficit / RATE, 0.4, places=2)

    def test_small_jitter_is_ignored(self):
        # 20 ms of deviation is below the 50 ms threshold
        deficit = capture.drift_deficit(0.52, RATE, int(0.5 * RATE), 0)
        self.assertEqual(deficit, 0)

    def test_device_clock_ahead_is_reported_negative(self):
        deficit = capture.drift_deficit(0.5, RATE, int(0.7 * RATE), 0)
        self.assertLess(deficit, 0)

    def test_correction_keeps_tracks_aligned(self):
        """Simulation: track A loses 300 ms halfway through, track B does not.
        Without correction A ends 300 ms early - with correction they match."""
        total_blocks = 200
        block_s = BLOCK / RATE
        dropout_s = 0.3

        for corrected in (False, True):
            frames = 0
            elapsed = 0.0
            for index in range(total_blocks):
                # The wall clock keeps running; during a dropout time passes
                # without samples arriving.
                elapsed += block_s
                if index == 100:
                    elapsed += dropout_s
                if corrected:
                    deficit = capture.drift_deficit(elapsed, RATE, frames, BLOCK)
                    if deficit > 0:
                        frames += deficit
                frames += BLOCK

            # On the common timeline the track should be as long as the wall
            # clock says.
            drift_s = abs(frames / RATE - elapsed)
            if corrected:
                self.assertLess(drift_s, 0.02,
                                "the correction does not compensate the dropout")
            else:
                self.assertAlmostEqual(drift_s, dropout_s, places=2)


class TestTrackLoading(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, seconds, rate=RATE):
        path = os.path.join(self.dir, name)
        data = np.sin(2 * np.pi * 440 * np.arange(int(seconds * rate)) / rate)
        sf.write(path, data.astype(np.float32), rate, subtype="PCM_16")
        return path

    def test_resamples_to_16k(self):
        path = self._write("track.wav", 2.0)
        result = capture.TrackResult(path=path, rate=RATE, frames=2 * RATE)
        audio = capture.load_track(result)
        self.assertAlmostEqual(len(audio) / dsp.TARGET_RATE, 2.0, places=2)

    def test_start_offset_is_padded_at_the_front(self):
        """The track whose stream started later missed the beginning and has
        to move back on the common timeline."""
        path = self._write("track.wav", 1.0)
        result = capture.TrackResult(path=path, rate=RATE, frames=RATE,
                                     start_offset_s=0.25)
        audio = capture.load_track(result)
        pad = int(0.25 * dsp.TARGET_RATE)
        np.testing.assert_allclose(audio[:pad], 0.0, atol=1e-7)
        self.assertGreater(float(np.max(np.abs(audio[pad:]))), 0.5)
        self.assertAlmostEqual(len(audio) / dsp.TARGET_RATE, 1.25, places=2)

    def test_missing_file_returns_empty(self):
        result = capture.TrackResult(path=os.path.join(self.dir, "gone.wav"),
                                     rate=RATE, frames=0)
        self.assertEqual(len(capture.load_track(result)), 0)
        self.assertEqual(len(capture.load_track(None)), 0)

    def test_duration_property(self):
        result = capture.TrackResult(path="x", rate=16000, frames=32000)
        self.assertAlmostEqual(result.duration_s, 2.0)


class TestConfigureIsAtomic(unittest.TestCase):
    """A reconfiguration must not be observable half-done.

    configure() closes the old streams and only then opens the new ones. It
    briefly holds no active track, and start_recording() reads exactly that
    state - so hitting Start right after a device change reported "Neither
    audio source is active" for a perfectly healthy device.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.engine = capture.AudioEngine(pa=None, tmp_dir=self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_is_configuring_reports_the_window(self):
        self.assertFalse(self.engine.is_configuring)

        seen = []
        barrier = threading.Event()
        released = threading.Event()

        def slow_configure(_mic, _sys):
            barrier.set()
            released.wait(timeout=5.0)
            return []

        # Stand in for the real device work; only the locking is under test.
        self.engine._configure = slow_configure

        worker = threading.Thread(
            target=lambda: self.engine.configure(None, None), daemon=True)
        worker.start()

        self.assertTrue(barrier.wait(timeout=5.0))
        seen.append(self.engine.is_configuring)
        released.set()
        worker.join(timeout=5.0)

        self.assertEqual(seen, [True], "is_configuring must be True mid-flight")
        self.assertFalse(self.engine.is_configuring)

    def test_two_reconfigurations_do_not_interleave(self):
        """Without the lock the calls interleave as A-enter, B-enter, ..."""
        events = []
        lock = threading.Lock()

        def tracked_configure(_mic, name):
            with lock:
                events.append(f"enter-{name}")
            time.sleep(0.05)
            with lock:
                events.append(f"leave-{name}")
            return []

        self.engine._configure = tracked_configure

        threads = [threading.Thread(target=self.engine.configure,
                                    args=(None, name), daemon=True)
                   for name in ("A", "B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        self.assertEqual(len(events), 4)
        # Every enter must be followed by its own leave.
        for index in range(0, 4, 2):
            self.assertTrue(events[index].startswith("enter-"), events)
            self.assertEqual(events[index + 1],
                             events[index].replace("enter-", "leave-"), events)


class _FakeStream:
    """Hands out prepared blocks, then ends the reader loop."""

    def __init__(self, blocks, stop):
        self._blocks = list(blocks)
        self._stop = stop

    def read(self, _frames, exception_on_overflow=False):
        if not self._blocks:
            self._stop.set()
            return b""
        return self._blocks.pop(0).astype(np.float32).tobytes()


class TestCaptureTap(unittest.TestCase):
    """The tap must see exactly the samples that reach the raw file.

    Live transcription counts its position in samples and hands that number to
    the closing pass as 'already done'. If the tap missed the silence the drift
    correction inserts, the two halves of the transcript would drift apart by
    exactly that much.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.engine = capture.AudioEngine(pa=None, tmp_dir=self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _read_blocks(self, blocks, clock_values):
        device = SimpleNamespace(index=0, name="fake", max_input_channels=1,
                                 default_rate=RATE)
        track = capture._Track("mic", device)
        stop = threading.Event()
        track.stream = _FakeStream(blocks, stop)

        path = os.path.join(self.dir, "raw.wav")
        track.writer = sf.SoundFile(path, mode="w", samplerate=RATE, channels=1,
                                    subtype="PCM_16")
        seen = []
        self.engine.set_tap(
            lambda kind, samples, rate, position:
            seen.append((kind, np.array(samples), rate, position)))

        clock = iter(clock_values)
        with patch.object(capture.time, "perf_counter", side_effect=lambda: next(clock)):
            self.engine._reader_loop(track, stop)
        track.writer.close()

        written, _rate = sf.read(path, dtype="float32")
        return seen, written

    def test_tap_matches_the_file_sample_for_sample(self):
        blocks = [np.full(BLOCK, 0.25, dtype=np.float32) for _ in range(4)]
        # Wall clock in step with the samples: no correction.
        clock = itertools.chain([0.0], (index * BLOCK / RATE for index in range(1, 5)))

        seen, written = self._read_blocks(blocks, clock)

        self.assertEqual([kind for kind, _s, _r, _p in seen], ["mic"] * 4)
        self.assertTrue(all(rate == RATE for _k, _s, rate, _p in seen))
        tapped = np.concatenate([samples for _k, samples, _r, _p in seen])
        self.assertEqual(len(tapped), len(written))
        np.testing.assert_allclose(tapped, written, atol=1e-4)

        # The reported position is where the block actually sits in the file.
        self.assertEqual([position for _k, _s, _r, position in seen],
                         [0, BLOCK, 2 * BLOCK, 3 * BLOCK])

    def test_inserted_silence_reaches_the_tap_as_well(self):
        blocks = [np.full(BLOCK, 0.25, dtype=np.float32) for _ in range(4)]
        # A full second passes between the first and the second block: the
        # drift correction pads the gap, and the tap has to see the padding.
        seen, written = self._read_blocks(blocks, itertools.chain([0.0],
                                                                 itertools.repeat(1.0)))

        self.assertEqual(len(seen), 5)                  # 4 blocks + one padding
        tapped = np.concatenate([samples for _k, samples, _r, _p in seen])
        self.assertEqual(len(tapped), len(written))
        np.testing.assert_allclose(tapped, written, atol=1e-4)

        padding = seen[1][1]
        self.assertGreater(len(padding), RATE * 0.9)
        self.assertEqual(float(np.max(np.abs(padding))), 0.0)

    def test_a_broken_tap_does_not_stop_the_capture(self):
        blocks = [np.full(BLOCK, 0.25, dtype=np.float32) for _ in range(3)]
        device = SimpleNamespace(index=0, name="fake", max_input_channels=1,
                                 default_rate=RATE)
        track = capture._Track("mic", device)
        stop = threading.Event()
        track.stream = _FakeStream(blocks, stop)
        path = os.path.join(self.dir, "raw.wav")
        track.writer = sf.SoundFile(path, mode="w", samplerate=RATE, channels=1,
                                    subtype="PCM_16")

        def exploding_tap(_kind, _samples, _rate, _position):
            raise RuntimeError("consumer is gone")

        self.engine.set_tap(exploding_tap)
        self.engine._reader_loop(track, stop)
        track.writer.close()

        self.assertIsNone(track.error)
        self.assertEqual(track.frames, 3 * BLOCK)
        self.assertIsNone(self.engine._tap)             # detached, not retried


if __name__ == "__main__":
    unittest.main()
