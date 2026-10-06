"""Tests for the recording timeline (audit finding H3)."""

import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_transcriber.audio import capture, devices, dsp

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


class TestMixdown(unittest.TestCase):
    """The .wav next to the transcript: the copy you listen to."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "mix.wav")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _track(self, kind, data, rate=RATE, offset=0.0):
        path = os.path.join(self.dir, f"session.{kind}.raw.wav")
        sf.write(path, np.asarray(data, dtype=np.float32), rate, subtype="PCM_16")
        return capture.TrackResult(path=path, rate=rate, frames=len(data),
                                   start_offset_s=offset)

    @staticmethod
    def _tone(freq, seconds, rate=RATE, amplitude=0.3):
        t = np.arange(int(seconds * rate)) / rate
        return amplitude * np.sin(2 * np.pi * freq * t)

    def _write(self, mic=None, sys=None, **gains):
        recording = capture.RecordingResult(mic=mic, sys=sys)
        rate, frames = capture.write_mixdown(recording, self.out, **gains)
        data, file_rate = sf.read(self.out, dtype="float32", always_2d=True)
        self.assertEqual((file_rate, len(data)), (rate, frames))
        return data, file_rate

    def test_both_voices_in_one_mono_file(self):
        """They used to be split: microphone left, system audio right."""
        data, _rate = self._write(
            mic=self._track("mic", np.r_[self._tone(440, 1.0), np.zeros(RATE)]),
            sys=self._track("sys", np.r_[np.zeros(RATE), self._tone(440, 1.0)]))
        self.assertEqual(data.shape[1], 1)
        first, second = data[:RATE, 0], data[RATE:, 0]
        self.assertAlmostEqual(dsp.rms(first), dsp.rms(second), places=2)
        self.assertGreater(dsp.rms(first), 0.15)

    def test_the_full_rate_is_kept(self):
        """At 16 kHz everything above 8 kHz was gone."""
        data, rate = self._write(mic=self._track("mic", self._tone(12000, 1.0)))
        self.assertEqual(rate, RATE)
        self.assertGreater(dsp.rms(data[:, 0]), 0.15)

    def test_a_slower_track_is_resampled_to_the_faster_one(self):
        data, rate = self._write(
            mic=self._track("mic", self._tone(440, 2.0, rate=44100), rate=44100),
            sys=self._track("sys", self._tone(440, 1.0)))
        self.assertEqual(rate, RATE)
        self.assertAlmostEqual(len(data) / rate, 2.0, places=2)

    def test_a_track_that_started_later_moves_back(self):
        data, rate = self._write(
            mic=self._track("mic", self._tone(440, 0.5)),
            sys=self._track("sys", self._tone(440, 1.0), offset=1.5))
        pad = int(1.5 * rate)
        np.testing.assert_allclose(data[int(0.5 * rate):pad, 0], 0.0, atol=1e-4)
        self.assertGreater(dsp.rms(data[pad:, 0]), 0.15)
        self.assertAlmostEqual(len(data) / rate, 2.5, places=2)

    def test_the_gain_sliders_apply(self):
        quiet, _rate = self._write(mic=self._track("mic", self._tone(440, 1.0)),
                                   mic_gain_db=-6.0)
        self.assertAlmostEqual(dsp.rms(quiet[:, 0]),
                               dsp.rms(self._tone(440, 1.0)) / 2, places=2)

    def test_a_loud_mix_is_lowered_but_never_raised(self):
        loud, _rate = self._write(
            mic=self._track("mic", self._tone(440, 1.0, amplitude=0.9)),
            sys=self._track("sys", self._tone(440, 1.0, amplitude=0.9)))
        self.assertLessEqual(float(np.max(np.abs(loud))), 0.951)
        self.assertGreater(float(np.max(np.abs(loud))), 0.9)

        soft, _rate = self._write(
            mic=self._track("mic", self._tone(440, 1.0, amplitude=0.1)))
        self.assertAlmostEqual(float(np.max(np.abs(soft))), 0.1, places=2)

    def test_longer_than_one_block(self):
        """Blocks of both tracks must line up across the block boundaries."""
        seconds = capture.MIX_BLOCK_S * 2.5
        tone = self._tone(440, seconds, amplitude=0.2)
        data, rate = self._write(mic=self._track("mic", tone),
                                 sys=self._track("sys", tone))
        self.assertEqual(len(data), len(tone))
        np.testing.assert_allclose(data[:, 0], 2 * tone, atol=2e-3)

    def test_a_missing_track_is_left_out(self):
        gone = capture.TrackResult(path=os.path.join(self.dir, "gone.wav"),
                                   rate=44100, frames=0)
        data, rate = self._write(mic=self._track("mic", self._tone(440, 1.0)),
                                 sys=gone)
        self.assertEqual(rate, RATE)
        self.assertAlmostEqual(len(data) / rate, 1.0, places=2)


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


# ----------------------------------------------------------------------
class FakeStream:
    """A device that delivers `blocks` blocks at real-time pace, then dies."""

    def __init__(self, blocks, rate=16000):
        self.blocks = blocks
        self.rate = rate

    def start_stream(self):
        pass

    def stop_stream(self):
        pass

    def close(self):
        pass

    def read(self, frames, exception_on_overflow=False):
        if self.blocks <= 0:
            raise OSError("device unplugged")
        self.blocks -= 1
        time.sleep(frames / float(self.rate))
        return np.full(frames, 0.1, dtype=np.float32).tobytes()


class FakePortAudio:
    def __init__(self, blocks=5, open_error=None):
        self.blocks = blocks
        self.open_error = open_error

    def open(self, **_kwargs):
        if self.open_error is not None:
            raise self.open_error
        return FakeStream(self.blocks)


def usb_microphone():
    return devices.Device(index=3, name="USB Microphone", host_api="WASAPI",
                          max_input_channels=1, max_output_channels=0,
                          default_rate=16000, is_loopback=False)


class TestSourceFailure(unittest.TestCase):
    """A source that stops delivering audio used to vanish without a word.

    Its capture thread ended, the recording went on with the other track and the
    dead one simply stopped; track.error was only ever read when Start failed.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.engine = None

    def tearDown(self):
        if self.engine is not None:
            self.engine.stop_streams()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _engine(self, **options):
        self.engine = capture.AudioEngine(FakePortAudio(**options), self.dir)
        return self.engine

    @staticmethod
    def _wait_for_error(engine, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if engine.stream_errors():
                return True
            time.sleep(0.02)
        return False

    def test_a_source_dying_mid_recording_is_noted_in_the_result(self):
        engine = self._engine(blocks=5)
        self.assertEqual(engine.configure(usb_microphone(), None), [])
        engine.start_recording("take")
        self.assertTrue(self._wait_for_error(engine))
        result = engine.stop_recording()

        self.assertTrue(result.has_audio, "what it delivered is kept")
        notes = " ".join(result.warnings)
        self.assertIn("USB Microphone", notes)
        self.assertIn("stopped delivering audio", notes)
        self.assertIn("device unplugged", notes)
        self.assertIn("silent", notes)

    def test_the_window_hears_about_it_once(self):
        engine = self._engine(blocks=3)
        engine.configure(usb_microphone(), None)
        self.assertTrue(self._wait_for_error(engine))

        first = engine.new_stream_errors()
        self.assertEqual([kind for kind, _message in first], ["mic"])
        self.assertIn("device unplugged", first[0][1])
        self.assertEqual(engine.new_stream_errors(), [])

    def test_a_stream_that_cannot_be_opened_is_not_reported_twice(self):
        engine = self._engine(open_error=OSError("device busy"))
        warnings = engine.configure(usb_microphone(), None)

        self.assertEqual(len(warnings), 1)
        self.assertIn("could not be opened", warnings[0])
        self.assertEqual(engine.new_stream_errors(), [],
                         "configure() already returned it as a warning")

    def test_a_healthy_recording_gets_no_failure_note(self):
        engine = self._engine(blocks=10_000)
        engine.configure(usb_microphone(), None)
        engine.start_recording("take")
        time.sleep(0.4)
        result = engine.stop_recording()
        self.assertFalse([w for w in result.warnings if "stopped delivering" in w])
        self.assertEqual(engine.new_stream_errors(), [])


# ----------------------------------------------------------------------
class TestUnfinishedRecordings(unittest.TestCase):
    """Raw tracks that never reached a transcript can be found and processed."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _raw(self, base, kind, seconds=2.0, rate=RATE, age_s=0.0):
        path = os.path.join(self.dir, f"{base}.{kind}.raw.wav")
        data = np.random.default_rng(0).normal(0, 0.1, int(seconds * rate))
        sf.write(path, data.astype(np.float32), rate, subtype="PCM_16")
        if age_s:
            moment = time.time() - age_s
            os.utime(path, (moment, moment))
        return path

    def test_tracks_are_grouped_by_recording(self):
        self._raw("Meeting 03.10.2026", "mic", seconds=2.0)
        self._raw("Meeting 03.10.2026", "sys", seconds=3.0)
        self._raw("other", "mic")
        found = capture.find_unfinished(self.dir)

        self.assertEqual(sorted(item.base_name for item in found),
                         ["Meeting 03.10.2026", "other"])
        meeting = next(item for item in found
                       if item.base_name == "Meeting 03.10.2026")
        self.assertEqual(sorted(meeting.tracks), ["mic", "sys"])
        self.assertAlmostEqual(meeting.duration_s, 3.0, places=2)
        self.assertEqual(meeting.tracks["mic"].rate, RATE)

    def test_the_newest_comes_first(self):
        self._raw("old", "mic", age_s=3600)
        self._raw("new", "mic")
        self._raw("middle", "mic", age_s=60)
        self.assertEqual([item.base_name for item in capture.find_unfinished(self.dir)],
                         ["new", "middle", "old"])

    def test_recordings_in_use_are_left_out(self):
        self._raw("running", "mic")
        self._raw("waiting", "mic")
        found = capture.find_unfinished(self.dir, exclude={"running"})
        self.assertEqual([item.base_name for item in found], ["waiting"])

    def test_empty_foreign_scratch_and_kept_files_are_ignored(self):
        sf.write(os.path.join(self.dir, "empty.mic.raw.wav"),
                 np.zeros(0, dtype=np.float32), RATE, subtype="PCM_16")
        with open(os.path.join(self.dir, "notes.txt"), "w") as handle:
            handle.write("x")
        with open(os.path.join(self.dir, "broken.mic.raw.wav"), "wb") as handle:
            handle.write(b"not audio at all")
        for name in ("mic-3f9a1c2e7b.asr.wav", "done.mic.wav", "x.raw.wav"):
            sf.write(os.path.join(self.dir, name),
                     np.ones(1000, dtype=np.float32) * 0.1, RATE, subtype="PCM_16")
        self.assertEqual(capture.find_unfinished(self.dir), [])

    def test_a_missing_folder_is_not_an_error(self):
        self.assertEqual(capture.find_unfinished(os.path.join(self.dir, "nope")), [])

    def test_a_track_that_was_never_closed_is_still_found(self):
        """The recording process died: the WAV was never closed, its header
        never finished. libsndfile reads the data up to the end of the file."""
        path = os.path.join(self.dir, "crashed.mic.raw.wav")
        script = "; ".join([
            "import os, numpy as np, soundfile as sf",
            f"f = sf.SoundFile({path!r}, mode='w', samplerate=48000, channels=1, "
            f"subtype='PCM_16')",
            "[f.write(np.full(1024, 0.1, dtype='float32')) for _ in range(50)]",
            "os._exit(1)",                     # killed: never closed
        ])
        subprocess.run([sys.executable, "-c", script], check=False)

        found = capture.find_unfinished(self.dir)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].tracks["mic"].frames, 50 * 1024)

    def test_the_finalizer_gets_a_ready_made_recording(self):
        self._raw("meeting", "mic", seconds=2.0)
        self._raw("meeting", "sys", seconds=2.0)
        unfinished = capture.find_unfinished(self.dir)[0]
        recording = capture.recording_from_unfinished(unfinished)

        self.assertTrue(recording.has_audio)
        self.assertEqual(recording.mic.start_offset_s, 0.0)
        self.assertIn("Recovered", recording.warnings[0])
        audio = capture.load_track(recording.mic)
        self.assertAlmostEqual(len(audio) / dsp.TARGET_RATE, 2.0, places=1)

    def test_removing_deletes_the_tracks(self):
        mic = self._raw("meeting", "mic")
        sys_track = self._raw("meeting", "sys")
        capture.remove_unfinished(capture.find_unfinished(self.dir)[0])
        self.assertFalse(os.path.exists(mic))
        self.assertFalse(os.path.exists(sys_track))
        self.assertEqual(capture.find_unfinished(self.dir), [])

    def test_an_empty_recording_discards_its_files(self):
        path = os.path.join(self.dir, "empty.mic.raw.wav")
        sf.write(path, np.zeros(0, dtype=np.float32), RATE, subtype="PCM_16")
        recording = capture.RecordingResult(
            mic=capture.TrackResult(path=path, rate=RATE, frames=0))
        self.assertFalse(recording.has_audio)
        recording.discard()
        self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
