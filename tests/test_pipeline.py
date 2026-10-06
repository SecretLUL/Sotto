"""End-to-end tests of the post-processing (audit findings K1, K4, H5)."""

import os
import queue
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_transcriber import config, pipeline
from audio_transcriber.audio import capture, dsp
from audio_transcriber.events import Failed, Finished, Log, Progress, Status
from audio_transcriber.transcribe.base import Backend, Segment, TranscriptionError

RATE = 48000

# A Turkish and an Arabic word: neither survives whisper-cli's ANSI arguments
# on a western Windows (code page 1252).
HARD_NAME = "toplantı_şirket_اجتماع"


class FakeBridge:
    """Collects events instead of forwarding them to Tkinter."""

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

    def first(self, event_type, events=None):
        for event in (events if events is not None else self.all()):
            if isinstance(event, event_type):
                return event
        return None


class FakeBackend(Backend):
    """Returns predefined segments per track and counts the calls."""

    calls = []

    def __init__(self, per_track):
        self.per_track = per_track

    def transcribe(self, wav_path, language="de", log=None, track="", progress=None):
        FakeBackend.calls.append((track, wav_path, language))
        return [Segment(start, end, text, track)
                for start, end, text in self.per_track.get(track, [])]

    def cancel(self):
        pass


class TestFinalizer(unittest.TestCase):
    def setUp(self):
        FakeBackend.calls = []
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "output")
        self.tmp = os.path.join(self.out, ".tmp")
        os.makedirs(self.tmp)

        # Only TMP_DIR is read from the pipeline module namespace. The output
        # directory comes from Settings.get_output_dir(), so it has to be
        # steered via output_dir below - patching a pipeline.OUT_DIR would do
        # nothing and the run would land in the real output/ folder.
        self._orig_tmp = pipeline.TMP_DIR
        pipeline.TMP_DIR = self.tmp

        self.settings = config.Settings(model="small", mic_gain_db=-8.0,
                                        loop_gain_db=10.0, language="en",
                                        keep_raw_tracks=False,
                                        output_dir=self.out)
        self.bridge = FakeBridge()

    def tearDown(self):
        pipeline.TMP_DIR = self._orig_tmp
        shutil.rmtree(self.dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def _make_recording(self, seconds=10.0, bursts=None):
        """Two raw tracks made of noise bursts.

        Default: you speak from 1-3 s, the other party from 5-7 s. Segments a
        test claims later must also exist acoustically here - otherwise the
        hallucination filter drops them, and rightly so.
        """
        bursts = bursts or {"mic": [(1.0, 3.0)], "sys": [(5.0, 7.0)]}
        rng = np.random.default_rng(4)
        result = capture.RecordingResult()
        for kind in ("mic", "sys"):
            data = np.zeros(int(seconds * RATE), dtype=np.float32)
            for start, end in bursts.get(kind, []):
                i_start, i_end = int(start * RATE), int(end * RATE)
                data[i_start:i_end] = rng.normal(0, 0.2, i_end - i_start)
            path = os.path.join(self.tmp, f"session.{kind}.raw.wav")
            sf.write(path, data, RATE, subtype="PCM_16")
            setattr(result, kind, capture.TrackResult(
                path=path, rate=RATE, frames=len(data), device_name=kind))
        return result

    def _run(self, recording, per_track):
        finalizer = pipeline.Finalizer(
            self.bridge, self.settings,
            backend_factory=lambda s: FakeBackend(per_track))
        finalizer._process(recording, "session")
        return self.bridge.all()

    # ------------------------------------------------------------------
    def test_full_run_produces_mix_and_transcript(self):
        recording = self._make_recording()
        events = self._run(recording, {
            "mic": [(1.0, 3.0, "Hello, can you hear me?")],
            "sys": [(5.0, 7.0, "Yes, loud and clear.")],
        })

        finished = self.bridge.first(Finished, events)
        self.assertIsNotNone(finished, "no Finished event was posted")

        # Audible mixdown: both voices in the middle, at the recorded rate
        info = sf.info(finished.audio_path)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.samplerate, RATE)
        self.assertAlmostEqual(info.frames / info.samplerate, 10.0, places=1)

        with open(finished.txt_path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("[00:01] [You]: Hello, can you hear me?", text)
        self.assertIn("[00:05] [Participant]: Yes, loud and clear.", text)

    def test_both_tracks_are_transcribed_separately(self):
        """The core decision: separate tracks instead of guessed attribution."""
        self.settings.separate_tracks = True
        self._run(self._make_recording(), {"mic": [(1.0, 3.0, "A")],
                                           "sys": [(5.0, 7.0, "B")]})
        tracks = sorted(call[0] for call in FakeBackend.calls)
        self.assertEqual(tracks, ["mic", "sys"])

    def test_transcript_covers_the_end_of_the_recording(self):
        """Regression K4: in the previous version the saved transcript was
        missing the end of the recording (16.6 % measured on a real file)
        because the last live pass was written as the final result."""
        recording = self._make_recording(
            seconds=10.0,
            bursts={"mic": [(1.0, 3.0)], "sys": [(5.0, 7.0), (8.5, 9.8)]})
        events = self._run(recording, {
            "mic": [(1.0, 3.0, "Beginning")],
            "sys": [(5.0, 7.0, "Middle"), (8.5, 9.8, "And this is the end.")],
        })
        finished = self.bridge.first(Finished, events)
        self.assertIn("And this is the end.", finished.text)
        # The last segment sits in the final 15 % of the recording - exactly
        # the part the previous version lost systematically.
        self.assertTrue(finished.text.rstrip().endswith("And this is the end."))

    def test_cloud_backend_is_reached_when_selected(self):
        """Regression K1: with the live preview enabled the cloud path was
        unreachable in the previous version."""
        self.settings.model = config.CLOUD_MODEL      # ElevenLabs
        self.settings.live_transcribe = True
        used = []

        def factory(settings):
            used.append(settings.uses_cloud())
            return FakeBackend({"mic": [(1.0, 3.0, "Cloud")], "sys": []})

        finalizer = pipeline.Finalizer(self.bridge, self.settings,
                                       backend_factory=factory)
        finalizer._process(self._make_recording(), "session")
        self.assertTrue(used and all(used), "the cloud path was not used")

    def test_gain_settings_do_not_change_speaker_assignment(self):
        """Regression H2 at pipeline level."""
        segments = {"mic": [(1.0, 3.0, "I am talking.")],
                    "sys": [(5.0, 7.0, "And so am I.")]}

        results = []
        for mic_gain, sys_gain in ((0.0, 0.0), (-8.0, 10.0), (12.0, -15.0)):
            self.setUp()
            self.settings.mic_gain_db = mic_gain
            self.settings.loop_gain_db = sys_gain
            events = self._run(self._make_recording(), segments)
            results.append(self.bridge.first(Finished, events).text)
        self.assertEqual(len(set(results)), 1,
                         "the gain sliders influence speaker attribution")

    def test_temporary_tracks_are_removed(self):
        recording = self._make_recording()
        raw_paths = [recording.mic.path, recording.sys.path]
        self._run(recording, {"mic": [(1.0, 3.0, "A")], "sys": []})
        for path in raw_paths:
            self.assertFalse(os.path.exists(path), f"{path} was left behind")
        leftovers = [name for name in os.listdir(self.tmp)
                     if name.endswith(".asr.wav")]
        self.assertEqual(leftovers, [])

    def test_kept_raw_tracks_land_next_to_the_transcript(self):
        """README: 'Keep raw tracks' preserves the unmixed tracks for a DAW. They
        used to stay in the hidden temp folder - overwritten by the next run of
        the same name, and invisible in the output folder the user chose."""
        self.settings.keep_raw_tracks = True
        recording = self._make_recording()
        original = [recording.mic.path, recording.sys.path]
        events = self._run(recording, {"mic": [(1.0, 3.0, "A")], "sys": []})

        for kind in ("mic", "sys"):
            kept = os.path.join(self.out, f"session.{kind}.wav")
            self.assertTrue(os.path.exists(kept), kept)
            info = sf.info(kept)
            self.assertEqual((info.samplerate, info.channels), (RATE, 1))
        for path in original:
            self.assertFalse(os.path.exists(path), f"{path} was not moved")

        logs = " ".join(e.text for e in events if isinstance(e, Log))
        self.assertIn("Kept the microphone track: session.mic.wav", logs)
        self.assertIn("Kept the system track: session.sys.wav", logs)

    def test_without_the_option_no_raw_track_is_left_anywhere(self):
        self.settings.keep_raw_tracks = False
        recording = self._make_recording()
        self._run(recording, {"mic": [(1.0, 3.0, "A")], "sys": []})

        self.assertFalse(os.path.exists(recording.mic.path))
        self.assertFalse(os.path.exists(recording.sys.path))
        self.assertFalse([n for n in os.listdir(self.out)
                          if n.endswith((".mic.wav", ".sys.wav"))])

    def test_a_track_that_cannot_be_kept_stays_and_the_run_still_finishes(self):
        """The transcript is done by then - failing over a move would throw it
        away. The track stays in the temp folder, where the recovery finds it."""
        self.settings.keep_raw_tracks = True
        recording = self._make_recording()
        with patch("audio_transcriber.pipeline._move_file",
                   side_effect=OSError("disk full")):
            events = self._run(recording, {"mic": [(1.0, 3.0, "A")], "sys": []})

        self.assertIsNotNone(self.bridge.first(Finished, events))
        self.assertTrue(os.path.exists(recording.mic.path))
        logs = " ".join(e.text for e in events if isinstance(e, Log))
        self.assertIn("could not be kept (disk full)", logs)

    def test_moving_falls_back_to_copying_across_drives(self):
        folder = tempfile.mkdtemp()
        try:
            source = os.path.join(folder, "a.raw.wav")
            target = os.path.join(folder, "a.mic.wav")
            with open(source, "wb") as handle:
                handle.write(b"audio")
            with patch("audio_transcriber.pipeline.os.replace",
                       side_effect=OSError("different drive")):
                pipeline._move_file(source, target)
            self.assertFalse(os.path.exists(source))
            with open(target, "rb") as handle:
                self.assertEqual(handle.read(), b"audio")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    # ------------------------------------------------------------------
    def _spy(self, per_track):
        """A backend factory that notes every file it is asked to read."""
        seen = []

        class SpyBackend(FakeBackend):
            def transcribe(self, wav_path, language="de", log=None, track="",
                           progress=None):
                seen.append((os.path.basename(wav_path), os.path.exists(wav_path)))
                return super().transcribe(wav_path, language, log, track, progress)

        return seen, (lambda settings: SpyBackend(per_track))

    def _assert_recognition_files_are_ascii(self):
        seen, factory = self._spy({"mic": [(1.0, 3.0, "Merhaba")],
                                   "sys": [(5.0, 7.0, "Hello")]})
        finalizer = pipeline.Finalizer(self.bridge, self.settings,
                                       backend_factory=factory)
        finalizer._process(self._make_recording(), HARD_NAME)

        self.assertTrue(seen, "the backend was never called")
        for file_name, existed in seen:
            self.assertTrue(file_name.isascii(), file_name)
            self.assertTrue(existed, "the file must exist when whisper is called")
        # What the user gets to see keeps the name they chose.
        finished = self.bridge.first(Finished)
        self.assertEqual(os.path.basename(finished.txt_path), f"{HARD_NAME}.txt")
        self.assertEqual(os.path.basename(finished.audio_path), f"{HARD_NAME}.wav")

    def test_recognition_files_have_ascii_names_per_track(self):
        """whisper-cli takes ANSI arguments: a Turkish or Arabic recording name
        arrived mangled and it stopped with 'input file not found' (exit 2)."""
        self.settings.separate_tracks = True
        self._assert_recognition_files_are_ascii()

    def test_recognition_files_have_ascii_names_for_the_mixdown(self):
        self.settings.separate_tracks = False
        self._assert_recognition_files_are_ascii()

    def test_scratch_files_go_even_when_raw_tracks_are_kept(self):
        """Only the recorded originals are 'raw tracks'; the normalised copies
        made for recognition used to pile up in .tmp with that option on."""
        self.settings.keep_raw_tracks = True
        recording = self._make_recording()
        self._run(recording, {"mic": [(1.0, 3.0, "A")], "sys": []})

        self.assertTrue(os.path.exists(os.path.join(self.out, "session.mic.wav")),
                        "the recorded track is what gets kept")
        leftovers = [name for name in os.listdir(self.tmp)
                     if name.endswith(".asr.wav")]
        self.assertEqual(leftovers, [])

    def test_scratch_files_go_when_recognition_fails(self):
        class FailingBackend(FakeBackend):
            def transcribe(self, *args, **kwargs):
                raise TranscriptionError("model could not be loaded")

        recording = self._make_recording()
        finalizer = pipeline.Finalizer(
            self.bridge, self.settings,
            backend_factory=lambda s: FailingBackend({}))
        finalizer._run(recording, "session")

        self.assertIsNotNone(self.bridge.first(Failed))
        leftovers = [name for name in os.listdir(self.tmp)
                     if name.endswith(".asr.wav")]
        self.assertEqual(leftovers, [])
        self.assertTrue(os.path.exists(recording.mic.path),
                        "the recorded tracks must survive a failed run")

    def test_silent_recording_reports_clear_error(self):
        """Regression M3: the previous version only said 'no data'."""
        recording = capture.RecordingResult()
        for kind in ("mic", "sys"):
            path = os.path.join(self.tmp, f"silent.{kind}.raw.wav")
            sf.write(path, np.zeros(RATE * 3, dtype=np.float32), RATE,
                     subtype="PCM_16")
            setattr(recording, kind, capture.TrackResult(
                path=path, rate=RATE, frames=RATE * 3, device_name=kind))

        finalizer = pipeline.Finalizer(
            self.bridge, self.settings,
            backend_factory=lambda s: FakeBackend({}))
        finalizer._run(recording, "silent")

        failed = self.bridge.first(Failed)
        self.assertIsNotNone(failed)
        self.assertIn("silent", failed.message.lower())

    def test_backend_error_becomes_failed_event(self):
        """Regression K2: errors must not disappear into a NameError."""
        class ExplodingBackend(FakeBackend):
            def transcribe(self, *args, **kwargs):
                raise RuntimeError("model could not be loaded")

        finalizer = pipeline.Finalizer(
            self.bridge, self.settings,
            backend_factory=lambda s: ExplodingBackend({}))
        finalizer._run(self._make_recording(), "session")

        failed = self.bridge.first(Failed)
        self.assertIsNotNone(failed, "the error was swallowed")
        self.assertIn("model could not be loaded", failed.message)

    def test_start_offset_is_applied(self):
        recording = self._make_recording()
        recording.sys.start_offset_s = 0.5
        events = self._run(recording, {"mic": [], "sys": [(5.5, 7.5, "Offset")]})
        finished = self.bridge.first(Finished, events)
        self.assertIn("Offset", finished.text)


if __name__ == "__main__":
    unittest.main()
