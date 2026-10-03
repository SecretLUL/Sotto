"""Tests for the Audio File Upload and Transcription feature."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from audio_transcriber.audio import dsp
from audio_transcriber.audio.loader import load_audio_file
from audio_transcriber.config import Settings
from audio_transcriber.events import Failed, Finished, Log, Progress, Status
from audio_transcriber.pipeline import FileFinalizer
from audio_transcriber.transcribe.base import Segment, TranscriptionError


class TestAudioLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_file(self):
        fake_path = os.path.join(self.temp_dir, "nonexistent.wav")
        with self.assertRaises(TranscriptionError) as ctx:
            load_audio_file(fake_path)
        self.assertIn("not found", str(ctx.exception))

    def test_empty_file(self):
        empty_path = os.path.join(self.temp_dir, "empty.wav")
        with open(empty_path, "wb") as f:
            f.write(b"")
        with self.assertRaises(TranscriptionError) as ctx:
            load_audio_file(empty_path)
        self.assertIn("empty", str(ctx.exception))

    def test_load_valid_mono_wav(self):
        wav_path = os.path.join(self.temp_dir, "sample.wav")
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate, dtype=np.float32)
        sine = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(wav_path, sine, sample_rate)

        loaded = load_audio_file(wav_path, target_rate=16000)
        self.assertIsInstance(loaded, np.ndarray)
        self.assertEqual(loaded.ndim, 1)
        self.assertAlmostEqual(len(loaded), 16000, delta=100)

    def test_load_valid_stereo_wav(self):
        wav_path = os.path.join(self.temp_dir, "stereo.wav")
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate, dtype=np.float32)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right])
        sf.write(wav_path, stereo, sample_rate)

        loaded = load_audio_file(wav_path, target_rate=16000)
        self.assertEqual(loaded.ndim, 1)
        self.assertEqual(len(loaded), 16000)

    @patch("audio_transcriber.audio.loader._load_via_ffmpeg")
    def test_ffmpeg_fallback_trigger(self, mock_ffmpeg):
        mock_ffmpeg.return_value = np.zeros(16000, dtype=np.float32)
        m4a_path = os.path.join(self.temp_dir, "dummy.m4a")
        with open(m4a_path, "wb") as f:
            f.write(b"fake m4a header data")

        loaded = load_audio_file(m4a_path, target_rate=16000)
        mock_ffmpeg.assert_called_once()
        self.assertEqual(len(loaded), 16000)


class TestMockBridge:
    def __init__(self):
        self.events = []

    def post(self, event):
        self.events.append(event)

    def post_exception(self, msg, exc):
        self.events.append(Failed(message=f"{msg}: {exc}"))


class TestMockBackend:
    def __init__(self, segments=None):
        self.segments = segments or [
            Segment(start=0.0, end=2.5, text="Hello world from file.", track="file")
        ]
        self.cancelled = False

    def transcribe(self, path, language="de", log=None, track="", progress=None):
        if log:
            log("Mock transcribing...\n")
        return self.segments

    def cancel(self):
        self.cancelled = True


class TestFileFinalizer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = os.path.join(self.temp_dir, "out")
        self.tmp_dir = os.path.join(self.temp_dir, "tmp")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Only TMP_DIR is read from the pipeline module namespace. The output
        # directory comes from Settings.get_output_dir(), so patching an
        # OUT_DIR here would do nothing and the tests would quietly write into
        # the real output/ folder - every Settings() below sets output_dir.
        self.paths_patcher = patch.multiple(
            "audio_transcriber.pipeline",
            TMP_DIR=self.tmp_dir
        )
        self.paths_patcher.start()

    def tearDown(self):
        self.paths_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_finalizer_success(self):
        # Create a test audio file
        wav_path = os.path.join(self.temp_dir, "test_input.wav")
        sample_rate = 16000
        t = np.linspace(0, 2, sample_rate * 2, dtype=np.float32)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(wav_path, audio_data, sample_rate)

        bridge = TestMockBridge()
        settings = Settings(output_dir=self.out_dir)
        backend = TestMockBackend()

        finalizer = FileFinalizer(bridge, settings, backend_factory=lambda s: backend)
        thread = finalizer.run_async(wav_path, base_name="test_file")
        thread.join(timeout=5.0)

        # Check posted events
        finished_events = [e for e in bridge.events if isinstance(e, Finished)]
        self.assertEqual(len(finished_events), 1)
        finished = finished_events[0]
        self.assertIn("Hello world from file.", finished.text)
        self.assertTrue(os.path.exists(finished.txt_path))
        self.assertTrue(os.path.exists(finished.audio_path))

    def test_file_finalizer_custom_output_dir(self):
        wav_path = os.path.join(self.temp_dir, "test_custom.wav")
        custom_out = os.path.join(self.temp_dir, "custom_target_folder")
        sample_rate = 16000
        t = np.linspace(0, 2, sample_rate * 2, dtype=np.float32)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(wav_path, audio_data, sample_rate)

        bridge = TestMockBridge()
        settings = Settings(output_dir=custom_out)
        backend = TestMockBackend()

        finalizer = FileFinalizer(bridge, settings, backend_factory=lambda s: backend)
        thread = finalizer.run_async(wav_path, base_name="custom_file")
        thread.join(timeout=5.0)

        finished_events = [e for e in bridge.events if isinstance(e, Finished)]
        self.assertEqual(len(finished_events), 1)
        finished = finished_events[0]
        self.assertEqual(os.path.dirname(finished.txt_path), os.path.abspath(custom_out))
        self.assertEqual(os.path.dirname(finished.audio_path), os.path.abspath(custom_out))
        self.assertTrue(os.path.exists(finished.txt_path))
        self.assertTrue(os.path.exists(finished.audio_path))

    def test_file_finalizer_derives_base_name_from_file(self):
        """Without an explicit base_name the name comes from the file itself.

        Regression: run_async called paths.safe_output_name() while pipeline.py
        only imported OUT_DIR/TMP_DIR from .paths, so this branch raised
        NameError. Every other test passed base_name explicitly and never
        reached it.
        """
        wav_path = os.path.join(self.temp_dir, "Team Meeting 2026.wav")
        sample_rate = 16000
        t = np.linspace(0, 2, sample_rate * 2, dtype=np.float32)
        sf.write(wav_path, 0.5 * np.sin(2 * np.pi * 440 * t), sample_rate)

        bridge = TestMockBridge()
        settings = Settings(output_dir=self.out_dir)
        backend = TestMockBackend()

        finalizer = FileFinalizer(bridge, settings, backend_factory=lambda s: backend)
        thread = finalizer.run_async(wav_path)          # no base_name
        thread.join(timeout=5.0)

        failed = [e for e in bridge.events if isinstance(e, Failed)]
        self.assertEqual(failed, [], f"unexpected failure: {failed}")

        finished = [e for e in bridge.events if isinstance(e, Finished)]
        self.assertEqual(len(finished), 1)
        self.assertEqual(os.path.basename(finished[0].txt_path),
                         "Team Meeting 2026.txt")

    def test_file_finalizer_keeps_dots_in_the_file_name(self):
        """'Meeting 03.10.2026.wav' lost '.2026' (a second, dotted piece was
        taken for another extension) and ended up as 'Meeting 03.10'."""
        wav_path = os.path.join(self.temp_dir, "Meeting 03.10.2026.wav")
        sample_rate = 16000
        t = np.linspace(0, 2, sample_rate * 2, dtype=np.float32)
        sf.write(wav_path, 0.5 * np.sin(2 * np.pi * 440 * t), sample_rate)

        bridge = TestMockBridge()
        finalizer = FileFinalizer(bridge, Settings(output_dir=self.out_dir),
                                  backend_factory=lambda s: TestMockBackend())
        finalizer.run_async(wav_path).join(timeout=5.0)

        finished = [e for e in bridge.events if isinstance(e, Finished)]
        self.assertEqual(len(finished), 1, bridge.events)
        self.assertEqual(os.path.basename(finished[0].txt_path),
                         "Meeting 03.10.2026.txt")
        self.assertEqual(os.path.basename(finished[0].audio_path),
                         "Meeting 03.10.2026.wav")

    def test_file_finalizer_silent_audio(self):

        silent_path = os.path.join(self.temp_dir, "silent.wav")
        sf.write(silent_path, np.zeros(16000, dtype=np.float32), 16000)

        bridge = TestMockBridge()
        settings = Settings(output_dir=self.out_dir)
        backend = TestMockBackend()

        finalizer = FileFinalizer(bridge, settings, backend_factory=lambda s: backend)
        thread = finalizer.run_async(silent_path, base_name="silent_test")
        thread.join(timeout=5.0)

        failed_events = [e for e in bridge.events if isinstance(e, Failed)]
        self.assertEqual(len(failed_events), 1)
        self.assertIn("silent", failed_events[0].message.lower())


class TestSaveTranscript(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("tkinter.filedialog.asksaveasfilename")
    @patch("tkinter.messagebox.showinfo")
    def test_save_transcript_success(self, mock_info, mock_dialog):
        save_path = os.path.join(self.temp_dir, "custom_transcript.txt")
        mock_dialog.return_value = save_path

        mock_app = MagicMock()
        mock_app.transcript.text.get.return_value = "[00:01] Hello test transcript"
        mock_app.recording_base_name = "test_rec"
        mock_app.filename_entry.get.return_value = "default_name"

        from audio_transcriber.ui.app import RecorderApp
        RecorderApp._save_transcript(mock_app)

        mock_dialog.assert_called_once()
        self.assertTrue(os.path.exists(save_path))
        with open(save_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("[00:01] Hello test transcript", content)
        mock_app.status.set.assert_called_once()


if __name__ == "__main__":
    unittest.main()

