"""Integration tests against the real whisper-cli.exe.

Skipped when the binary or the model is missing.
"""

import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_transcriber import paths
from audio_transcriber.transcribe.base import TranscriptionError
from audio_transcriber.transcribe.whispercpp import WhisperCppBackend

TINY = paths.model_path("tiny")
SAMPLE = os.path.join(paths.OUT_DIR, ".tmp", "integration_sample.wav")


def _have_whisper():
    return os.path.exists(paths.WHISPER_EXE) and os.path.exists(TINY)


def _find_sample_audio():
    """Any existing recording in output/ works as sample material."""
    if not os.path.isdir(paths.OUT_DIR):
        return None
    for name in sorted(os.listdir(paths.OUT_DIR)):
        if name.endswith(".wav"):
            return os.path.join(paths.OUT_DIR, name)
    return None


class TestCommandBuilder(unittest.TestCase):
    def test_default_flags(self):
        backend = WhisperCppBackend("small", threads=10, use_vad=True)
        command = backend.build_command("whisper.exe", "model.bin", "a.wav", "de",
                                        vad_model="vad.bin")
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], "10")
        self.assertIn("-ng", command)      # Vulkan crashes on this build
        self.assertIn("-np", command)      # no diagnostics in the transcript
        self.assertIn("-sns", command)     # suppress non-speech tokens
        self.assertIn("--vad", command)
        self.assertEqual(command[command.index("-l") + 1], "de")

    def test_greedy_mode_for_live_preview(self):
        command = WhisperCppBackend("tiny", greedy=True).build_command(
            "w.exe", "m.bin", "a.wav", "de")
        self.assertEqual(command[command.index("-bs") + 1], "1")
        self.assertEqual(command[command.index("-bo") + 1], "1")

    def test_vad_omitted_when_model_missing(self):
        command = WhisperCppBackend("tiny", use_vad=True).build_command(
            "w.exe", "m.bin", "a.wav", "de", vad_model=None)
        self.assertNotIn("--vad", command)

    def test_vad_is_off_by_default(self):
        """It merges distant speech regions on this build - see whispercpp.py."""
        self.assertFalse(WhisperCppBackend("tiny").use_vad)

    def test_missing_audio_file_raises(self):
        with self.assertRaises(TranscriptionError):
            WhisperCppBackend("tiny").transcribe("does-not-exist.wav")


class TestAnsiSafeArguments(unittest.TestCase):
    """whisper-cli reads its arguments in the ANSI code page."""

    def test_model_input_and_vad_paths_are_made_safe_the_executable_is_not(self):
        backend = WhisperCppBackend("tiny", use_vad=True)
        backend.prepare = lambda progress=None, log=None: (
            "C:/bin/whisper-cli.exe", "C:/S\u0131rin/model.bin", "C:/S\u0131rin/vad.bin")
        commands = []

        class FakeProcess:
            returncode = 0

            def __init__(self):
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")

            def wait(self, timeout=None):
                return 0

        def fake_popen(command, **_kwargs):
            commands.append(command)
            return FakeProcess()

        with tempfile.TemporaryDirectory() as folder:
            wav = os.path.join(folder, "input.wav")
            open(wav, "wb").close()
            with patch("audio_transcriber.transcribe.whispercpp.subprocess.Popen",
                       fake_popen), \
                    patch("audio_transcriber.paths.ansi_safe_path",
                          lambda path: f"SAFE:{os.path.basename(path)}"):
                backend.transcribe(wav, language="de", track="mic")

        command = commands[0]
        self.assertEqual(command[0], "C:/bin/whisper-cli.exe")   # started via the wide API
        self.assertEqual(command[command.index("-m") + 1], "SAFE:model.bin")
        self.assertEqual(command[command.index("-f") + 1], "SAFE:input.wav")
        self.assertEqual(command[command.index("-vm") + 1], "SAFE:vad.bin")


@unittest.skipUnless(_have_whisper(), "whisper-cli.exe or ggml-tiny.bin missing")
class TestRealTranscription(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(os.path.dirname(SAMPLE), exist_ok=True)
        source = _find_sample_audio()
        if source:
            data, rate = sf.read(source, dtype="float32", always_2d=True)
            mono = data.mean(axis=1)[:rate * 25]
            sf.write(SAMPLE, mono, rate, subtype="PCM_16")
        else:
            sf.write(SAMPLE, np.zeros(16000 * 3, dtype=np.float32), 16000,
                     subtype="PCM_16")

    def test_produces_segments_with_subsecond_timestamps(self):
        backend = WhisperCppBackend("tiny", threads=8, use_vad=False)
        segments = backend.transcribe(SAMPLE, language="de", track="mic")

        if not segments:
            self.skipTest("the sample file contains no recognisable speech")

        self.assertTrue(all(segment.track == "mic" for segment in segments))
        self.assertTrue(all(segment.end >= segment.start for segment in segments))
        starts = [segment.start for segment in segments]
        self.assertEqual(starts, sorted(starts))
        # Regression H7: at least one timestamp carries a fractional part
        self.assertTrue(any(abs(segment.start - round(segment.start)) > 1e-6
                            for segment in segments),
                        "timestamps were rounded to whole seconds")

    @unittest.skipUnless(os.name == "nt", "ANSI arguments are a Windows matter")
    def test_input_in_a_non_ansi_folder_with_a_non_ansi_name(self):
        """Turkish and Arabic names made whisper-cli exit with code 2, 'input
        file not found' - on a western Windows it only sees the ANSI form."""
        base = tempfile.mkdtemp()
        try:
            folder = os.path.join(base, "Sırin_şirket")
            os.makedirs(folder)
            target = os.path.join(
                folder, "toplantı_اجتماع.wav")
            sf.write(target, np.random.default_rng(0).normal(0, 0.05, 32000)
                     .astype("float32"), 16000, subtype="PCM_16")
            if not paths.ansi_safe_path(target).isascii():
                self.skipTest("this volume has no 8.3 short names")

            # Raises TranscriptionError ("failed with code 2") without the fix.
            WhisperCppBackend("tiny", threads=2).transcribe(
                target, language="de", track="mic")
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_stderr_is_drained_without_deadlock(self):
        """Regression H10: stderr was an unread pipe. If this call completes,
        draining is proven to work."""
        backend = WhisperCppBackend("tiny", threads=8, use_vad=False)
        backend.transcribe(SAMPLE, language="de")

    def test_cancel_terminates_process(self):
        import threading
        import time
        backend = WhisperCppBackend("tiny", threads=1, use_vad=False)
        threading.Timer(1.0, backend.cancel).start()
        started = time.monotonic()
        try:
            backend.transcribe(SAMPLE, language="de")
        except TranscriptionError:
            pass
        self.assertLess(time.monotonic() - started, 30.0)


if __name__ == "__main__":
    unittest.main()
