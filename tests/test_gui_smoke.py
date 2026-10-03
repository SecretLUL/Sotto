"""Smoke test for the user interface.

Builds the complete window invisibly, checks that no exception is raised and
shuts it down cleanly. Covers audit finding M7 (the previous version had no
WM_DELETE_WINDOW handler and never terminated PyAudio).
"""

import os
import tempfile
import unittest

try:
    import tkinter as tk
    _HAS_TK = True
except Exception:                                   # pragma: no cover
    _HAS_TK = False


def _can_open_window():
    if not _HAS_TK:
        return False
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestAppLifecycle(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        from audio_transcriber import config, paths
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons
        icons._ICON_CACHE.clear()
        self.tmpdir = tempfile.mkdtemp()
        self._orig_cfg = config.CFG_PATH
        # Never touch the user's real settings (effective since load() and
        # save() look the path up when called)
        config.CFG_PATH = os.path.join(self.tmpdir, "settings.json")
        # ... and never open the real microphone: building the window starts
        # the level monitoring. The hardware test covers real streams.
        self._no_audio = patch.object(AudioEngine, "configure", return_value=[])
        self._no_audio.start()
        self.paths = paths

    def tearDown(self):
        import shutil
        from audio_transcriber import config
        from audio_transcriber.ui import icons
        self._no_audio.stop()
        icons._ICON_CACHE.clear()
        config.CFG_PATH = self._orig_cfg
        shutil.rmtree(self.tmpdir, ignore_errors=True)


    def test_build_and_close(self):
        from audio_transcriber import config
        from audio_transcriber.ui.app import RecorderApp

        root = tk.Tk()
        root.withdraw()                     # build it invisibly
        app = None
        try:
            app = RecorderApp(root)
            root.update()                   # run one event cycle

            self.assertTrue(app.model_combo["values"])
            self.assertTrue(app.lang_combo["values"])
            self.assertEqual(str(app.start_btn["state"]), "normal")
            self.assertEqual(str(app.stop_btn["state"]), "disabled")

            # Collect the settings from the interface
            app._sync_settings_from_ui()
            self.assertIsInstance(app.settings.mic_gain_db, float)
            self.assertIn(app.settings.language,
                          [code for _label, code in config.LANGUAGE_CHOICES])

            # Tick the level meters once
            app._tick()
            root.update()
        finally:
            if app is not None:
                app.on_close()
            else:                                        # pragma: no cover
                root.destroy()

    def test_gain_slider_updates_meters(self):
        """Regression test: gain sliders must dynamically adjust the VU meter levels."""
        from audio_transcriber.ui.app import RecorderApp
        from unittest.mock import PropertyMock, patch

        root = tk.Tk()
        root.withdraw()
        app = None
        try:
            app = RecorderApp(root)
            with patch.object(type(app.engine), 'mic_level', new_callable=PropertyMock) as mock_mic:
                mock_mic.return_value = 0.1  # ~ -20 dB

                # 0 dB Gain -> ~ -20 dB
                app._on_mic_gain(0.0)
                app._tick()
                db_0 = app.mic_meter.db
                self.assertAlmostEqual(db_0, -20.0, delta=1.0)

                # +10 dB Gain -> ~ -10 dB (+10 dB shift)
                app._on_mic_gain(10.0)
                app._tick()
                db_plus_10 = app.mic_meter.db
                self.assertAlmostEqual(db_plus_10, -10.0, delta=1.0)

                # -10 dB Gain -> ~ -30 dB (-10 dB shift)
                app._on_mic_gain(-10.0)
                app._tick()
                db_minus_10 = app.mic_meter.db
                self.assertAlmostEqual(db_minus_10, -30.0, delta=1.0)

                self.assertGreater(db_plus_10, db_0)
                self.assertLess(db_minus_10, db_0)
        finally:
            if app is not None:
                app.on_close()
            else:
                root.destroy()


    def test_output_name_sanitising(self):
        """Path traversal through the file name field must be impossible.

        The results are identical on every platform. os.path.basename() used to
        decide this by host rules, so a backslash path sanitised one way on
        Windows and another on Linux - caught by CI on the first Linux run.
        """
        cases = {
            "..\\..\\windows\\system32\\evil": "evil",
            "my_meeting.wav": "my_meeting",
            "  ": "my_meeting",
            "": "my_meeting",
            "C:/temp/report.wav": "report",
            'in<va>lid:"|?*': "in_va_lid_____",
            # POSIX separators must be blocked on Windows just as well
            "../../etc/passwd": "passwd",
            "/etc/shadow": "shadow",
            "..\\../mix/ed\\name.wav": "name",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.paths.safe_output_name(raw), expected)

    def test_output_name_never_escapes_its_directory(self):
        """The property behind the table above, stated directly."""
        hostile = [
            "../" * 8 + "etc/passwd",
            "..\\" * 8 + "windows\\system32\\cmd.exe",
            "/absolute/path", "C:\\Windows\\System32", "...", "..", ".",
            "con.txt", "  ..  ", "a/b\\c/d",
        ]
        for raw in hostile:
            with self.subTest(raw=raw):
                name = self.paths.safe_output_name(raw)
                self.assertNotIn("/", name)
                self.assertNotIn("\\", name)
                self.assertNotIn("..", name)
                self.assertTrue(name)
                # Must stay inside the directory it is joined onto
                joined = os.path.normpath(os.path.join("/base", name))
                self.assertTrue(joined.replace("\\", "/").startswith("/base/"))


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestOutputConflictDialog(unittest.TestCase):
    """The overwrite warning shown before an existing recording is replaced."""

    def setUp(self):
        from audio_transcriber.ui import icons
        icons._ICON_CACHE.clear()          # PhotoImages belong to their own root
        self.out_dir = tempfile.mkdtemp()
        for name in ("my_meeting.wav", "my_meeting.txt", "taken.wav"):
            with open(os.path.join(self.out_dir, name), "w", encoding="utf-8") as handle:
                handle.write("x")

        from audio_transcriber.ui import theme
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)

    def tearDown(self):
        import shutil
        from audio_transcriber.ui import icons
        self.root.destroy()
        icons._ICON_CACHE.clear()
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _open(self, base_name="my_meeting", extensions=None):
        from audio_transcriber.ui.dialogs import OutputConflictDialog
        return OutputConflictDialog(self.root, self.out_dir, base_name, extensions)

    def test_overwrite_keeps_the_original_name(self):
        dialog = self._open()
        dialog.overwrite_btn.invoke()
        self.assertEqual(dialog.result, "my_meeting")

    def test_numbering_picks_the_next_free_name(self):
        dialog = self._open()
        self.assertEqual(dialog.suggestion, "my_meeting_2")
        dialog.number_btn.invoke()
        self.assertEqual(dialog.result, "my_meeting_2")

    def test_renaming_accepts_a_free_name(self):
        dialog = self._open()
        dialog.entry.delete(0, tk.END)
        dialog.entry.insert(0, "second try.wav")
        dialog.rename_btn.invoke()
        # Sanitised the same way as the main window's file name field
        self.assertEqual(dialog.result, "second try")

    def test_renaming_onto_another_existing_file_stays_open(self):
        dialog = self._open()
        dialog.entry.delete(0, tk.END)
        dialog.entry.insert(0, "taken")
        dialog.rename_btn.invoke()
        try:
            self.assertIsNone(dialog.result)
            self.assertTrue(dialog.winfo_exists())
            # The warning now talks about the new collision
            self.assertIn("taken.wav", dialog.message.cget("text"))
            self.assertEqual(dialog.suggestion, "taken_2")
        finally:
            dialog._cancel()

    def test_cancelling_returns_nothing(self):
        dialog = self._open()
        dialog.cancel_btn.invoke()
        self.assertIsNone(dialog.result)

    def test_kept_raw_tracks_are_part_of_the_conflict_when_they_are_written(self):
        from audio_transcriber import paths
        with open(os.path.join(self.out_dir, "fresh.mic.wav"), "w") as handle:
            handle.write("x")
        dialog = self._open("fresh", paths.output_extensions(keep_raw_tracks=True))
        try:
            self.assertIn("fresh.mic.wav", dialog.message.cget("text"))
            self.assertEqual(dialog.suggestion, "fresh_2")
            self.assertIn("would be replaced", dialog.hint.cget("text"))
        finally:
            dialog._cancel()


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestControlStates(unittest.TestCase):
    """Upload must stay blocked from 'Start' until the run is completely over.

    Regression: only upload_and_transcribe() disabled the button, so it stayed
    live while a recording was being made and, worse, while the finished
    recording was still being processed. A click then started a second
    pipeline next to the first, and self.finalizer pointed at only one of them.
    """

    def setUp(self):
        from unittest.mock import patch
        from audio_transcriber import config
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons
        from audio_transcriber.ui.app import RecorderApp

        icons._ICON_CACHE.clear()
        self.out_dir = tempfile.mkdtemp()
        settings = config.Settings(output_dir=self.out_dir, live_transcribe=False,
                                   live_preview=False)
        self.patches = [
            # Neither the user's real settings nor real audio streams.
            patch.object(config, "load", return_value=(settings, [])),
            patch.object(AudioEngine, "configure", return_value=[]),
        ]
        for patcher in self.patches:
            patcher.start()

        self.root = tk.Tk()
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.app._monitor_thread = None
        self.app.filename_entry.delete(0, tk.END)
        self.app.filename_entry.insert(0, "ui_state_test")

    def tearDown(self):
        import shutil
        from audio_transcriber.ui import icons
        self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        icons._ICON_CACHE.clear()
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _state(self, button):
        return str(button["state"])

    def _begin_a_recording(self):
        """Walk Start through the real code, with the engine itself stubbed."""
        from unittest.mock import patch
        self.app.engine.start_recording = lambda name: None
        with patch.object(self.app, "_current_devices",
                          return_value=(object(), None, "")):
            self.app.start_recording()

    def _stop_the_recording(self):
        from unittest.mock import patch
        from audio_transcriber.audio.capture import RecordingResult, TrackResult
        recording = RecordingResult(
            mic=TrackResult(path="x.raw.wav", rate=16000, frames=16000))
        self.app.engine.stop_recording = lambda: recording
        with patch("audio_transcriber.ui.app.pipeline.Finalizer") as finalizer:
            self.app.stop_recording()
        return finalizer

    def test_upload_is_blocked_from_start_until_processing_is_done(self):
        from unittest.mock import patch
        from audio_transcriber.events import Finished

        self.assertEqual(self._state(self.app.upload_btn), "normal")

        self._begin_a_recording()
        self.assertEqual(self._state(self.app.upload_btn), "disabled", "recording")
        self.assertEqual(self._state(self.app.stop_btn), "normal")

        finalizer = self._stop_the_recording()
        finalizer.return_value.run_async.assert_called_once()
        self.assertEqual(self._state(self.app.upload_btn), "disabled", "processing")
        self.assertEqual(self._state(self.app.start_btn), "disabled", "processing")

        with patch("audio_transcriber.ui.app.messagebox"):
            self.app._on_finished(Finished(text="x", txt_path="x.txt",
                                           audio_path="x.wav"))
        self.assertEqual(self._state(self.app.upload_btn), "normal", "done")
        self.assertEqual(self._state(self.app.start_btn), "normal", "done")
        self.assertEqual(self._state(self.app.stop_btn), "disabled", "done")

    def test_a_failed_run_releases_upload_too(self):
        from unittest.mock import patch
        from audio_transcriber.events import Failed

        self._begin_a_recording()
        self._stop_the_recording()
        self.assertEqual(self._state(self.app.upload_btn), "disabled")

        with patch("audio_transcriber.ui.app.messagebox"):
            self.app._on_failed(Failed(message="boom"))
        self.assertEqual(self._state(self.app.upload_btn), "normal")

    def test_a_cancelled_overwrite_dialog_releases_both_buttons(self):
        from unittest.mock import patch
        with open(os.path.join(self.out_dir, "ui_state_test.wav"), "w") as handle:
            handle.write("x")                          # makes the name collide

        with patch("audio_transcriber.ui.app.dialogs.ask_output_conflict",
                   return_value=None) as ask, \
                patch.object(self.app, "_current_devices",
                             return_value=(object(), None, "")):
            self.app.start_recording()

        ask.assert_called_once()
        self.assertEqual(self._state(self.app.start_btn), "normal")
        self.assertEqual(self._state(self.app.upload_btn), "normal")

    def test_an_engine_that_refuses_to_start_releases_both_buttons(self):
        from unittest.mock import patch

        def refuse(_name):
            raise RuntimeError("Neither audio source is active.")

        self.app.engine.start_recording = refuse
        with patch("audio_transcriber.ui.app.messagebox") as box, \
                patch.object(self.app, "_current_devices",
                             return_value=(object(), None, "")):
            self.app.start_recording()

        box.showerror.assert_called_once()
        self.assertEqual(self._state(self.app.start_btn), "normal")
        self.assertEqual(self._state(self.app.upload_btn), "normal")

    def test_a_source_that_dies_is_reported_while_recording(self):
        """The recording carries on with the other track; without this the flat
        meter was the only sign that the microphone was gone."""
        pending = [[("mic", "Read error on 'USB Mic': device unplugged")]]
        self.app.engine.new_stream_errors = lambda: pending.pop(0) if pending else []
        self.app.engine._recording = True            # as if Start had worked
        try:
            self.app._tick()
            self.app._tick()                         # nothing new the second time
        finally:
            self.app.engine._recording = False

        text = self.app.transcript.text.get("1.0", "end-1c")
        self.assertEqual(text.count("device unplugged"), 1)
        self.assertEqual(self.app.status._text, "recording - microphone stopped")

    def test_the_gpu_switch_reaches_the_settings(self):
        self.assertFalse(self.app.settings.use_gpu)
        self.app.gpu_var.set(True)
        self.app._sync_settings_from_ui()
        self.assertTrue(self.app.settings.use_gpu)

    def test_a_direct_upload_call_cannot_slip_past_the_disabled_button(self):
        from unittest.mock import patch
        self.app.upload_btn.config(state="disabled")
        with patch("audio_transcriber.ui.app.filedialog") as dialog:
            self.app.upload_and_transcribe()
        dialog.askopenfilename.assert_not_called()


def _make_raw_tracks(folder, base, kinds=("mic", "sys"), seconds=2.0, rate=16000):
    """Raw track files the way the capture engine leaves them behind."""
    import numpy as np
    import soundfile as sf
    made = []
    for kind in kinds:
        path = os.path.join(folder, f"{base}.{kind}.raw.wav")
        data = np.random.default_rng(0).normal(0, 0.1, int(seconds * rate))
        sf.write(path, data.astype("float32"), rate, subtype="PCM_16")
        made.append(path)
    return made


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestRecoveryDialog(unittest.TestCase):
    """Process, delete or postpone a recording that never reached a transcript."""

    def setUp(self):
        from audio_transcriber.audio import capture
        from audio_transcriber.ui import icons, theme
        icons._ICON_CACHE.clear()
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)
        self.unfinished = capture.UnfinishedRecording(
            "Team Meeting",
            {"mic": capture.TrackResult(path="x", rate=48000, frames=48000 * 65)},
            modified=1_700_000_000.0)

    def tearDown(self):
        from audio_transcriber.ui import icons
        self.root.destroy()
        icons._ICON_CACHE.clear()

    def _open(self, more=0):
        from audio_transcriber.ui.dialogs import RecoveryDialog
        return RecoveryDialog(self.root, self.unfinished, more)

    def test_the_message_says_what_was_found(self):
        dialog = self._open()
        try:
            text = dialog.message.cget("text")
            self.assertIn("'Team Meeting'", text)
            self.assertIn("01:05", text)
            self.assertIn("microphone", text)
            self.assertNotIn("system audio", text)
        finally:
            dialog._dismiss()

    def test_the_three_answers(self):
        for button, expected in (("process_btn", "process"),
                                 ("delete_btn", "delete"),
                                 ("later_btn", None)):
            with self.subTest(button=button):
                dialog = self._open()
                getattr(dialog, button).invoke()
                self.assertEqual(dialog.result, expected)

    def test_closing_the_window_means_later_never_delete(self):
        dialog = self._open()
        dialog._dismiss()
        self.assertIsNone(dialog.result)

    def test_more_waiting_recordings_are_mentioned(self):
        dialog = self._open(more=2)
        try:
            labels = []

            def collect(widget):
                if isinstance(widget, tk.Label):
                    labels.append(str(widget.cget("text")))
                for child in widget.winfo_children():
                    collect(child)

            collect(dialog)
            self.assertIn("2 more unfinished recordings are waiting",
                          " ".join(labels))
        finally:
            dialog._dismiss()


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestRecoveryInTheApp(unittest.TestCase):
    """An interrupted recording is offered again instead of being lost.

    Closing the window mid-recording, a crash or a failed run leaves the raw
    tracks in the temp folder; nothing used to look at them again.
    """

    def setUp(self):
        import tempfile
        from unittest.mock import patch
        from audio_transcriber import config, paths
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons

        icons._ICON_CACHE.clear()
        self.tmp = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()
        settings = config.Settings(output_dir=self.out_dir, live_transcribe=False,
                                   live_preview=False)
        self.patches = [
            patch.object(config, "load", return_value=(settings, [])),
            patch.object(AudioEngine, "configure", return_value=[]),
            patch.object(paths, "TMP_DIR", self.tmp),
        ]
        for patcher in self.patches:
            patcher.start()
        self.app = None
        self.closed = False

    def tearDown(self):
        import shutil
        from audio_transcriber.ui import icons
        if self.app is not None and not self.closed:
            self.app.engine._recording = False
            self.app._work_thread = None
            self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        icons._ICON_CACHE.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _build_app(self):
        from audio_transcriber.ui.app import RecorderApp
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.app._monitor_thread = None
        return self.app

    def _banner_shown(self):
        return self.app.recovery_banner.winfo_manager() == "pack"

    # -- the banner ------------------------------------------------------
    def test_the_banner_shows_at_start_when_a_recording_was_interrupted(self):
        _make_raw_tracks(self.tmp, "Interrupted meeting")
        self._build_app()
        self.assertTrue(self._banner_shown())
        self.assertIn("Interrupted meeting", self.app.recovery_label.cget("text"))

    def test_there_is_no_banner_when_nothing_is_waiting(self):
        self._build_app()
        self.assertFalse(self._banner_shown())

    # -- the offer -------------------------------------------------------
    def test_processing_hands_the_tracks_to_the_finalizer(self):
        from unittest.mock import MagicMock, patch
        from audio_transcriber.audio.capture import RecordingResult
        made = _make_raw_tracks(self.tmp, "Interrupted meeting")
        self._build_app()
        worker = MagicMock()
        worker.is_alive.return_value = True

        with patch("audio_transcriber.ui.app.dialogs.ask_recovery",
                   return_value="process"):
            with patch("audio_transcriber.ui.app.pipeline.Finalizer") as finalizer:
                finalizer.return_value.run_async.return_value = worker
                self.app.offer_recovery()

        finalizer.return_value.run_async.assert_called_once()
        recording, base_name = finalizer.return_value.run_async.call_args[0]
        self.assertIsInstance(recording, RecordingResult)
        self.assertEqual(base_name, "Interrupted meeting")
        self.assertEqual(sorted([recording.mic.path, recording.sys.path]),
                         sorted(made))
        self.assertIn("Recovered", " ".join(recording.warnings))
        self.assertEqual(str(self.app.upload_btn["state"]), "disabled")
        self.assertEqual(str(self.app.start_btn["state"]), "disabled")
        self.assertFalse(self._banner_shown(), "no banner while it is processed")
        self.assertIs(self.app._work_thread, worker)

    def test_a_failed_run_brings_the_banner_back_and_says_the_tracks_are_kept(self):
        from unittest.mock import patch
        from audio_transcriber.events import Failed
        _make_raw_tracks(self.tmp, "Interrupted meeting")
        self._build_app()

        with patch("audio_transcriber.ui.app.messagebox"):
            self.app._on_failed(Failed(message="No ElevenLabs API key"))

        text = self.app.transcript.text.get("1.0", "end-1c")
        self.assertIn("The recorded tracks were kept", text)
        self.assertTrue(self._banner_shown())

    def test_deleting_asks_first_and_then_removes_the_tracks(self):
        from unittest.mock import patch
        made = _make_raw_tracks(self.tmp, "Interrupted meeting")
        self._build_app()

        with patch("audio_transcriber.ui.app.dialogs.ask_recovery",
                   return_value="delete"):
            with patch("audio_transcriber.ui.app.messagebox") as box:
                box.askyesno.return_value = False
                self.app.offer_recovery()
                self.assertTrue(all(os.path.exists(path) for path in made),
                                "declined: nothing is deleted")

                box.askyesno.return_value = True
                self.app.offer_recovery()

        self.assertFalse(any(os.path.exists(path) for path in made))
        self.assertFalse(self._banner_shown())

    def test_later_changes_nothing(self):
        from unittest.mock import patch
        made = _make_raw_tracks(self.tmp, "Interrupted meeting")
        self._build_app()
        with patch("audio_transcriber.ui.app.dialogs.ask_recovery",
                   return_value=None):
            with patch("audio_transcriber.ui.app.pipeline.Finalizer") as finalizer:
                self.app.offer_recovery()
        finalizer.assert_not_called()
        self.assertTrue(all(os.path.exists(path) for path in made))
        self.assertTrue(self._banner_shown())

    def test_nothing_is_offered_while_a_recording_is_running(self):
        from unittest.mock import patch
        _make_raw_tracks(self.tmp, "Interrupted meeting")
        self._build_app()
        self.app.engine._recording = True
        with patch("audio_transcriber.ui.app.dialogs.ask_recovery") as ask:
            self.app.offer_recovery()
        ask.assert_not_called()

    def test_the_recording_being_made_is_not_taken_for_an_interrupted_one(self):
        from audio_transcriber.audio import capture
        _make_raw_tracks(self.tmp, "live take")
        _make_raw_tracks(self.tmp, "left over")
        self._build_app()
        self.app.engine._recording = True
        self.app.recording_base_name = "live take"
        names = [item.base_name for item in
                 capture.find_unfinished(self.tmp, exclude=self.app._in_use())]
        self.assertEqual(names, ["left over"])

    # -- closing the window ----------------------------------------------
    def test_closing_during_a_recording_asks_first(self):
        from unittest.mock import patch
        self._build_app()
        self.app.engine._recording = True
        with patch("audio_transcriber.ui.app.messagebox") as box:
            box.askyesno.return_value = False
            self.app.on_close()
            self.assertTrue(self.root.winfo_exists(), "declined: stay open")
            self.assertFalse(self.app._shutting_down)
            self.assertIn("recording is running", box.askyesno.call_args[0][1])

            box.askyesno.return_value = True
            self.app.on_close()
        self.closed = True
        self.assertTrue(self.app._shutting_down)

    def test_closing_while_processing_asks_too(self):
        from unittest.mock import MagicMock, patch
        self._build_app()
        worker = MagicMock()
        worker.is_alive.return_value = True
        self.app._work_thread = worker
        with patch("audio_transcriber.ui.app.messagebox") as box:
            box.askyesno.return_value = False
            self.app.on_close()
        box.askyesno.assert_called_once()
        self.assertIn("being processed", box.askyesno.call_args[0][1])
        self.assertTrue(self.root.winfo_exists())

    def test_closing_when_idle_just_closes(self):
        from unittest.mock import patch
        self._build_app()
        with patch("audio_transcriber.ui.app.messagebox") as box:
            self.app.on_close()
        box.askyesno.assert_not_called()
        self.closed = True

    # -- an empty recording ----------------------------------------------
    def test_an_empty_recording_leaves_no_files_to_recover(self):
        from unittest.mock import patch
        import numpy as np
        import soundfile as sf
        from audio_transcriber.audio.capture import RecordingResult, TrackResult
        path = os.path.join(self.tmp, "nothing.mic.raw.wav")
        sf.write(path, np.zeros(0, dtype="float32"), 16000, subtype="PCM_16")
        self._build_app()
        recording = RecordingResult(mic=TrackResult(path=path, rate=16000, frames=0))
        self.app.engine.stop_recording = lambda: recording
        with patch("audio_transcriber.ui.app.messagebox"):
            self.app.stop_recording()
        self.assertFalse(os.path.exists(path))


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestTranscriptProgressLine(unittest.TestCase):
    """Download progress is one line that is updated in place.

    Regression: the line had no newline, so whatever was logged next ran on
    straight after the last percentage ('...20.0 MB/sTranscription started.').
    """

    def setUp(self):
        from audio_transcriber.ui import theme, widgets
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)
        self.transcript = widgets.Transcript(self.root)

    def tearDown(self):
        self.root.destroy()

    def _content(self):
        return self.transcript.text.get("1.0", "end-1c")

    def test_updates_in_place_and_the_next_log_starts_a_new_line(self):
        t = self.transcript
        t.append("Downloading model 'small'…\n")
        for percent in ("  5.0", " 50.0", "100.0"):
            t.replace_last_line(f"small: {percent} %")
        t.append("Transcription started.\n")
        self.assertEqual(
            self._content(),
            "Downloading model 'small'…\nsmall: 100.0 %\nTranscription started.\n")

    def test_never_overwrites_a_half_written_line(self):
        t = self.transcript
        t.append("no newline here")
        t.replace_last_line("progress 1 %")
        t.replace_last_line("progress 2 %")
        self.assertEqual(self._content(), "no newline here\nprogress 2 %")

    def test_two_downloads_in_a_row_keep_their_own_lines(self):
        t = self.transcript
        t.replace_last_line("whisper.cpp: 100.0 %")
        t.append("Extracting archive…\n")
        t.replace_last_line("model: 10.0 %")
        self.assertEqual(
            self._content(),
            "whisper.cpp: 100.0 %\nExtracting archive…\nmodel: 10.0 %")

    def test_clear_forgets_the_progress_line(self):
        t = self.transcript
        t.replace_last_line("model: 10.0 %")
        t.clear()
        t.append("fresh\n")
        self.assertEqual(self._content(), "fresh\n")

    def test_set_transcript_forgets_the_progress_line(self):
        t = self.transcript
        t.replace_last_line("model: 10.0 %")
        t.set_transcript("[00:01] [You]: Hello")
        t.append("next\n")
        self.assertEqual(self._content(), "[00:01] [You]: Hello\nnext\n")


if __name__ == "__main__":
    unittest.main()
