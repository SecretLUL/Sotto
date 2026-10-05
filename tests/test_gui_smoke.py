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
        """Path traversal through the file name field must be impossible."""
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
        from audio_transcriber import config, preflight
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
            # Nor this machine's disk space and downloaded models: a warning
            # would open a real dialog, and these tests are not about that.
            patch.object(preflight, "check", return_value=[]),
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
        from unittest.mock import patch
        from audio_transcriber.ui import icons
        # A test may leave a stand-in worker "alive". Closing then asks whether
        # to quit - in a real dialog, which nobody answers: the whole run hung.
        with patch("audio_transcriber.ui.app.messagebox") as box:
            box.askyesno.return_value = True
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


    # -- a run works on a copy of the settings ---------------------------
    def _change_the_settings_in_the_window(self):
        """What moving to the Settings tab and pressing 'Save settings' does
        to the live object (without writing the file)."""
        self.app.lang_combo.current(7)                    # Turkish
        self.app.model_combo.current(6)                   # large-v3
        self.app.api_entry.insert(0, "another-key")
        self.app._sync_settings_from_ui()

    def test_the_closing_pass_gets_a_copy_not_the_live_settings(self):
        self._begin_a_recording()
        finalizer = self._stop_the_recording()
        given = finalizer.call_args[0][1]
        self.assertIsNot(given, self.app.settings)
        self.assertEqual(given.language, self.app.settings.language)

    def test_changes_made_during_processing_do_not_reach_the_run(self):
        self._begin_a_recording()
        finalizer = self._stop_the_recording()
        given = finalizer.call_args[0][1]
        before = (given.language, given.model, given.api_key)

        self._change_the_settings_in_the_window()

        self.assertNotEqual(self.app.settings.language, before[0], "test needs a change")
        self.assertEqual((given.language, given.model, given.api_key), before)

    def test_what_changes_while_recording_does_not_change_the_language_half_way(self):
        """Live chunks and the closing pass must be recognised the same way."""
        from unittest.mock import patch
        self.app.live_var.set(True)
        with patch("audio_transcriber.ui.app.pipeline.LiveTranscriber") as live:
            self._begin_a_recording()
        live_settings = live.call_args[0][1]
        language = live_settings.language

        self._change_the_settings_in_the_window()
        finalizer = self._stop_the_recording()

        self.assertEqual(live_settings.language, language)
        self.assertIs(finalizer.call_args[0][1], live_settings,
                      "one copy for the whole recording")

    def test_levels_set_while_recording_still_shape_the_mixdown(self):
        """The gain sliders are meant to be moved while listening to the meters;
        the mixdown is made when the recording stops."""
        self._begin_a_recording()
        self.app._on_mic_gain(6.0)
        self.app._on_sys_gain(-4.0)
        finalizer = self._stop_the_recording()
        given = finalizer.call_args[0][1]
        self.assertEqual((given.mic_gain_db, given.loop_gain_db), (6.0, -4.0))

    def test_the_preview_works_on_a_copy_too(self):
        from unittest.mock import patch
        self.app.preview_var.set(True)
        with patch("audio_transcriber.ui.app.pipeline.LivePreview") as preview:
            self._begin_a_recording()
        self.assertIsNot(preview.call_args[0][1], self.app.settings)

    def test_an_upload_gets_a_copy_not_the_live_settings(self):
        from unittest.mock import patch
        with patch("audio_transcriber.ui.app.filedialog") as dialog,                 patch("audio_transcriber.ui.app.pipeline.FileFinalizer") as worker:
            dialog.askopenfilename.return_value = os.path.join(self.out_dir, "in.wav")
            self.app.upload_and_transcribe()
        worker.assert_called_once()
        self.assertIsNot(worker.call_args[0][1], self.app.settings)


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestPreflightInTheApp(unittest.TestCase):
    """Trouble is reported before a run starts, not after the recording."""

    def setUp(self):
        from unittest.mock import patch
        from audio_transcriber import config
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons

        icons._ICON_CACHE.clear()
        self.out_dir = tempfile.mkdtemp()
        settings = config.Settings(output_dir=self.out_dir, live_transcribe=False,
                                   live_preview=False)
        self.patches = [
            patch.object(config, "load", return_value=(settings, [])),
            patch.object(AudioEngine, "configure", return_value=[]),
        ]
        for patcher in self.patches:
            patcher.start()
        from audio_transcriber.ui.app import RecorderApp
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.app._monitor_thread = None
        self.app.filename_entry.delete(0, tk.END)
        self.app.filename_entry.insert(0, "preflight_test")
        self.started = []
        self.closed = False
        self.app.engine.start_recording = lambda name: self.started.append(name)

    def tearDown(self):
        import shutil
        from unittest.mock import patch
        from audio_transcriber.ui import icons
        if not self.closed:
            with patch("audio_transcriber.ui.app.messagebox") as box:
                box.askyesno.return_value = True
                self.app.engine._recording = False
                self.app._work_thread = None
                self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        icons._ICON_CACHE.clear()
        shutil.rmtree(self.out_dir, ignore_errors=True)

    # -- helpers ---------------------------------------------------------
    def _findings(self, *findings):
        from unittest.mock import patch
        from audio_transcriber import preflight
        return patch.object(preflight, "check", return_value=list(findings))

    def _press_start(self, box=None):
        from unittest.mock import patch
        with patch.object(self.app, "_current_devices",
                          return_value=(object(), None, "")):
            if box is None:
                self.app.start_recording()
            else:
                with patch("audio_transcriber.ui.app.messagebox", box):
                    self.app.start_recording()

    def _state(self, button):
        return str(button["state"])

    def _log(self):
        return self.app.transcript.text.get("1.0", "end-1c")

    # -- errors ----------------------------------------------------------
    def test_an_error_stops_the_start_and_says_why(self):
        from unittest.mock import MagicMock
        from audio_transcriber import preflight
        box = MagicMock()
        with self._findings(preflight.Finding(preflight.ERROR, "No API key set.")):
            self._press_start(box)

        box.showerror.assert_called_once()
        self.assertIn("No API key set.", box.showerror.call_args[0][1])
        self.assertEqual(self.started, [], "nothing was recorded")
        self.assertEqual(self._state(self.app.start_btn), "normal")
        self.assertEqual(self._state(self.app.upload_btn), "normal")

    def test_all_errors_are_listed_together(self):
        from unittest.mock import MagicMock
        from audio_transcriber import preflight
        box = MagicMock()
        with self._findings(preflight.Finding(preflight.ERROR, "First problem."),
                            preflight.Finding(preflight.ERROR, "Second problem.")):
            self._press_start(box)
        message = box.showerror.call_args[0][1]
        self.assertIn("First problem.", message)
        self.assertIn("Second problem.", message)

    def test_it_checks_the_settings_as_they_are_in_the_window_now(self):
        """End to end through the real check: cloud chosen, no key typed."""
        from unittest.mock import MagicMock, patch
        self.app.model_combo.current(0)                      # ElevenLabs
        self.app.api_entry.delete(0, tk.END)
        box = MagicMock()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            self._press_start(box)
        box.showerror.assert_called_once()
        self.assertIn("API key", box.showerror.call_args[0][1])
        self.assertEqual(self.started, [])

    # -- warnings --------------------------------------------------------
    def test_a_warning_asks_and_no_means_no(self):
        from unittest.mock import MagicMock
        from audio_transcriber import preflight
        box = MagicMock()
        box.askokcancel.return_value = False
        with self._findings(preflight.Finding(preflight.WARNING, "Only 1 GB free.")):
            self._press_start(box)

        box.askokcancel.assert_called_once()
        self.assertIn("Only 1 GB free.", box.askokcancel.call_args[0][1])
        box.showerror.assert_not_called()
        self.assertEqual(self.started, [])
        self.assertEqual(self._state(self.app.start_btn), "normal")

    def test_a_warning_that_is_accepted_lets_the_recording_start(self):
        from unittest.mock import MagicMock
        from audio_transcriber import preflight
        box = MagicMock()
        box.askokcancel.return_value = True
        with self._findings(preflight.Finding(preflight.WARNING, "Only 1 GB free.")):
            self._press_start(box)
        self.assertEqual(self.started, ["preflight_test"])
        self.assertEqual(self._state(self.app.stop_btn), "normal")

    # -- notes -----------------------------------------------------------
    def test_a_note_is_shown_in_the_log_and_nobody_is_asked(self):
        from unittest.mock import MagicMock
        from audio_transcriber import preflight
        box = MagicMock()
        note = "The model 'large-v3' (3.1 GB) is not on this computer yet."
        with self._findings(preflight.Finding(preflight.NOTE, note)):
            self._press_start(box)
        box.showerror.assert_not_called()
        box.askokcancel.assert_not_called()
        self.assertEqual(self.started, ["preflight_test"])
        self.assertIn(note, self._log())

    def test_a_clean_check_says_nothing_and_just_starts(self):
        from unittest.mock import MagicMock
        box = MagicMock()
        with self._findings():
            self._press_start(box)
        box.showerror.assert_not_called()
        box.askokcancel.assert_not_called()
        self.assertEqual(self.started, ["preflight_test"])

    # -- the model is fetched while recording ----------------------------
    def test_the_model_is_fetched_in_the_background_once_recording_runs(self):
        from unittest.mock import patch
        with self._findings(), \
                patch("audio_transcriber.ui.app.pipeline.ModelPrefetch") as prefetch:
            self._press_start()
        prefetch.assert_called_once()
        self.assertIsNot(prefetch.call_args[0][1], self.app.settings,
                         "it works on the copy the run uses")
        prefetch.return_value.start.assert_called_once()

    def test_nothing_is_fetched_when_the_recording_does_not_start(self):
        from unittest.mock import patch

        def refuse(_name):
            raise RuntimeError("Neither audio source is active.")

        self.app.engine.start_recording = refuse
        with self._findings(), \
                patch("audio_transcriber.ui.app.messagebox"), \
                patch("audio_transcriber.ui.app.pipeline.ModelPrefetch") as prefetch:
            self._press_start()
        prefetch.assert_not_called()

    def test_closing_the_window_cancels_the_fetch(self):
        from unittest.mock import patch
        with self._findings(), \
                patch("audio_transcriber.ui.app.pipeline.ModelPrefetch") as prefetch:
            self._press_start()
        self.app.engine._recording = False       # let tearDown close quietly
        self.app._work_thread = None
        with patch("audio_transcriber.ui.app.messagebox"):
            self.app.on_close()
        self.closed = True
        prefetch.return_value.cancel.assert_called_once()

    # -- uploads ---------------------------------------------------------
    def test_an_upload_is_checked_before_a_file_is_chosen(self):
        from unittest.mock import MagicMock, patch
        from audio_transcriber import preflight
        box = MagicMock()
        with self._findings(preflight.Finding(preflight.ERROR, "No API key set.")) as check, \
                patch("audio_transcriber.ui.app.messagebox", box), \
                patch("audio_transcriber.ui.app.filedialog") as dialog:
            self.app.upload_and_transcribe()

        box.showerror.assert_called_once()
        dialog.askopenfilename.assert_not_called()
        self.assertFalse(check.call_args.kwargs["recording"],
                         "an upload needs no room for a recording")

    def test_the_notes_of_an_upload_are_shown_in_the_log(self):
        from unittest.mock import MagicMock, patch
        from audio_transcriber import preflight
        note = "The model 'medium' (1.5 GB) is not on this computer yet."
        with self._findings(preflight.Finding(preflight.NOTE, note)), \
                patch("audio_transcriber.ui.app.messagebox", MagicMock()), \
                patch("audio_transcriber.ui.app.filedialog") as dialog, \
                patch("audio_transcriber.ui.app.pipeline.FileFinalizer"):
            dialog.askopenfilename.return_value = os.path.join(self.out_dir, "talk.wav")
            self.app.upload_and_transcribe()
        self.assertIn(note, self._log())


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestApiKeyHonesty(unittest.TestCase):
    """What the window says about the key must be true."""

    def setUp(self):
        from unittest.mock import patch
        from audio_transcriber import config
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons

        icons._ICON_CACHE.clear()
        self.out_dir = tempfile.mkdtemp()
        settings = config.Settings(output_dir=self.out_dir, live_transcribe=False,
                                   live_preview=False)
        self.patches = [
            patch.object(config, "load", return_value=(settings, [])),
            patch.object(AudioEngine, "configure", return_value=[]),
        ]
        for patcher in self.patches:
            patcher.start()
        self.app = None

    def tearDown(self):
        import shutil
        from audio_transcriber.ui import icons
        if self.app is not None:
            self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        icons._ICON_CACHE.clear()
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _build(self):
        from audio_transcriber.ui.app import RecorderApp
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.app._monitor_thread = None
        return self.app

    def _labels(self, widget):
        found = []

        def collect(item):
            if isinstance(item, tk.Label):
                found.append(str(item.cget("text")))
            for child in item.winfo_children():
                collect(child)

        collect(widget)
        return " ".join(found)

    def test_the_hint_says_how_the_key_is_kept(self):
        text = self._labels(self._build().key_field)
        self.assertIn("DPAPI", text)

    def test_saving_a_key_that_could_not_be_kept_tells_the_user(self):
        from unittest.mock import patch
        from audio_transcriber import config
        app = self._build()
        note = ("The API key could not be encrypted (CryptProtectData failed) "
                "and was not saved.")
        with patch.object(config, "save", return_value=[note]), \
                patch("audio_transcriber.ui.app.messagebox") as box:
            app.save_settings()

        box.showwarning.assert_called_once()
        self.assertIn(note, box.showwarning.call_args[0][1])
        self.assertNotEqual(app.status._text, "settings saved")

    def test_a_plain_save_is_still_just_a_save(self):
        from unittest.mock import patch
        from audio_transcriber import config
        app = self._build()
        with patch.object(config, "save", return_value=[]), \
                patch("audio_transcriber.ui.app.messagebox") as box:
            app.save_settings()
        box.showwarning.assert_not_called()
        self.assertEqual(app.status._text, "settings saved")


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
        from audio_transcriber import config, paths, preflight
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
            patch.object(preflight, "check", return_value=[]),
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
        self.assertIsNot(finalizer.call_args[0][1], self.app.settings)
        self.assertEqual(str(self.app.upload_btn["state"]), "disabled")
        self.assertEqual(str(self.app.start_btn["state"]), "disabled")
        self.assertFalse(self._banner_shown(), "no banner while it is processed")
        self.assertIs(self.app._work_thread, worker)

    def test_a_recovered_recording_is_checked_first_too(self):
        """No key, no point starting - and the tracks stay where they are."""
        from unittest.mock import patch
        from audio_transcriber import preflight
        made = _make_raw_tracks(self.tmp, "Interrupted meeting")
        self._build_app()
        error = [preflight.Finding(preflight.ERROR, "No API key set.")]

        with patch("audio_transcriber.ui.app.dialogs.ask_recovery",
                   return_value="process"), \
                patch.object(preflight, "check", return_value=error) as check, \
                patch("audio_transcriber.ui.app.messagebox") as box, \
                patch("audio_transcriber.ui.app.pipeline.Finalizer") as finalizer:
            self.app.offer_recovery()

        box.showerror.assert_called_once()
        self.assertFalse(check.call_args.kwargs["recording"])
        finalizer.assert_not_called()
        self.assertTrue(all(os.path.exists(path) for path in made))
        self.assertTrue(self._banner_shown())
        self.assertEqual(str(self.app.start_btn["state"]), "normal")

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


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestSaveButton(unittest.TestCase):
    """'Save settings' is grey once saved and lights up again on a change."""

    def setUp(self):
        from unittest.mock import patch
        from audio_transcriber import config
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons

        icons._ICON_CACHE.clear()
        self.out_dir = tempfile.mkdtemp()
        self.loaded = (config.Settings(output_dir=self.out_dir,
                                       live_transcribe=False,
                                       live_preview=False), [])
        self.patches = [
            patch.object(config, "load", side_effect=lambda: self.loaded),
            patch.object(config, "save", return_value=[]),
            patch.object(AudioEngine, "configure", return_value=[]),
        ]
        self.save = [patcher.start() for patcher in self.patches][1]
        self.app = None

    def tearDown(self):
        import shutil
        from audio_transcriber.ui import icons
        if self.app is not None:
            self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        icons._ICON_CACHE.clear()
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _build(self):
        from audio_transcriber.ui.app import RecorderApp
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.app._monitor_thread = None
        return self.app

    def _state(self):
        self.app._refresh_save_button()
        return str(self.app.save_settings_btn["state"])

    def test_nothing_changed_since_the_start_means_nothing_to_save(self):
        self._build()
        self.assertEqual(self._state(), "disabled")

    def test_a_change_lights_it_up_and_saving_greys_it_again(self):
        app = self._build()
        app.gpu_var.set(not app.gpu_var.get())
        self.assertEqual(self._state(), "normal")
        app.save_settings_btn.invoke()
        self.save.assert_called_once()
        self.assertEqual(self._state(), "disabled")

    def test_changing_it_back_is_no_change(self):
        app = self._build()
        chosen = app.lang_combo.current()
        app.lang_combo.current(chosen + 1)
        self.assertEqual(self._state(), "normal")
        app.lang_combo.current(chosen)
        self.assertEqual(self._state(), "disabled")

    def test_every_kind_of_setting_counts(self):
        app = self._build()
        changes = (
            lambda: app.output_dir_entry.insert(tk.END, "x"),
            lambda: app.filename_entry.insert(tk.END, "x"),
            lambda: app.mic_gain.set(3.0),
            # The key field takes input only while the cloud engine is chosen.
            lambda: (app.model_combo.current(0), app._refresh_key_state()),
            lambda: app.api_entry.insert(0, "key"),
        )
        for change in changes:
            with self.subTest(change=change):
                saved = app._settings_in_window()
                change()
                self.assertEqual(self._state(), "normal")
                app._saved_settings = app._settings_in_window()
                self.assertNotEqual(saved, app._saved_settings)

    def test_a_failed_save_leaves_it_in_colour(self):
        from unittest.mock import patch
        app = self._build()
        app.gpu_var.set(not app.gpu_var.get())
        self.save.side_effect = OSError("disk full")
        with patch("audio_transcriber.ui.app.messagebox"):
            app.save_settings_btn.invoke()
        self.assertEqual(self._state(), "normal")

    def test_a_settings_file_that_had_to_be_corrected_is_worth_saving(self):
        self.loaded = (self.loaded[0], ["Setting 'model_index' was invalid."])
        self._build()
        self.assertEqual(self._state(), "normal")


@unittest.skipUnless(_can_open_window(), "no graphical display available")
class TestModelPanel(unittest.TestCase):
    """The Settings tab says whether the local model is here, and fetches it."""

    def setUp(self):
        from unittest.mock import patch
        from audio_transcriber import config
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.transcribe import binaries
        from audio_transcriber.ui import icons
        from audio_transcriber.ui.app import RecorderApp

        icons._ICON_CACHE.clear()
        self.present = set()
        self.engine = "whisper-cli.exe"
        settings = config.Settings(model="small", live_transcribe=False,
                                   live_preview=False)
        self.patches = [
            patch.object(config, "load", return_value=(settings, [])),
            patch.object(AudioEngine, "configure", return_value=[]),
            # Not this machine's models, nor a real download.
            patch.object(binaries, "model_present",
                         lambda name: name in self.present),
            patch.object(binaries, "find_whisper_executable",
                         lambda *a, **k: self.engine),
            patch("audio_transcriber.ui.app.preflight.no_room_for_models",
                  return_value=None),
            patch("audio_transcriber.ui.app.pipeline.ModelDownload"),
        ]
        self.download = [patcher.start() for patcher in self.patches][-1]
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.app._monitor_thread = None

    def tearDown(self):
        from audio_transcriber.ui import icons
        self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        icons._ICON_CACHE.clear()

    def _choose(self, key):
        from audio_transcriber import config
        self.app.model_combo.current(config.model_index(key))
        self.app.model_combo.event_generate("<<ComboboxSelected>>")
        self.root.update()

    def _shown(self, widget):
        return bool(widget.winfo_manager())

    def _title(self):
        return self.app.model_state_label.cget("text")

    def test_a_missing_model_says_so_and_offers_the_download(self):
        self.app._refresh_model_state()
        self.assertEqual(self._title(), "Not downloaded")
        self.assertIn("'small'", self.app.model_detail.cget("text"))
        self.assertTrue(self._shown(self.app.model_btn))
        self.assertEqual(self.app.model_btn["text"], "Download")

    def test_a_model_that_is_here_says_so_and_needs_no_button(self):
        self.present.add("small")
        self.app._refresh_model_state()
        self.assertEqual(self._title(), "Downloaded")
        self.assertFalse(self._shown(self.app.model_btn))

    def test_the_model_list_marks_what_is_downloaded(self):
        self.present.add("base")
        self.app._refresh_model_state()
        values = self.app.model_combo["values"]
        marked = [value for value in values if "downloaded" in value]
        self.assertEqual(len(marked), 1)
        self.assertTrue(marked[0].startswith("base"))
        self.assertTrue(self.app.model_combo.get().startswith("small"),
                        "the chosen model stays chosen")

    def test_without_whisper_cpp_it_is_not_ready_yet(self):
        self.present.add("small")
        self.engine = None
        self.app._refresh_model_state()
        self.assertEqual(self._title(), "Almost ready")
        self.assertTrue(self._shown(self.app.model_btn))

    def test_the_cloud_has_no_model_to_show(self):
        from audio_transcriber import config
        self._choose(config.CLOUD_MODEL)
        self.assertFalse(self._shown(self.app.model_panel))
        self._choose("tiny")
        self.assertTrue(self._shown(self.app.model_panel))

    def test_download_shows_the_progress_and_ends_ready(self):
        from audio_transcriber.events import ModelFetch, ModelFetchEnded
        self.download.return_value.model_name = "small"
        self.download.return_value.cancelled = False
        self.app.model_btn.invoke()

        self.download.assert_called_once_with(self.app.bridge, "small")
        self.download.return_value.start.assert_called_once()
        self.assertTrue(self._shown(self.app.model_progress))
        self.assertEqual(self.app.model_btn["text"], "Cancel")
        self.assertIsNone(self.app.model_bar.fraction, "connecting: no length yet")

        mb = 1 << 20
        self.app.bridge.post(ModelFetch("small", "Downloading model 'small'…",
                                        done=250 * mb, total=500 * mb,
                                        rate=25 * mb))
        self.app.bridge.drain_now()
        self.assertAlmostEqual(self.app.model_bar.fraction, 0.5)
        self.assertEqual(self.app.model_percent.cget("text"), "50 %")
        self.assertEqual(self.app.model_detail.cget("text"),
                         "250 MB of 500 MB · 25.0 MB/s · about 10 s left")

        self.present.add("small")
        self.app.bridge.post(ModelFetchEnded("small"))
        self.app.bridge.drain_now()
        self.assertEqual(self._title(), "Downloaded")
        self.assertFalse(self._shown(self.app.model_progress))
        self.assertIsNone(self.app._model_download)

    def test_the_button_cancels_a_running_download(self):
        from audio_transcriber.events import ModelFetchEnded
        self.app.model_btn.invoke()
        self.app.model_btn.invoke()
        self.download.return_value.cancel.assert_called_once()
        self.assertEqual(self._title(), "Cancelling…")

        self.app.bridge.post(ModelFetchEnded("small", cancelled=True))
        self.app.bridge.drain_now()
        self.assertEqual(self._title(), "Not downloaded")

    def test_a_failed_download_says_why_and_offers_another_try(self):
        from audio_transcriber.events import ModelFetchEnded
        self.app.model_btn.invoke()
        self.app.bridge.post(ModelFetchEnded("small", error="could not connect"))
        self.app.bridge.drain_now()
        self.assertEqual(self._title(), "Download failed")
        self.assertIn("could not connect", self.app.model_detail.cget("text"))
        self.assertEqual(self.app.model_btn["text"], "Try again")

    def test_no_room_means_no_download(self):
        from unittest.mock import patch
        with patch("audio_transcriber.ui.app.preflight.no_room_for_models",
                   return_value="Not enough free space for the model 'small'"):
            self.app.model_btn.invoke()
        self.download.assert_not_called()
        self.assertEqual(self._title(), "Download failed")

    def test_closing_the_window_cancels_the_download(self):
        self.app.model_btn.invoke()
        self.app.on_close()
        self.download.return_value.cancel.assert_called_once()
        self.app.on_close = lambda: None              # already closed


if __name__ == "__main__":
    unittest.main()
