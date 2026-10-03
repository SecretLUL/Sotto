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
        from audio_transcriber import config, paths
        from audio_transcriber.ui import icons
        icons._ICON_CACHE.clear()
        self.tmpdir = tempfile.mkdtemp()
        self._orig_cfg = config.CFG_PATH
        # Never touch the user's real settings
        config.CFG_PATH = os.path.join(self.tmpdir, "settings.json")
        self.paths = paths

    def tearDown(self):
        import shutil
        from audio_transcriber import config
        from audio_transcriber.ui import icons
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

    def _open(self, base_name="my_meeting"):
        from audio_transcriber.ui.dialogs import OutputConflictDialog
        return OutputConflictDialog(self.root, self.out_dir, base_name)

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


if __name__ == "__main__":
    unittest.main()
