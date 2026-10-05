"""The entry point: what main() does, and in which order."""

import os
import subprocess
import sys
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs main() with the window, the app and the migration replaced by stubs that
# only note when they were called.
SCRIPT = textwrap.dedent("""
    from unittest.mock import patch

    order = []

    class FakeApp:
        def offer_recovery(self):
            order.append("offer_recovery")

        def start_update_check(self):
            order.append("start_update_check")

    class FakeRoot:
        def after(self, _milliseconds, callback):
            order.append("after")
            callback()

        def mainloop(self):
            order.append("mainloop")

    patchers = [
        patch("tkinter.Tk", side_effect=lambda: order.append("Tk") or FakeRoot()),
        patch("audio_transcriber.ui.chrome.dark_title_bar",
              side_effect=lambda root: order.append("dark_title_bar")),
        patch("audio_transcriber.ui.app.RecorderApp",
              side_effect=lambda root: order.append("RecorderApp") or FakeApp()),
        patch("audio_transcriber.paths.migrate_legacy_data",
              side_effect=lambda: order.append("migrate") or []),
    ]
    import main
    for patcher in patchers:
        patcher.start()
    code = main.main()
    print(code, ",".join(order))
""")


class TestEntryPoint(unittest.TestCase):
    def test_legacy_data_is_migrated_before_the_window_exists(self):
        """Settings are read while the app is built, so whatever an older packaged
        build left in _internal has to be moved before that - not after.

        A subprocess: on Windows main() switches the process to high-DPI mode,
        which must not leak into the GUI tests that share this process.
        """
        result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        # The title bar is arranged before anything is built into the window,
        # which keeps it invisible until its frame is dark (ui/chrome.py).
        self.assertEqual(
            result.stdout.strip(),
            "0 migrate,Tk,dark_title_bar,RecorderApp,after,offer_recovery,"
            "after,start_update_check,mainloop")

    def test_the_installer_of_an_update_opens_no_window(self):
        """Started with --apply-update, the new version only installs itself:
        no window, no migration, no settings - and its exit code is main()'s."""
        script = textwrap.dedent("""
            import sys
            from unittest.mock import patch
            calls = []
            sys.argv = ["Sotto", "--apply-update", "--source", "s", "--target", "t"]
            with patch("tkinter.Tk", side_effect=AssertionError("a window")), \
                 patch("audio_transcriber.paths.migrate_legacy_data",
                       side_effect=AssertionError("a migration")), \
                 patch("audio_transcriber.update.apply_from_command_line",
                       side_effect=lambda argv: calls.append(argv) or 7):
                import main
                code = main.main()
            print(code, calls)
        """)
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(),
                         "7 [['--apply-update', '--source', 's', '--target', 't']]")


if __name__ == "__main__":
    unittest.main()
