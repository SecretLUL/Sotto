"""Tests for central path handling: output names, scratch files, data location."""

import os
import shutil
import tempfile
import unittest

from audio_transcriber import paths


class TestOutputNameKeepsDots(unittest.TestCase):
    """Regression: os.path.splitext() removed ANY trailing '.something'.

    'Meeting 03.10.2026' became 'Meeting 03.10' and 'v1.2 review' became 'v1'.
    """

    def test_dots_inside_a_name_survive(self):
        cases = {
            "Meeting 03.10.2026": "Meeting 03.10.2026",
            "Q3.planning": "Q3.planning",
            "v1.2 review": "v1.2 review",
            "notes.final": "notes.final",
            "Team Meeting 2026.03.10": "Team Meeting 2026.03.10",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(paths.safe_output_name(raw), expected)

    def test_known_extensions_are_removed_case_insensitively(self):
        cases = {
            "report.wav": "report",
            "REPORT.WAV": "REPORT",
            "talk.Mp3": "talk",
            "a.b.c.txt": "a.b.c",
            "memo.m4a": "memo",
            "take.opus": "take",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(paths.safe_output_name(raw), expected)

    def test_only_one_extension_is_removed(self):
        self.assertEqual(paths.safe_output_name("report.wav.mp3"), "report.wav")

    def test_a_bare_extension_is_a_name_not_an_empty_one(self):
        # splitext() calls '.wav' a file without extension; so did the old code.
        self.assertEqual(paths.safe_output_name(".wav"), "wav")

    def test_directories_never_leak_into_the_name(self):
        self.assertEqual(
            paths.safe_output_name("C:\\Users\\me\\Meeting 03.10.2026.wav"),
            "Meeting 03.10.2026")


class TestReservedDeviceNames(unittest.TestCase):
    """'NUL.txt' is the NUL device according to Microsoft - not a file."""

    def _first_segment(self, name):
        return name.partition(".")[0].rstrip().upper()

    def test_device_names_are_defused_but_stay_recognisable(self):
        for raw in ("nul", "CON", "aux.txt", "COM1", "lpt9", "con.backup",
                    "Nul .wav", "COM\u00b2"):
            with self.subTest(raw=raw):
                name = paths.safe_output_name(raw)
                self.assertNotIn(self._first_segment(name), paths._RESERVED_NAMES)
                self.assertTrue(name.lower().startswith(raw.lower()[:3]), name)

    def test_lookalikes_are_left_alone(self):
        for raw in ("console", "nullable", "com10", "auxiliary", "lpt", "com"):
            with self.subTest(raw=raw):
                self.assertEqual(paths.safe_output_name(raw), raw)


class TestScratchName(unittest.TestCase):
    """whisper-cli cannot open files named in Turkish or Arabic (ANSI argv)."""

    def test_is_ascii_and_keeps_the_suffix(self):
        name = paths.scratch_name("mic", suffix=".asr.wav")
        self.assertTrue(name.isascii())
        self.assertRegex(name, r"^mic-[0-9a-f]{10}\.asr\.wav$")

    def test_labels_are_joined(self):
        self.assertRegex(paths.scratch_name("mic", "live", "3"),
                         r"^mic-live-3-[0-9a-f]{10}\.wav$")
        self.assertRegex(paths.scratch_name(), r"^[0-9a-f]{10}\.wav$")

    def test_nothing_a_user_could_type_survives(self):
        name = paths.scratch_name("\u00fc/..\\toplant\u0131", "")
        self.assertTrue(name.isascii())
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)
        self.assertRegex(name, r"^toplant-[0-9a-f]{10}\.wav$")

    def test_two_calls_never_collide(self):
        names = {paths.scratch_name("mic") for _ in range(200)}
        self.assertEqual(len(names), 200)


class TestAnsiSafePath(unittest.TestCase):
    """Directories are not ours to name: 'C:\\Users\\Sirin' may be Turkish."""

    TURKISH = "C:\\Users\\S\u0131rin\\model.bin"

    @staticmethod
    def _never(_path):
        raise AssertionError("the short form must not be asked for")

    def test_ascii_paths_are_left_alone(self):
        self.assertEqual(
            paths.ansi_safe_path("C:\\bin\\model.bin", windows=True,
                                 short_path=self._never),
            "C:\\bin\\model.bin")

    def test_other_platforms_pass_everything_through(self):
        self.assertEqual(
            paths.ansi_safe_path(self.TURKISH, windows=False,
                                 short_path=self._never),
            self.TURKISH)

    def test_uses_the_short_form_when_there_is_one(self):
        self.assertEqual(
            paths.ansi_safe_path(self.TURKISH, windows=True,
                                 short_path=lambda _p: "C:\\Users\\SIRIN~1\\model.bin"),
            "C:\\Users\\SIRIN~1\\model.bin")

    def test_falls_back_when_there_is_no_usable_short_form(self):
        def broken(_path):
            raise OSError("no such file")

        for short_path in (lambda _p: None, lambda _p: "", broken,
                           lambda _p: self.TURKISH):          # still not ASCII
            with self.subTest(short_path=short_path):
                self.assertEqual(
                    paths.ansi_safe_path(self.TURKISH, windows=True,
                                         short_path=short_path),
                    self.TURKISH)


@unittest.skipUnless(os.name == "nt", "8.3 short names are a Windows feature")
class TestRealShortPath(unittest.TestCase):
    def test_a_non_ascii_location_becomes_ascii_and_stays_the_same_file(self):
        base = tempfile.mkdtemp()
        try:
            folder = os.path.join(base, "S\u0131rin_\u015firket")
            os.makedirs(folder)
            target = os.path.join(folder, "toplant\u0131_\u0627\u062c.bin")
            with open(target, "wb") as handle:
                handle.write(b"x")

            result = paths.ansi_safe_path(target)
            if not result.isascii():
                self.skipTest("this volume has no 8.3 short names")
            self.assertTrue(os.path.samefile(result, target))
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
