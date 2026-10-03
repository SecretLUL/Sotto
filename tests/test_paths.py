"""Tests for central path handling: output names, scratch files, data location."""

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


if __name__ == "__main__":
    unittest.main()
