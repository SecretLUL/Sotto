"""Tests for the timestamp parser (audit findings H7 and N4)."""

import re
import unittest

from audio_transcriber.transcribe.base import (format_clock, format_timestamp,
                                               parse_line, parse_timestamp)

# Regex of the previous version - kept here as the regression reference.
OLD_RE = re.compile(
    r'^\s*\[(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\s*-->\s*'
    r'(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\]\s*(.*)$')


class TestParsing(unittest.TestCase):
    def test_whisper_default_format(self):
        line = "[00:00:03.480 --> 00:00:06.120]   Good morning everyone."
        start, end, text = parse_line(line)
        self.assertAlmostEqual(start, 3.480, places=3)
        self.assertAlmostEqual(end, 6.120, places=3)
        self.assertEqual(text, "Good morning everyone.")

    def test_milliseconds_are_kept(self):
        """Regression H7: the previous version discarded the fractional part,
        so the analysis window for speaker attribution could be off by up to a
        full second."""
        line = "[00:00:03.900 --> 00:00:04.100] Yes."
        start, end, _ = parse_line(line)
        self.assertAlmostEqual(start, 3.9, places=3)
        self.assertAlmostEqual(end, 4.1, places=3)

        old = OLD_RE.match(line.strip())
        old_start = int(old.group(1)) * 3600 + int(old.group(2)) * 60 + int(old.group(3))
        self.assertEqual(old_start, 3)                  # truncated
        self.assertGreater(start - old_start, 0.8)      # almost a second of error

    def test_hours(self):
        start, end, _ = parse_line("[01:23:45.000 --> 01:23:47.500] Text")
        self.assertAlmostEqual(start, 5025.0, places=3)
        self.assertAlmostEqual(end, 5027.5, places=3)

    def test_short_mm_ss_format(self):
        start, end, _ = parse_line("[02:15 --> 02:19] Short form")
        self.assertAlmostEqual(start, 135.0)
        self.assertAlmostEqual(end, 139.0)

    def test_non_matching_lines(self):
        for line in ("whisper_model_load: loading model", "", "   ",
                     "system_info: n_threads = 4"):
            self.assertIsNone(parse_line(line))

    def test_parse_timestamp_units(self):
        self.assertEqual(parse_timestamp("0", "30", None, None), 30.0)
        self.assertEqual(parse_timestamp("1", "30", None, None), 90.0)
        self.assertEqual(parse_timestamp("1", "00", "00", None), 3600.0)


class TestFormatting(unittest.TestCase):
    def test_minutes(self):
        self.assertEqual(format_timestamp(0), "[00:00]")
        self.assertEqual(format_timestamp(75.9), "[01:15]")
        self.assertEqual(format_timestamp(3599), "[59:59]")

    def test_hours_regression(self):
        """Regression N4: the previous version formatted 90 minutes as [90:12]."""
        self.assertEqual(format_timestamp(5412), "[01:30:12]")
        self.assertEqual(format_timestamp(7200), "[02:00:00]")

    def test_negative_is_clamped(self):
        self.assertEqual(format_timestamp(-5), "[00:00]")


class TestFormatClock(unittest.TestCase):
    """The same clock as format_timestamp(), for running text instead of a line."""

    def test_minutes_and_hours(self):
        self.assertEqual(format_clock(0), "00:00")
        self.assertEqual(format_clock(65), "01:05")
        self.assertEqual(format_clock(3725), "01:02:05")

    def test_negative_is_clamped(self):
        self.assertEqual(format_clock(-5), "00:00")


if __name__ == "__main__":
    unittest.main()
