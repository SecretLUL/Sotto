"""The overwrite warning: which files collide and what replaces them.

Regression: a second run under the same file name silently replaced both the
.wav and the .txt of the previous one.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from audio_transcriber import paths


def _touch(directory, *file_names):
    for file_name in file_names:
        with open(os.path.join(directory, file_name), "w", encoding="utf-8") as handle:
            handle.write("x")


class TestExistingOutputs(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_empty_directory_has_no_conflicts(self):
        self.assertEqual(paths.existing_outputs(self.out_dir, "my_meeting"), [])

    def test_missing_directory_has_no_conflicts(self):
        missing = os.path.join(self.out_dir, "does", "not", "exist")
        self.assertEqual(paths.existing_outputs(missing, "my_meeting"), [])

    def test_reports_only_the_files_that_are_there(self):
        _touch(self.out_dir, "my_meeting.wav")
        self.assertEqual(paths.existing_outputs(self.out_dir, "my_meeting"),
                         ["my_meeting.wav"])

    def test_reports_audio_and_transcript(self):
        _touch(self.out_dir, "my_meeting.wav", "my_meeting.txt")
        self.assertEqual(paths.existing_outputs(self.out_dir, "my_meeting"),
                         ["my_meeting.wav", "my_meeting.txt"])

    def test_other_names_do_not_count(self):
        _touch(self.out_dir, "my_meeting_2.wav", "other.txt")
        self.assertEqual(paths.existing_outputs(self.out_dir, "my_meeting"), [])

    def test_a_leftover_transcript_alone_is_a_conflict(self):
        """The .txt survives even when keep_raw_tracks removed the audio."""
        _touch(self.out_dir, "my_meeting.txt")
        self.assertEqual(paths.existing_outputs(self.out_dir, "my_meeting"),
                         ["my_meeting.txt"])


class TestNextFreeName(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_first_suggestion_is_two(self):
        _touch(self.out_dir, "my_meeting.wav")
        self.assertEqual(paths.next_free_name(self.out_dir, "my_meeting"),
                         "my_meeting_2")

    def test_skips_numbers_that_are_taken(self):
        _touch(self.out_dir, "my_meeting.wav", "my_meeting_2.txt",
               "my_meeting_3.wav")
        self.assertEqual(paths.next_free_name(self.out_dir, "my_meeting"),
                         "my_meeting_4")

    def test_a_numbered_name_keeps_counting(self):
        """my_meeting_2 becomes my_meeting_3, never my_meeting_2_2."""
        _touch(self.out_dir, "my_meeting_2.wav")
        self.assertEqual(paths.next_free_name(self.out_dir, "my_meeting_2"),
                         "my_meeting_3")

    def test_digits_without_a_separator_are_part_of_the_name(self):
        _touch(self.out_dir, "meeting2026.wav")
        self.assertEqual(paths.next_free_name(self.out_dir, "meeting2026"),
                         "meeting2026_2")

    def test_the_suggestion_is_always_free(self):
        _touch(self.out_dir, "my_meeting.wav", "my_meeting_2.wav",
               "my_meeting_3.txt")
        suggestion = paths.next_free_name(self.out_dir, "my_meeting")
        self.assertEqual(paths.existing_outputs(self.out_dir, suggestion), [])


class TestConflictResolutionInTheApp(unittest.TestCase):
    """RecorderApp._resolve_output_conflict without building a window."""

    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        self.app = MagicMock()
        self.app.settings.get_output_dir.return_value = self.out_dir

    def tearDown(self):
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _resolve(self, base_name, extensions=None):
        from audio_transcriber.ui.app import RecorderApp
        return RecorderApp._resolve_output_conflict(self.app, base_name,
                                                    extensions)

    def test_a_free_name_never_opens_the_dialog(self):
        with patch("audio_transcriber.ui.app.dialogs.ask_output_conflict") as ask:
            self.assertEqual(self._resolve("my_meeting"), "my_meeting")
        ask.assert_not_called()

    def test_an_existing_name_asks_and_returns_the_answer(self):
        _touch(self.out_dir, "my_meeting.wav")
        with patch("audio_transcriber.ui.app.dialogs.ask_output_conflict",
                   return_value="my_meeting_2") as ask:
            self.assertEqual(self._resolve("my_meeting"), "my_meeting_2")
        ask.assert_called_once()
        self.assertEqual(ask.call_args[0][1:3], (self.out_dir, "my_meeting"))

    def test_cancelling_aborts_the_run(self):
        _touch(self.out_dir, "my_meeting.txt")
        with patch("audio_transcriber.ui.app.dialogs.ask_output_conflict",
                   return_value=None):
            self.assertIsNone(self._resolve("my_meeting"))

    def test_a_kept_raw_track_counts_as_a_conflict_for_a_recording(self):
        """With 'Keep raw tracks' a recording also writes name.mic.wav and
        name.sys.wav, so an old one of those must trigger the question - while
        an upload, which never writes them, is not bothered by it."""
        _touch(self.out_dir, "my_meeting.mic.wav")
        recording = paths.output_extensions(keep_raw_tracks=True)
        with patch("audio_transcriber.ui.app.dialogs.ask_output_conflict",
                   return_value="my_meeting_2") as ask:
            self.assertEqual(self._resolve("my_meeting", recording), "my_meeting_2")
        ask.assert_called_once()
        self.assertEqual(ask.call_args[0][3], recording)

        with patch("audio_transcriber.ui.app.dialogs.ask_output_conflict") as ask:
            self.assertEqual(self._resolve("my_meeting"), "my_meeting")   # upload
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
