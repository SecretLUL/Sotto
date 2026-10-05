"""Checks before a recording or an upload starts.

All of these used to come to light only after the recording: no API key for the
cloud engine, a 3 GB model that still had to be downloaded, an output folder
that cannot be written, a disk that filled up half way through a meeting.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from audio_transcriber import config, paths, preflight
from audio_transcriber.transcribe import binaries

GB = 10 ** 9
GIB = 2 ** 30

_REAL_FREE_BYTES = preflight._free_bytes       # before any test patches it


def texts(findings, level):
    return [finding.text for finding in findings if finding.level == level]


class PreflightCase(unittest.TestCase):
    """Everything is in order unless a test says otherwise."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.out = os.path.join(self.dir, "out")
        self.settings = config.Settings(model="small", output_dir=self.out,
                                        live_transcribe=False, live_preview=False)
        self.free = {}                     # folder -> free bytes; default: plenty
        self.present = {"small", "tiny", "base", "medium", "large-v3",
                        "large-v3-turbo"}

        def free_bytes(folder):
            for prefix, value in self.free.items():
                if os.path.normcase(folder).startswith(os.path.normcase(prefix)):
                    return value
            return 500 * GB

        for target, replacement in (
                (preflight, ("_free_bytes", free_bytes)),
                (binaries, ("model_present", lambda name: name in self.present))):
            patcher = patch.object(target, replacement[0], replacement[1])
            patcher.start()
            self.addCleanup(patcher.stop)

    def check(self, recording=True, **changes):
        for key, value in changes.items():
            setattr(self.settings, key, value)
        return preflight.check(self.settings, recording=recording)


class TestEngine(PreflightCase):
    def test_when_everything_is_in_order_there_is_nothing_to_say(self):
        self.assertEqual(self.check(), [])

    def test_the_cloud_without_a_key_is_an_error(self):
        errors = texts(self.check(model=config.CLOUD_MODEL, api_key=""),
                       preflight.ERROR)
        self.assertEqual(len(errors), 1)
        for fragment in ("API key", "Settings", "ELEVENLABS_API_KEY", "local model"):
            self.assertIn(fragment, errors[0])

    def test_a_blank_key_is_no_key(self):
        findings = self.check(model=config.CLOUD_MODEL, api_key="   ")
        self.assertEqual(len(texts(findings, preflight.ERROR)), 1)

    def test_the_cloud_with_a_key_needs_no_model(self):
        self.present.clear()
        self.assertEqual(self.check(model=config.CLOUD_MODEL, api_key="sk_x"), [])

    def test_the_key_is_not_asked_for_when_a_local_model_is_chosen(self):
        self.assertEqual(self.check(api_key=""), [])


class TestModels(PreflightCase):
    def test_a_model_that_is_not_there_yet_is_a_note_with_its_size(self):
        self.present.discard("large-v3")
        notes = texts(self.check(model="large-v3"), preflight.NOTE)
        self.assertEqual(len(notes), 1)
        self.assertIn("large-v3", notes[0])
        self.assertIn("3.1 GB", notes[0])
        self.assertIn("while you record", notes[0])

    def test_small_sizes_are_given_in_megabytes(self):
        self.present.discard("tiny")
        note = texts(self.check(model="tiny"), preflight.NOTE)[0]
        self.assertIn("78 MB", note)

    def test_the_model_for_the_live_text_is_mentioned_when_that_runs(self):
        self.present -= {"large-v3", "small"}
        notes = texts(self.check(model="large-v3", live_transcribe=True),
                      preflight.NOTE)
        self.assertEqual(len(notes), 2)
        self.assertTrue(any("'large-v3'" in note for note in notes))
        self.assertTrue(any("'small'" in note for note in notes))

    def test_the_preview_needs_the_small_model_too(self):
        self.present.discard("small")
        notes = texts(self.check(model="large-v3", live_transcribe=False,
                                 live_preview=True), preflight.NOTE)
        self.assertEqual(len(notes), 1)
        self.assertIn("'small'", notes[0])

    def test_a_cloud_run_with_the_preview_on_still_needs_the_small_model(self):
        """The read-along is local even when the transcript comes from the cloud."""
        self.present.discard("small")
        findings = self.check(model=config.CLOUD_MODEL, api_key="sk_x",
                              live_preview=True)
        self.assertEqual(texts(findings, preflight.ERROR), [])
        notes = texts(findings, preflight.NOTE)
        self.assertEqual(len(notes), 1)
        self.assertIn("'small'", notes[0])

    def test_live_text_and_preview_together_count_the_small_model_once(self):
        self.present -= {"large-v3", "small"}
        notes = texts(self.check(model="large-v3", live_transcribe=True,
                                 live_preview=True), preflight.NOTE)
        self.assertEqual(len(notes), 2)

    def test_without_live_transcription_only_the_final_model_counts(self):
        self.present -= {"large-v3", "small"}
        notes = texts(self.check(model="large-v3", live_transcribe=False),
                      preflight.NOTE)
        self.assertEqual(len(notes), 1)
        self.assertIn("'large-v3'", notes[0])

    def test_the_same_model_is_not_mentioned_twice(self):
        self.present.discard("small")
        notes = texts(self.check(model="small", live_transcribe=True), preflight.NOTE)
        self.assertEqual(len(notes), 1)

    def test_an_upload_says_the_download_comes_first(self):
        self.present.discard("medium")
        note = texts(self.check(recording=False, model="medium"), preflight.NOTE)[0]
        self.assertIn("downloaded first", note)
        self.assertNotIn("while you record", note)

    def test_an_upload_has_no_live_model(self):
        self.present -= {"large-v3", "small"}
        notes = texts(self.check(recording=False, model="large-v3",
                                 live_transcribe=True), preflight.NOTE)
        self.assertEqual(len(notes), 1)

    def test_not_enough_room_for_the_download_is_an_error(self):
        self.present.discard("large-v3")
        self.free[paths.BIN_DIR] = 2 * GB
        errors = texts(self.check(model="large-v3"), preflight.ERROR)
        self.assertEqual(len(errors), 1)
        self.assertIn("3.1 GB", errors[0])
        self.assertIn("2.0 GB", errors[0])
        self.assertEqual(texts(self.check(model="large-v3"), preflight.NOTE), [],
                         "an impossible download is an error, not also a note")

    def test_the_settings_tab_asks_the_same_question_before_a_download(self):
        self.free[paths.BIN_DIR] = 2 * GB
        problem = preflight.no_room_for_models(["large-v3"])
        self.assertIn("3.1 GB", problem)
        self.assertIn("2.0 GB", problem)
        self.assertIsNone(preflight.no_room_for_models(["small"]))

    def test_every_model_the_app_offers_has_a_known_size(self):
        for name in (name for _label, name in config.MODEL_CHOICES if name):
            with self.subTest(model=name):
                self.assertGreater(binaries.MODEL_SIZE_BYTES[name], 50 * 10 ** 6)


class TestOutputFolder(PreflightCase):
    def test_a_folder_that_cannot_be_written_is_an_error(self):
        with patch.object(paths, "is_writable", return_value=False):
            errors = texts(self.check(), preflight.ERROR)
        self.assertEqual(len(errors), 1)
        self.assertIn(self.out, errors[0])
        self.assertIn("settings", errors[0])

    def test_a_folder_that_cannot_even_be_created_is_an_error(self):
        blocker = os.path.join(self.dir, "a file")
        with open(blocker, "w") as handle:
            handle.write("x")
        errors = texts(self.check(output_dir=os.path.join(blocker, "inside")),
                       preflight.ERROR)
        self.assertEqual(len(errors), 1)
        self.assertIn("cannot be created", errors[0])

    def test_a_folder_that_does_not_exist_yet_is_fine_and_gets_made(self):
        self.assertEqual(self.check(), [])
        self.assertTrue(os.path.isdir(self.out))

    def test_uploads_need_a_writable_folder_too(self):
        with patch.object(paths, "is_writable", return_value=False):
            self.assertEqual(len(texts(self.check(recording=False), preflight.ERROR)), 1)


class TestDiskSpace(PreflightCase):
    def test_plenty_of_room_says_nothing(self):
        self.free[paths.TMP_DIR] = 200 * GB
        self.assertEqual(self.check(), [])

    def test_little_room_is_a_warning_that_says_how_long_it_lasts(self):
        self.free[paths.TMP_DIR] = int(1.5 * GIB)       # 1.5 / 0.9 per hour
        warnings = texts(self.check(), preflight.WARNING)
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 h 40 min", warnings[0])

    def test_less_than_an_hour_is_given_in_minutes(self):
        self.free[paths.TMP_DIR] = int(0.45 * GIB)      # half an hour
        warnings = texts(self.check(), preflight.WARNING)
        self.assertIn("30 min", warnings[0])
        self.assertNotIn(" h ", warnings[0])

    def test_hardly_any_room_is_an_error(self):
        self.free[paths.TMP_DIR] = 200 * 10 ** 6
        errors = texts(self.check(), preflight.ERROR)
        self.assertEqual(len(errors), 1)
        self.assertIn("200 MB", errors[0])
        self.assertEqual(texts(self.check(), preflight.WARNING), [],
                         "not both an error and a warning for the same thing")

    def test_the_tighter_of_the_two_folders_decides(self):
        self.free[paths.TMP_DIR] = 200 * GB
        self.free[self.out] = int(1.0 * GIB)
        warnings = texts(self.check(), preflight.WARNING)
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 h 7 min", warnings[0])

    def test_an_upload_is_not_a_recording(self):
        self.free[paths.TMP_DIR] = 200 * 10 ** 6
        self.assertEqual(self.check(recording=False), [])

    def test_free_space_that_cannot_be_read_is_no_finding(self):
        with patch.object(preflight, "_free_bytes", return_value=None):
            self.assertEqual(self.check(), [])


class TestOrderAndHelpers(PreflightCase):
    def test_errors_come_before_warnings_before_notes(self):
        self.present.discard("large-v3")
        self.free[paths.TMP_DIR] = int(1.5 * GIB)
        with patch.object(paths, "is_writable", return_value=False):
            findings = self.check(model="large-v3")
        levels = [finding.level for finding in findings]
        self.assertEqual(levels, sorted(levels, key=[preflight.ERROR, preflight.WARNING,
                                                      preflight.NOTE].index))
        self.assertEqual(set(levels), {preflight.ERROR, preflight.WARNING, preflight.NOTE})

    def test_the_real_free_space_function_works_for_a_folder_not_made_yet(self):
        value = _REAL_FREE_BYTES(os.path.join(self.dir, "not", "there", "yet"))
        self.assertIsInstance(value, int)
        self.assertGreater(value, 0)

    def test_the_real_function_gives_up_quietly_when_it_cannot_ask(self):
        with patch("shutil.disk_usage", side_effect=OSError("no such volume")):
            self.assertIsNone(_REAL_FREE_BYTES(self.dir))


if __name__ == "__main__":
    unittest.main()
