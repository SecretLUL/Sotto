"""Tests for central path handling: output names, scratch files, data location."""

import os
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

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


class TestRawTrackNames(unittest.TestCase):
    """What a recording writes, so the overwrite check knows about all of it."""

    def test_an_upload_writes_two_files_a_recording_with_raw_tracks_four(self):
        self.assertEqual(paths.output_extensions(), (".wav", ".txt"))
        self.assertEqual(paths.output_extensions(keep_raw_tracks=True),
                         (".wav", ".txt", ".mic.wav", ".sys.wav"))

    def test_raw_tracks_sit_next_to_the_transcript(self):
        self.assertEqual(
            paths.raw_track_path("out", "Meeting 03.10.2026", "sys"),
            os.path.join("out", "Meeting 03.10.2026.sys.wav"))

    def test_a_free_number_has_to_clear_the_raw_tracks_too(self):
        folder = tempfile.mkdtemp()
        try:
            for name in ("my_meeting.wav", "my_meeting_2.mic.wav"):
                with open(os.path.join(folder, name), "wb"):
                    pass
            self.assertEqual(paths.next_free_name(folder, "my_meeting"),
                             "my_meeting_2")
            self.assertEqual(
                paths.next_free_name(folder, "my_meeting",
                                     paths.output_extensions(True)),
                "my_meeting_3")
        finally:
            shutil.rmtree(folder, ignore_errors=True)


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
            paths.ansi_safe_path("C:\\bin\\model.bin", short_path=self._never),
            "C:\\bin\\model.bin")

    def test_uses_the_short_form_when_there_is_one(self):
        self.assertEqual(
            paths.ansi_safe_path(self.TURKISH,
                                 short_path=lambda _p: "C:\\Users\\SIRIN~1\\model.bin"),
            "C:\\Users\\SIRIN~1\\model.bin")

    def test_falls_back_when_there_is_no_usable_short_form(self):
        def broken(_path):
            raise OSError("no such file")

        for short_path in (lambda _p: None, lambda _p: "", broken,
                           lambda _p: self.TURKISH):          # still not ASCII
            with self.subTest(short_path=short_path):
                self.assertEqual(
                    paths.ansi_safe_path(self.TURKISH, short_path=short_path),
                    self.TURKISH)


class TestResolveDataDir(unittest.TestCase):
    """A packaged build must not hang its data off __file__.

    In PyInstaller's onedir layout that points into _internal, so the models,
    recordings and settings landed there - gone with every update of the app
    folder, and not writable wherever the app is installed read-only.
    """

    SOURCE = os.path.abspath(os.path.join("checkout", "Audio-Transcriber"))
    APP = os.path.abspath(os.path.join("apps", "Sotto"))
    INTERNAL = os.path.join(APP, "_internal")
    EXE = os.path.join(APP, "Sotto.exe")
    HOME = os.path.abspath(os.path.join("users", "me"))

    def _resolve(self, **overrides):
        options = dict(source_dir=self.SOURCE, frozen=False, executable=self.EXE,
                       env={}, home=self.HOME,
                       writable=lambda _directory: True)
        options.update(overrides)
        return paths.resolve_data_dir(**options)

    def test_from_source_it_is_the_repository(self):
        self.assertEqual(self._resolve(), self.SOURCE)

    def test_a_packaged_build_uses_the_folder_of_the_executable(self):
        self.assertEqual(self._resolve(frozen=True, source_dir=self.INTERNAL),
                         self.APP)

    def test_that_folder_is_probed_not_assumed(self):
        probed = []

        def writable(directory):
            probed.append(directory)
            return True

        self._resolve(frozen=True, source_dir=self.INTERNAL, writable=writable)
        self.assertEqual(probed, [self.APP])

    def test_an_unwritable_folder_falls_back_to_the_profile(self):
        local = os.path.join(self.HOME, "AppData", "Local")
        self.assertEqual(
            self._resolve(frozen=True, writable=lambda _d: False,
                          env={"LOCALAPPDATA": local}),
            os.path.join(local, "AudioTranscriber"))
        # ... also when the variables are missing
        self.assertEqual(
            self._resolve(frozen=True, writable=lambda _d: False),
            os.path.join(self.HOME, "AppData", "Local", "AudioTranscriber"))

    def test_the_environment_variable_wins_everywhere(self):
        chosen = os.path.abspath(os.path.join("somewhere", "else"))
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                self.assertEqual(
                    self._resolve(frozen=frozen,
                                  env={paths.HOME_ENV: f"  {chosen}  "}),
                    chosen)

    def test_a_blank_variable_is_ignored(self):
        self.assertEqual(self._resolve(env={paths.HOME_ENV: "   "}), self.SOURCE)

    def test_the_data_paths_all_hang_off_the_data_directory(self):
        for path in (paths.BIN_DIR, paths.OUT_DIR, paths.CFG_PATH, paths.LOG_PATH):
            self.assertEqual(os.path.dirname(path), paths.DATA_DIR)
        self.assertEqual(paths.APP_DIR, paths.DATA_DIR)


class TestWritableProbe(unittest.TestCase):
    def test_a_real_directory_is_writable_and_stays_clean(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(paths.is_writable(folder))
            self.assertEqual(os.listdir(folder), [])

    def test_a_missing_directory_is_not(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(paths.is_writable(os.path.join(folder, "missing")))

    def test_program_files_is_refused_at_once(self):
        """The first version used tempfile, which on Windows retries up to
        10 000 times when os.access() wrongly says the folder is writable:
        minutes of a frozen start-up in exactly the folder this check is for."""
        folder = os.environ.get("ProgramFiles", "C:\\Program Files")
        if not os.path.isdir(folder):
            self.skipTest("no Program Files folder")

        outcome = []
        worker = threading.Thread(
            target=lambda: outcome.append(paths.is_writable(folder)), daemon=True)
        worker.start()
        worker.join(timeout=5.0)

        self.assertFalse(worker.is_alive(),
                         "the probe is still retrying after 5 s")
        if outcome[0]:
            self.skipTest("running elevated: Program Files really is writable")


class TestMigrateLegacyData(unittest.TestCase):
    """Older packaged builds kept their data in _internal; a version that looks
    next to the executable must not make a 3 GB model look lost."""

    def setUp(self):
        self.data = os.path.join(tempfile.mkdtemp(), "Sotto")
        self.internal = os.path.join(self.data, "_internal")
        self._write(self.internal, "settings.json", "{}")
        self._write(self.internal, os.path.join("bin", "ggml-small.bin"), "model")
        self._write(self.internal, os.path.join("output", "meeting.txt"), "text")
        os.makedirs(os.path.join(self.internal, "output", ".tmp"))

    def tearDown(self):
        shutil.rmtree(os.path.dirname(self.data), ignore_errors=True)

    @staticmethod
    def _write(base, name, text):
        path = os.path.join(base, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    @staticmethod
    def _read(*parts):
        with open(os.path.join(*parts), encoding="utf-8") as handle:
            return handle.read()

    def _migrate(self, **overrides):
        options = dict(source_dir=self.internal, data_dir=self.data, frozen=True)
        options.update(overrides)
        return paths.migrate_legacy_data(**options)

    def test_everything_moves_next_to_the_executable(self):
        self.assertEqual(self._migrate(), ["settings.json", "bin", "output"])
        self.assertEqual(self._read(self.data, "bin", "ggml-small.bin"), "model")
        self.assertEqual(self._read(self.data, "output", "meeting.txt"), "text")
        self.assertEqual(self._read(self.data, "settings.json"), "{}")
        for name in paths.LEGACY_ITEMS:
            self.assertFalse(os.path.exists(os.path.join(self.internal, name)), name)

    def test_nothing_happens_outside_a_packaged_build(self):
        self.assertEqual(self._migrate(frozen=False), [])
        self.assertTrue(os.path.exists(os.path.join(self.internal, "bin")))

    def test_nothing_happens_when_both_directories_are_the_same(self):
        self.assertEqual(self._migrate(data_dir=self.internal), [])
        self.assertTrue(os.path.exists(os.path.join(self.internal, "bin")))

    def test_existing_data_is_never_overwritten(self):
        self._write(self.data, "settings.json", '{"new": true}')
        self.assertEqual(self._migrate(), ["bin", "output"])
        self.assertEqual(self._read(self.data, "settings.json"), '{"new": true}')
        self.assertEqual(self._read(self.internal, "settings.json"), "{}")

    def test_an_empty_placeholder_directory_is_replaced(self):
        os.makedirs(os.path.join(self.data, "bin"))
        self.assertIn("bin", self._migrate())
        self.assertEqual(self._read(self.data, "bin", "ggml-small.bin"), "model")

    def test_a_second_run_finds_nothing_to_do(self):
        self._migrate()
        self.assertEqual(self._migrate(), [])

    def test_when_a_move_is_impossible_the_data_stays_and_nothing_raises(self):
        with patch("audio_transcriber.paths.os.rename",
                   side_effect=OSError("another drive")):
            self.assertEqual(self._migrate(), [])
        self.assertEqual(self._read(self.internal, "bin", "ggml-small.bin"), "model")

    def test_with_nothing_to_migrate_it_is_quiet(self):
        shutil.rmtree(self.internal)
        self.assertEqual(self._migrate(), [])

    def test_the_real_defaults_leave_a_source_checkout_alone(self):
        """Running from source (as the tests do) must never move anything."""
        self.assertEqual(paths.migrate_legacy_data(), [])


class TestMigrateFromAudioTranscriber(unittest.TestCase):
    """Until 2.0 the folder and the executable were called AudioTranscriber. A
    portable install keeps its data in that folder; Sotto, unpacked next to it
    or into it, would otherwise start without models and settings."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.old = os.path.join(self.root, "AudioTranscriber")
        self._write(self.old, "AudioTranscriber.exe", "")
        self._write(self.old, "settings.json", "{}")
        self._write(self.old, os.path.join("bin", "ggml-small.bin"), "model")
        self._write(self.old, os.path.join("output", "meeting.txt"), "text")

    _write = staticmethod(TestMigrateLegacyData._write)
    _read = staticmethod(TestMigrateLegacyData._read)

    def _migrate(self, app_dir, **overrides):
        options = dict(source_dir=os.path.join(app_dir, "_internal"),
                       data_dir=app_dir, frozen=True,
                       executable=os.path.join(app_dir, "Sotto.exe"))
        options.update(overrides)
        return paths.migrate_legacy_data(**options)

    def test_sotto_next_to_the_old_folder_takes_its_data(self):
        new = os.path.join(self.root, "Sotto")
        self.assertEqual(self._migrate(new), ["settings.json", "bin", "output"])
        self.assertEqual(self._read(new, "bin", "ggml-small.bin"), "model")
        self.assertEqual(self._read(new, "output", "meeting.txt"), "text")
        self.assertEqual(self._read(new, "settings.json"), "{}")
        for name in paths.LEGACY_ITEMS:
            self.assertFalse(os.path.exists(os.path.join(self.old, name)), name)

    def test_sotto_unpacked_into_the_old_folder_takes_its_data(self):
        new = os.path.join(self.old, "Sotto")
        self.assertEqual(self._migrate(new), ["settings.json", "bin", "output"])
        self.assertEqual(self._read(new, "bin", "ggml-small.bin"), "model")

    def test_a_folder_without_the_old_executable_is_left_alone(self):
        os.remove(os.path.join(self.old, "AudioTranscriber.exe"))
        self.assertEqual(self._migrate(os.path.join(self.root, "Sotto")), [])
        self.assertTrue(os.path.exists(os.path.join(self.old, "bin")))

    def test_nothing_moves_where_sotto_keeps_its_data_elsewhere(self):
        """A read-only install uses the per-user folder, whose name never
        changed - the folder next to it is none of its business."""
        new = os.path.join(self.root, "Sotto")
        per_user = os.path.join(self.root, "profile", "AudioTranscriber")
        self.assertEqual(self._migrate(new, data_dir=per_user), [])
        self.assertTrue(os.path.exists(os.path.join(self.old, "bin")))

    def test_what_sotto_has_already_is_never_overwritten(self):
        new = os.path.join(self.root, "Sotto")
        self._write(new, "settings.json", '{"new": true}')
        self.assertEqual(self._migrate(new), ["bin", "output"])
        self.assertEqual(self._read(new, "settings.json"), '{"new": true}')
        self.assertEqual(self._read(self.old, "settings.json"), "{}")

    def test_internal_comes_first_and_the_old_folder_fills_in(self):
        new = os.path.join(self.root, "Sotto")
        self._write(os.path.join(new, "_internal"), "settings.json", '{"internal": 1}')
        self.assertEqual(self._migrate(new), ["settings.json", "bin", "output"])
        self.assertEqual(self._read(new, "settings.json"), '{"internal": 1}')
        self.assertEqual(self._read(new, "bin", "ggml-small.bin"), "model")

    def test_a_folder_that_was_renamed_back_is_not_its_own_source(self):
        """Sotto in a folder called AudioTranscriber: the 'old folder next to
        it' is itself, and there is nothing to move."""
        self._write(self.old, "Sotto.exe", "")
        self.assertEqual(self._migrate(self.old), [])
        self.assertEqual(self._read(self.old, "bin", "ggml-small.bin"), "model")

    def test_a_running_old_version_keeps_its_data_and_nothing_raises(self):
        with patch("audio_transcriber.paths.os.rename",
                   side_effect=OSError("in use")):
            self.assertEqual(self._migrate(os.path.join(self.root, "Sotto")), [])
        self.assertEqual(self._read(self.old, "bin", "ggml-small.bin"), "model")


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
