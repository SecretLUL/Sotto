"""Tests for settings and key storage (audit finding H6)."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from audio_transcriber import config, secretstore

SAMPLE_KEY = "sk_testkey_0123456789abcdef"


class TestSecretStore(unittest.TestCase):
    @unittest.skipUnless(secretstore.is_available(), "DPAPI is Windows only")
    def test_roundtrip(self):
        token = secretstore.encrypt(SAMPLE_KEY)
        self.assertNotIn(SAMPLE_KEY, token)
        self.assertEqual(secretstore.decrypt(token), SAMPLE_KEY)

    def test_empty_values(self):
        self.assertEqual(secretstore.decrypt(""), "")
        self.assertEqual(secretstore.decrypt("not-base64!!"), "")
        self.assertEqual(secretstore.decrypt("AAAAAAAAAA"), "")


class TestSettingsFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "settings.json")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def test_roundtrip(self):
        settings = config.Settings(mic_device="20: Microphone",
                                   loop_device="22: Loopback",
                                   mic_gain_db=-8.0, loop_gain_db=10.0,
                                   model="large-v3-turbo", language="en",
                                   filename="meeting",
                                   output_dir="/custom/output/folder")
        settings.api_key = SAMPLE_KEY
        config.save(settings, self.path)

        loaded, warnings = config.load(self.path)
        self.assertEqual(warnings, [])
        self.assertEqual(loaded.mic_device, "20: Microphone")
        self.assertEqual(loaded.mic_gain_db, -8.0)
        self.assertEqual(loaded.model, "large-v3-turbo")
        self.assertEqual(loaded.language, "en")
        self.assertEqual(loaded.output_dir, "/custom/output/folder")
        self.assertEqual(loaded.get_output_dir(), os.path.abspath("/custom/output/folder"))
        if secretstore.is_available():
            self.assertEqual(loaded.api_key, SAMPLE_KEY)


    def test_key_is_never_written_in_clear_text(self):
        """The central requirement of H6."""
        settings = config.Settings()
        settings.api_key = SAMPLE_KEY
        config.save(settings, self.path)

        with open(self.path, "rb") as handle:
            raw = handle.read()
        self.assertNotIn(SAMPLE_KEY.encode(), raw)

        data = json.loads(raw.decode("utf-8"))
        self.assertNotIn("elevenlabs_api_key", data)
        if secretstore.is_available():
            self.assertTrue(data["elevenlabs_api_key_enc"])

    def test_migration_from_plaintext_schema_v1(self):
        """An existing file in the old format is adopted - with a warning."""
        legacy = {
            "mic_device": "1: Microphone (Yeti X)",
            "loop_device": "4: Headphones (PRO X 2 LIGHTSPEED)",
            "mic_gain_db": -8.0,
            "loop_gain_db": 10.0,
            "model_index": 0,
            "elevenlabs_api_key": SAMPLE_KEY,
            "live_transcribe": True,
            "filename": "test.wav",
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        loaded, warnings = config.load(self.path)
        self.assertEqual(loaded.api_key, SAMPLE_KEY)
        self.assertTrue(loaded.migrated_plaintext_key)
        self.assertTrue(any("revoke" in warning for warning in warnings))
        self.assertEqual(loaded.mic_gain_db, -8.0)

        # After saving, the key is no longer in the file as clear text
        config.save(loaded, self.path)
        with open(self.path, "rb") as handle:
            self.assertNotIn(SAMPLE_KEY.encode(), handle.read())

    def test_corrupt_file_falls_back_to_defaults(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{ this is not json")
        loaded, warnings = config.load(self.path)
        self.assertTrue(warnings)
        self.assertEqual(loaded.model, config.Settings().model)

    def test_invalid_single_value_is_ignored(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"mic_gain_db": "very loud", "filename": "ok"}, handle)
        loaded, warnings = config.load(self.path)
        self.assertEqual(loaded.filename, "ok")
        self.assertEqual(loaded.mic_gain_db, 0.0)
        self.assertTrue(warnings)

    def test_save_is_atomic(self):
        config.save(config.Settings(), self.path)
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_the_default_path_is_looked_up_when_called(self):
        """Regression: load() and save() bound CFG_PATH when the module was
        imported, so pointing config.CFG_PATH at a temporary file - as the GUI
        tests do to 'never touch the user's real settings' - changed nothing."""
        with patch.object(config, "CFG_PATH", self.path):
            config.save(config.Settings(filename="patched"))
            self.assertTrue(os.path.exists(self.path))
            loaded, _warnings = config.load()
        self.assertEqual(loaded.filename, "patched")

    def test_unexpected_json_still_falls_back_to_the_environment_key(self):
        """The other early returns did; this one forgot."""
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("[1, 2, 3]")
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env-key-123"}):
            loaded, warnings = config.load(self.path)
        self.assertTrue(warnings)
        self.assertEqual(loaded.api_key, "env-key-123")

    def test_a_key_from_the_environment_is_not_copied_into_the_file(self):
        """A stored copy beats the variable on the next start, so saving it
        would make a rotated ELEVENLABS_API_KEY silently change nothing."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env-key-123"}):
            settings, _warnings = config.load(self.path)       # no file yet
            self.assertEqual(settings.api_key, "env-key-123")
            config.save(settings, self.path)
            with open(self.path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["elevenlabs_api_key_enc"], "")

            # A key the user typed in is still stored.
            settings.api_key = SAMPLE_KEY
            config.save(settings, self.path)
            with open(self.path, encoding="utf-8") as handle:
                stored = json.load(handle)["elevenlabs_api_key_enc"]
        if secretstore.is_available():
            self.assertTrue(stored)


class TestGpuSetting(unittest.TestCase):
    def test_the_gpu_is_off_unless_asked_for(self):
        self.assertFalse(config.Settings().use_gpu)

    def test_it_survives_a_save_and_load(self):
        folder = tempfile.mkdtemp()
        try:
            path = os.path.join(folder, "settings.json")
            config.save(config.Settings(use_gpu=True), path)
            loaded, _warnings = config.load(path)
            self.assertTrue(loaded.use_gpu)
        finally:
            for name in os.listdir(folder):
                os.remove(os.path.join(folder, name))
            os.rmdir(folder)

    def test_it_reaches_both_backends_the_final_pass_and_the_live_ones(self):
        from audio_transcriber import pipeline
        settings = config.Settings(model="small", use_gpu=True)
        self.assertTrue(pipeline.build_backend(settings).allow_gpu)
        self.assertTrue(pipeline.build_backend(settings, greedy=True,
                                               live=True).allow_gpu)
        self.assertFalse(pipeline.build_backend(config.Settings(model="small"))
                         .allow_gpu)


class TestModelStoredByName(unittest.TestCase):
    """settings.json names the model; it used to hold a position in a list.

    A new entry in the list - or one moved - changed which model every existing
    settings file meant: "small" turned into "medium" without the user touching
    anything.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "settings.json")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def _write(self, data):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def _stored(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_a_whisper_model_is_saved_by_its_name(self):
        config.save(config.Settings(model="large-v3"), self.path)
        stored = self._stored()
        self.assertEqual(stored["model"], "large-v3")
        self.assertNotIn("model_index", stored)
        loaded, warnings = config.load(self.path)
        self.assertEqual((loaded.model, warnings), ("large-v3", []))
        self.assertEqual(loaded.model_name(), "large-v3")

    def test_the_cloud_entry_is_saved_by_name_too(self):
        config.save(config.Settings(model=config.CLOUD_MODEL), self.path)
        loaded, warnings = config.load(self.path)
        self.assertEqual(warnings, [])
        self.assertTrue(loaded.uses_cloud())
        self.assertIsNone(loaded.model_name())

    def test_a_new_entry_in_the_list_does_not_move_the_choice(self):
        config.save(config.Settings(model="small"), self.path)
        grown = [("Brand new model", "brand-new")] + list(config.MODEL_CHOICES)
        with patch.object(config, "MODEL_CHOICES", grown):
            loaded, warnings = config.load(self.path)
            self.assertEqual(warnings, [])
            self.assertEqual(loaded.model_name(), "small")
            self.assertEqual(config.MODEL_CHOICES[config.model_index(loaded.model)][1],
                             "small")

    def test_a_file_from_before_this_holds_a_position_and_keeps_its_meaning(self):
        for index, (_label, name) in enumerate(config.MODEL_CHOICES):
            with self.subTest(index=index):
                self._write({"model_index": index})
                loaded, warnings = config.load(self.path)
                self.assertEqual(warnings, [])
                self.assertEqual(loaded.model_name(), name)

    def test_such_a_position_is_clamped_as_it_always_was(self):
        self._write({"model_index": 99})
        self.assertEqual(config.load(self.path)[0].model_name(),
                         config.MODEL_CHOICES[-1][1])
        self._write({"model_index": -4})
        self.assertEqual(config.load(self.path)[0].model_name(),
                         config.MODEL_CHOICES[0][1])

    def test_a_name_wins_over_a_position_when_both_are_there(self):
        self._write({"model": "tiny", "model_index": 6})
        self.assertEqual(config.load(self.path)[0].model_name(), "tiny")

    def test_a_position_that_is_not_a_number_is_ignored_with_a_warning(self):
        self._write({"model_index": "the big one"})
        loaded, warnings = config.load(self.path)
        self.assertEqual(loaded.model, config.Settings().model)
        self.assertTrue(any("model_index" in warning for warning in warnings))

    def test_an_unknown_name_falls_back_to_the_default_and_says_so(self):
        self._write({"model": "large-v9"})
        loaded, warnings = config.load(self.path)
        self.assertEqual(loaded.model, config.Settings().model)
        self.assertTrue(any("large-v9" in warning for warning in warnings))

    def test_a_settings_object_with_a_bad_name_still_gives_a_usable_model(self):
        self.assertEqual(config.Settings(model="nonsense").model_name(),
                         config.Settings().model_name())

    def test_every_entry_of_the_list_round_trips_through_its_stored_name(self):
        for index, (_label, name) in enumerate(config.MODEL_CHOICES):
            with self.subTest(entry=name):
                key = config.model_key(index)
                self.assertEqual(config.model_index(key), index)
                self.assertEqual(config.Settings(model=key).model_name(), name)

    def test_out_of_range_positions_for_the_combo_box_stay_in_range(self):
        self.assertEqual(config.model_key(-1), config.model_key(0))
        self.assertEqual(config.model_key(999), config.model_key(len(config.MODEL_CHOICES) - 1))
        self.assertEqual(config.model_index("no such model"),
                         config.model_index(config.Settings().model))


class TestModelSelection(unittest.TestCase):
    def test_cloud_entry(self):
        settings = config.Settings(model=config.CLOUD_MODEL)
        self.assertTrue(settings.uses_cloud())
        self.assertIsNone(settings.model_name())

    def test_live_model_never_exceeds_small(self):
        """Regression H9: the live preview must not start large-v3 on the CPU
        and block every core."""
        for key in config.model_keys():
            settings = config.Settings(model=key)
            self.assertIn(settings.live_model_name(), ("tiny", "base", "small"))

    def test_thread_count_leaves_headroom(self):
        settings = config.Settings(whisper_threads=0)
        self.assertGreaterEqual(settings.threads(), 1)
        self.assertLessEqual(settings.threads(), max(1, (os.cpu_count() or 4) - 2))
        self.assertEqual(config.Settings(whisper_threads=6).threads(), 6)


class TestSettingsSnapshot(unittest.TestCase):
    """A run works on a copy: the window goes on changing the live settings
    (the gain sliders, 'Save settings') while a transcription is under way."""

    def test_a_snapshot_is_independent_of_the_original(self):
        original = config.Settings(language="en", mic_gain_db=3.0,
                                   model="medium", output_dir="/a")
        snapshot = original.snapshot()
        self.assertEqual(snapshot, original)

        original.language = "tr"
        original.mic_gain_db = -9.0
        original.model = "tiny"
        original.output_dir = "/b"
        self.assertEqual((snapshot.language, snapshot.mic_gain_db, snapshot.model,
                          snapshot.output_dir), ("en", 3.0, "medium", "/a"))

    def test_the_runtime_only_fields_are_copied_too(self):
        original = config.Settings()
        original.api_key = SAMPLE_KEY
        original.migrated_plaintext_key = True
        snapshot = original.snapshot()
        self.assertEqual(snapshot.api_key, SAMPLE_KEY)
        self.assertTrue(snapshot.migrated_plaintext_key)

    def test_a_snapshot_is_a_settings_object_with_the_same_behaviour(self):
        snapshot = config.Settings(model=config.CLOUD_MODEL, whisper_threads=3).snapshot()
        self.assertIsInstance(snapshot, config.Settings)
        self.assertTrue(snapshot.uses_cloud())
        self.assertEqual(snapshot.threads(), 3)


if __name__ == "__main__":
    unittest.main()
