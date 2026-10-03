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
                                   model_index=5, language="en",
                                   filename="meeting",
                                   output_dir="/custom/output/folder")
        settings.api_key = SAMPLE_KEY
        config.save(settings, self.path)

        loaded, warnings = config.load(self.path)
        self.assertEqual(warnings, [])
        self.assertEqual(loaded.mic_device, "20: Microphone")
        self.assertEqual(loaded.mic_gain_db, -8.0)
        self.assertEqual(loaded.model_index, 5)
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
        self.assertEqual(loaded.model_index, config.Settings().model_index)

    def test_invalid_single_value_is_ignored(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"mic_gain_db": "very loud", "filename": "ok"}, handle)
        loaded, warnings = config.load(self.path)
        self.assertEqual(loaded.filename, "ok")
        self.assertEqual(loaded.mic_gain_db, 0.0)
        self.assertTrue(warnings)

    def test_model_index_is_clamped(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"model_index": 99}, handle)
        loaded, _warnings = config.load(self.path)
        self.assertEqual(loaded.model_index, len(config.MODEL_CHOICES) - 1)

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
        settings = config.Settings(model_index=3, use_gpu=True)
        self.assertTrue(pipeline.build_backend(settings).allow_gpu)
        self.assertTrue(pipeline.build_backend(settings, greedy=True,
                                               live=True).allow_gpu)
        self.assertFalse(pipeline.build_backend(config.Settings(model_index=3))
                         .allow_gpu)


class TestModelSelection(unittest.TestCase):
    def test_cloud_entry(self):
        settings = config.Settings(model_index=0)
        self.assertTrue(settings.uses_cloud())
        self.assertIsNone(settings.model_name())

    def test_live_model_never_exceeds_small(self):
        """Regression H9: the live preview must not start large-v3 on the CPU
        and block every core."""
        for index in range(len(config.MODEL_CHOICES)):
            settings = config.Settings(model_index=index)
            self.assertIn(settings.live_model_name(), ("tiny", "base", "small"))

    def test_thread_count_leaves_headroom(self):
        settings = config.Settings(whisper_threads=0)
        self.assertGreaterEqual(settings.threads(), 1)
        self.assertLessEqual(settings.threads(), max(1, (os.cpu_count() or 4) - 2))
        self.assertEqual(config.Settings(whisper_threads=6).threads(), 6)


if __name__ == "__main__":
    unittest.main()
