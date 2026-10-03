"""Settings: typed, versioned, with secure key storage.

Replaces the two ad-hoc dict methods of the previous version (save_settings /
load_settings), which silently swallowed every exception.
"""

import json
import os
from dataclasses import dataclass, asdict, field, fields

from . import secretstore
from .paths import CFG_PATH

SCHEMA_VERSION = 2

# (display name, whisper model name or None for cloud)
MODEL_CHOICES = [
    ("ElevenLabs Scribe (Cloud API - state of the art)", None),
    ("tiny (75 MB - very fast)", "tiny"),
    ("base (142 MB - fast)", "base"),
    ("small (466 MB - balanced)", "small"),
    ("medium (1.5 GB - slow)", "medium"),
    ("large-v3-turbo (1.5 GB - fast and accurate)", "large-v3-turbo"),
    ("large-v3 (3.1 GB - slow, highest accuracy)", "large-v3"),
]

LANGUAGE_CHOICES = [
    ("German", "de"), ("English", "en"), ("Detect automatically", "auto"),
    ("Turkish", "tr"), ("French", "fr"), ("Spanish", "es"),
    ("Italian", "it"), ("Arabic", "ar"),
]


@dataclass
class Settings:
    # --- Devices ----------------------------------------------------------
    mic_device: str = ""
    loop_device: str = ""

    # --- Levels (affect only the audible mixdown, NOT speaker attribution -
    #     see audit finding H2) --------------------------------------------
    mic_gain_db: float = 0.0
    loop_gain_db: float = 0.0

    # --- AI ---------------------------------------------------------------
    model_index: int = 3            # default: local 'small'
    language: str = "de"
    whisper_threads: int = 0        # 0 = automatic
    # Silero VAD: off by default. Measured against a real recording, the VAD
    # path of this whisper.cpp build merges speech regions that are far apart
    # into a single segment (1.79 s - 41.83 s as one line) and loses the last
    # ~20 seconds. Accurate timestamps matter more for speaker attribution
    # than the small gain against hallucinations, which diarize.py filters
    # out energy-wise anyway.
    use_vad: bool = False
    # Recognise while recording, so stopping only has to catch up with the
    # tail. Local models only - see pipeline.live_transcription_possible().
    live_transcribe: bool = True
    live_preview: bool = True       # show the running text while recording
    elevenlabs_model_id: str = "scribe_v2"

    # --- Output -----------------------------------------------------------
    filename: str = "my_meeting"
    output_dir: str = ""            # empty = default paths.OUT_DIR
    separate_tracks: bool = True    # transcribe both tracks independently
    keep_raw_tracks: bool = False   # keep raw tracks after processing

    # --- Internal ---------------------------------------------------------
    schema_version: int = SCHEMA_VERSION
    elevenlabs_api_key_enc: str = ""

    # Runtime field, never persisted in clear text
    api_key: str = field(default="", repr=False, compare=False)

    # Set by the loader when a plain-text key was migrated
    migrated_plaintext_key: bool = field(default=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    def get_output_dir(self):
        """Returns the configured output directory or default OUT_DIR."""
        if self.output_dir and self.output_dir.strip():
            return os.path.abspath(self.output_dir.strip())
        from .paths import OUT_DIR
        return OUT_DIR

    def model_name(self):

        """whisper model name for the current choice, or None for cloud."""
        index = max(0, min(self.model_index, len(MODEL_CHOICES) - 1))
        return MODEL_CHOICES[index][1]

    def uses_cloud(self):
        return self.model_name() is None

    def live_model_name(self):
        """Model used for the live preview.

        Fixes H9: the previous version ran the preview with the full user
        model - possibly large-v3 on the CPU, every 3 seconds, over the entire
        recording so far. The preview now always uses a small model; final
        quality comes from the closing pass.
        """
        chosen = self.model_name()
        if chosen in ("tiny", "base", "small"):
            return chosen
        return "small"

    def threads(self):
        if self.whisper_threads > 0:
            return self.whisper_threads
        # Measured on the development machine: small -t 4 = 15.6 s,
        # -t 10 = 10.7 s. Two cores stay free for the GUI and audio threads.
        return max(1, min((os.cpu_count() or 4) - 2, 12))


# ----------------------------------------------------------------------
def load(path=CFG_PATH):
    """Load the settings. Returns (Settings, warnings).

    Invalid individual values are dropped instead of invalidating the whole
    file; the previous version silently reset everything to defaults on any
    error.
    """
    warnings = []
    settings = Settings()

    if not os.path.exists(path):
        settings.api_key = secretstore.from_environment()
        return settings, warnings

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        warnings.append(f"settings.json could not be read ({exc}); "
                        f"falling back to defaults.")
        settings.api_key = secretstore.from_environment()
        return settings, warnings

    if not isinstance(raw, dict):
        warnings.append("settings.json has an unexpected format.")
        return settings, warnings

    known = {f.name: f.type for f in fields(Settings)}
    for key, value in raw.items():
        if key not in known or key in ("api_key", "migrated_plaintext_key"):
            continue
        try:
            current = getattr(settings, key)
            if isinstance(current, bool):
                setattr(settings, key, bool(value))
            elif isinstance(current, int):
                setattr(settings, key, int(value))
            elif isinstance(current, float):
                setattr(settings, key, float(value))
            else:
                setattr(settings, key, str(value))
        except (TypeError, ValueError):
            warnings.append(f"Setting '{key}' was invalid and has been ignored.")

    settings.model_index = max(0, min(settings.model_index, len(MODEL_CHOICES) - 1))
    settings.mic_gain_db = max(-40.0, min(40.0, settings.mic_gain_db))
    settings.loop_gain_db = max(-40.0, min(40.0, settings.loop_gain_db))

    # --- Obtain the key ------------------------------------------------
    settings.api_key = secretstore.decrypt(settings.elevenlabs_api_key_enc)

    # Migration: adopt a plain-text key from schema v1 and warn about it
    legacy = str(raw.get("elevenlabs_api_key", "")).strip()
    if legacy and not settings.api_key:
        settings.api_key = legacy
        settings.migrated_plaintext_key = True
        warnings.append(
            "The API key was stored in clear text in settings.json. It has "
            "been adopted and will be encrypted the next time you save. "
            "IMPORTANT: revoke that key in the ElevenLabs dashboard and issue "
            "a new one - the old value sat unprotected on disk."
        )

    if not settings.api_key:
        settings.api_key = secretstore.from_environment()

    return settings, warnings


def save(settings, path=CFG_PATH):
    """Save atomically. The key is never written in clear text."""
    data = asdict(settings)
    data.pop("api_key", None)
    data.pop("migrated_plaintext_key", None)
    data["schema_version"] = SCHEMA_VERSION

    if settings.api_key:
        try:
            data["elevenlabs_api_key_enc"] = secretstore.encrypt(settings.api_key)
        except OSError:
            # No DPAPI (e.g. non-Windows): store nothing rather than clear
            # text. Users can set ELEVENLABS_API_KEY instead.
            data["elevenlabs_api_key_enc"] = ""
    else:
        data["elevenlabs_api_key_enc"] = ""

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)   # atomic: never a half-written settings.json
