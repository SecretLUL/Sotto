"""Audio file loader supporting all common audio formats.

Uses soundfile (libsndfile) for direct reading of WAV, MP3, FLAC, OGG, AIFF, etc.
Falls back to system FFmpeg subprocess for M4A, AAC, WMA, MP4, WEBM, OPUS, etc.
"""

import os
import subprocess
import tempfile
import numpy as np

from . import dsp, stream
from ..transcribe.base import TranscriptionError


def load_audio_file(file_path: str, target_rate: int = dsp.TARGET_RATE) -> np.ndarray:
    """Load an audio file of any format and return it as 16 kHz mono float32 array.

    Raises:
        TranscriptionError: If file is missing, empty, corrupt, or unreadable.
    """
    if not os.path.exists(file_path):
        raise TranscriptionError(f"Audio file not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise TranscriptionError(f"Audio file is empty: {file_path}")

    # 1. Try reading directly with soundfile (fast, native). Chunk by chunk:
    # the whole file at its native rate, its channel average and the resampled
    # copy used to be in memory at once - about 1.8 GB per hour of stereo.
    try:
        resampled = stream.read_mono_resampled(file_path, target_rate)
    except Exception:
        # soundfile failed or format unsupported (e.g. m4a, aac, wma) -> try FFmpeg fallback
        pass
    else:
        # A readable-but-empty file is a decoding result, not a decoding
        # failure. Raising inside the try above sent it down the FFmpeg path
        # and reported a misleading "FFmpeg not found" instead.
        if len(resampled) == 0:
            raise TranscriptionError(f"Audio file contains no sample data: {file_path}")
        return resampled

    # 2. Fallback: FFmpeg conversion to temp WAV
    return _load_via_ffmpeg(file_path, target_rate)


def _load_via_ffmpeg(file_path: str, target_rate: int) -> np.ndarray:
    """Convert audio file using ffmpeg to 16 kHz mono PCM WAV."""
    create_no_window = 0x08000000 if os.name == "nt" else 0
    temp_fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)

    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", file_path,
            "-vn",                      # disable video
            "-ac", "1",                 # mono
            "-ar", str(target_rate),    # sample rate
            "-f", "wav",
            temp_wav
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, creationflags=create_no_window)

        if result.returncode != 0:
            err_msg = result.stderr.strip() or f"ffmpeg exit code {result.returncode}"
            raise TranscriptionError(
                f"Could not read audio file '{os.path.basename(file_path)}': {err_msg}")

        resampled = stream.read_mono_resampled(temp_wav, target_rate)
        if len(resampled) == 0:
            raise TranscriptionError(f"Audio file contains no audio data: {file_path}")
        return resampled
    except FileNotFoundError as exc:
        raise TranscriptionError(
            f"Unsupported audio format for '{os.path.basename(file_path)}'. "
            f"FFmpeg is required for this file type but was not found on system PATH.") from exc
    except Exception as exc:
        if isinstance(exc, TranscriptionError):
            raise
        raise TranscriptionError(
            f"Failed to decode audio file '{os.path.basename(file_path)}': {exc}") from exc
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass
