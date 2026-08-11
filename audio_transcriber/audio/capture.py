"""Capture engine: two tracks, streamed to disk, drift corrected.

Fixes audit findings K3, H1, H3 and M3.

K3 (memory): the previous version kept every chunk in a queue AND in a list,
raw and multi-channel. An 8-channel loopback costs 5.5 GB per hour and copy,
about 13.6 GB/h in total. Here the capture thread downmixes to mono right
away and writes to a file; memory usage is constant (only a 30 second window
for the live preview stays in RAM).

H1 (thread leak): a single bool flag controlled all threads. On a device
change it was set to False and back to True 50 ms later - if an old thread
missed that window it ran forever and wrote into the same queue as the new
one. Every generation now owns its event.

H3 (synchronisation): both streams are opened first and then started back to
back; the start offset is measured and compensated when the recording ends.
Dropouts are detected against the wall clock and padded with silence so the
two timelines cannot drift apart.

M3 (silent failures): stream errors no longer vanish into 'except: pass' but
are collected and reported at start.
"""

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from . import dsp

BLOCK_FRAMES = 1024
LIVE_WINDOW_S = 30.0
DRIFT_CORRECT_S = 0.05      # silence is inserted from 50 ms of lag onwards


def drift_deficit(elapsed_s, rate, frames_written, block_len,
                  threshold_s=DRIFT_CORRECT_S):
    """How many samples are missing compared to the wall clock?

    Positive -> the source lost blocks (dropout); silence must be inserted so
                both tracks stay time-aligned.
    Negative -> the device clock runs ahead of the system clock.
    0        -> within tolerance, nothing to do.

    Extracted as its own function so the timeline correction can be tested
    without audio hardware (see tests/test_capture.py).
    """
    expected = int(elapsed_s * rate)
    deficit = expected - (frames_written + block_len)
    if deficit >= rate * threshold_s:
        return deficit
    if deficit < -rate * threshold_s:
        return deficit
    return 0


@dataclass
class TrackResult:
    """Result of one recorded track."""
    path: str
    rate: int
    frames: int
    start_offset_s: float = 0.0      # offset against the common origin
    inserted_silence_s: float = 0.0  # padded because of dropouts
    surplus_s: float = 0.0           # lead over the wall clock (clock drift)
    device_name: str = ""

    @property
    def duration_s(self):
        return self.frames / float(self.rate) if self.rate else 0.0


@dataclass
class RecordingResult:
    mic: TrackResult = None
    sys: TrackResult = None
    warnings: list = field(default_factory=list)

    @property
    def has_audio(self):
        return any(track is not None and track.frames > 0
                   for track in (self.mic, self.sys))


class _Track:
    """Internal state of a running track."""

    def __init__(self, kind, device):
        self.kind = kind                 # "mic" | "sys"
        self.device = device
        self.channels = max(1, device.max_input_channels)
        self.rate = int(device.default_rate)
        self.stream = None
        self.downmixer = dsp.ActiveChannelDownmixer(self.channels)
        self.level = 0.0                 # raw RMS (BEFORE the gain sliders)

        self.window = deque(maxlen=int(math.ceil(LIVE_WINDOW_S * self.rate / BLOCK_FRAMES)))
        self.lock = threading.Lock()
        self.writer = None
        self.path = None
        self.frames = 0
        self.inserted = 0
        self.surplus = 0
        self.first_block_at = None
        self.error = None


class AudioEngine:
    """Keeps two input streams open permanently: always for the level meters,
    additionally writing to disk while recording."""

    def __init__(self, pa, tmp_dir):
        self._pa = pa
        self._tmp_dir = tmp_dir
        self._stop = None                # event of the current generation
        self._threads = []
        self._tracks = {}
        self._recording = False
        self._gen_lock = threading.Lock()
        # Serialises whole reconfigurations. _gen_lock alone is not enough:
        # configure() releases it between stop_streams() and opening the new
        # streams, so two rapid device changes could interleave as
        # A.stop / B.stop / A.assign / B.assign and leak A's streams.
        self._configure_lock = threading.Lock()
        self._tap = None
        self.last_error = None

    # ------------------------------------------------------------------
    def set_tap(self, callback):
        """Watch every mono block on its way into the raw track file.

        callback(kind, samples, rate, position) runs on the capture thread and
        must return immediately - the live transcriber only appends to a
        buffer. Pass None to detach.

        The tap sees exactly what the file receives, including the silence the
        drift correction inserts, so a sample position means the same point in
        time on both sides. That is what lets the live segments and the closing
        pass share one timeline. `position` is where the block sits in the
        file, which lets a consumer notice that it attached too late to have
        heard the beginning.
        """
        self._tap = callback

    def _emit(self, track, samples):
        """Called right after the samples went into the file, so
        track.frames - len(samples) is exactly where they landed."""
        tap = self._tap
        if tap is None:
            return
        try:
            tap(track.kind, samples, track.rate, track.frames - len(samples))
        except Exception:
            # A broken consumer must never take the capture thread with it.
            # Detach instead of failing on every following block.
            self._tap = None

    @property
    def is_configuring(self):
        """True while a reconfiguration is in flight.

        During that window the engine may legitimately have no active track,
        which is not the same thing as having no working device.
        """
        return self._configure_lock.locked()

    # ------------------------------------------------------------------
    # Levels for the meters (raw, before gain - see H2)
    # ------------------------------------------------------------------
    @property
    def mic_level(self):
        track = self._tracks.get("mic")
        return track.level if track else 0.0

    @property
    def sys_level(self):
        track = self._tracks.get("sys")
        return track.level if track else 0.0

    @property
    def is_recording(self):
        return self._recording

    def stream_errors(self):
        return [track.error for track in self._tracks.values() if track.error]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def configure(self, mic_device, sys_device):
        """Restart monitoring. Returns a list of warnings.

        Must run off the GUI thread (join blocks for up to 2 s).
        """
        with self._configure_lock:
            return self._configure(mic_device, sys_device)

    def _configure(self, mic_device, sys_device):
        warnings = []
        self.stop_streams()

        with self._gen_lock:
            stop = threading.Event()
            tracks = {}
            opened = []

            try:
                # OPEN both streams first (start=False), then start them back
                # to back -> minimal start offset (H3).
                for kind, device in (("mic", mic_device), ("sys", sys_device)):
                    if device is None:
                        continue
                    track = _Track(kind, device)
                    try:
                        track.stream = self._open_stream(track)
                        opened.append(track)
                    except Exception as exc:
                        track.error = (
                            f"{'Microphone' if kind == 'mic' else 'System audio'} "
                            f"'{device.name}' could not be opened: {exc}")
                        warnings.append(track.error)
                    tracks[kind] = track

                for track in opened:
                    track.stream.start_stream()

            except Exception as exc:            # pragma: no cover - safety net
                for track in opened:
                    try:
                        track.stream.close()
                    except Exception:
                        pass
                warnings.append(f"Audio initialisation failed: {exc}")
                return warnings

            self._tracks = tracks
            self._stop = stop
            self._threads = []
            for track in opened:
                thread = threading.Thread(target=self._reader_loop,
                                          args=(track, stop), daemon=True,
                                          name=f"capture-{track.kind}")
                thread.start()
                self._threads.append(thread)

        return warnings

    def _open_stream(self, track):
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            import pyaudio
        return self._pa.open(
            format=pyaudio.paFloat32,
            channels=track.channels,
            rate=track.rate,
            input=True,
            input_device_index=track.device.index,
            frames_per_buffer=BLOCK_FRAMES,
            start=False,
        )

    def stop_streams(self):
        """End the current thread generation and wait for it."""
        with self._gen_lock:
            stop, threads, tracks = self._stop, self._threads, self._tracks
            self._stop, self._threads = None, []

        if stop is not None:
            stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
        for track in (tracks or {}).values():
            self._close_writer(track)
            if track.stream is not None:
                try:
                    track.stream.stop_stream()
                    track.stream.close()
                except Exception:
                    pass
                track.stream = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def start_recording(self, base_name):
        """Open the capture files. Raises RuntimeError if not a single track
        is active - in that case the previous version silently recorded
        nothing and only mentioned it when you pressed stop."""
        active = [track for track in self._tracks.values() if track.stream is not None]
        if not active:
            raise RuntimeError(
                "Neither audio source is active. " + " ".join(self.stream_errors()))

        os.makedirs(self._tmp_dir, exist_ok=True)
        for track in active:
            path = os.path.join(self._tmp_dir, f"{base_name}.{track.kind}.raw.wav")
            writer = sf.SoundFile(path, mode="w", samplerate=track.rate,
                                  channels=1, subtype="PCM_16")
            with track.lock:
                track.writer = writer
                track.path = path
                track.frames = 0
                track.inserted = 0
                track.surplus = 0
                track.first_block_at = None
        self._recording = True

    def stop_recording(self):
        """Close the capture files and return the result."""
        self._recording = False
        time.sleep(0.15)                 # let in-flight blocks finish writing

        result = RecordingResult()
        starts = [track.first_block_at for track in self._tracks.values()
                  if track.first_block_at is not None and track.path]
        origin = min(starts) if starts else None

        for kind, track in self._tracks.items():
            path, frames = self._close_writer(track)
            if path is None:
                continue
            offset = 0.0
            if origin is not None and track.first_block_at is not None:
                offset = max(0.0, track.first_block_at - origin)
            setattr(result, kind, TrackResult(
                path=path, rate=track.rate, frames=frames,
                start_offset_s=offset,
                inserted_silence_s=track.inserted / float(track.rate),
                surplus_s=track.surplus / float(track.rate),
                device_name=track.device.name,
            ))

        for track_result in (result.mic, result.sys):
            if track_result is None:
                continue
            if track_result.inserted_silence_s > 0.5:
                result.warnings.append(
                    f"Track '{track_result.device_name}': "
                    f"{track_result.inserted_silence_s:.1f} s of dropouts were "
                    f"padded with silence to keep the timeline correct.")
            if track_result.surplus_s > 1.0:
                result.warnings.append(
                    f"Track '{track_result.device_name}': clock deviation of "
                    f"{track_result.surplus_s:.1f} s against the system clock.")
        if (result.mic and result.sys
                and abs(result.mic.start_offset_s - result.sys.start_offset_s) > 0.25):
            result.warnings.append("The two tracks started more than 250 ms apart; "
                                   "the offset has been compensated.")
        return result

    def _close_writer(self, track):
        with track.lock:
            writer, path, frames = track.writer, track.path, track.frames
            track.writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            return path, frames
        return None, 0

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------
    def live_window(self):
        """The last LIVE_WINDOW_S seconds of both tracks as 16 kHz mono.

        Hard upper bound in RAM - unlike the previous version, which resampled
        and transcribed the entire recording so far for every preview pass
        (audit finding K4).
        """
        result = {}
        for kind, track in self._tracks.items():
            blocks = list(track.window)
            if not blocks:
                result[kind] = np.zeros(0, dtype=np.float32)
                continue
            mono = np.concatenate(blocks)
            result[kind] = dsp.resample(mono, track.rate, dsp.TARGET_RATE)
        return result.get("mic"), result.get("sys")

    # ------------------------------------------------------------------
    def _reader_loop(self, track, stop):
        rate = track.rate
        origin = None
        try:
            while not stop.is_set():
                try:
                    data = track.stream.read(BLOCK_FRAMES, exception_on_overflow=False)
                except Exception as exc:
                    track.error = f"Read error on '{track.device.name}': {exc}"
                    break
                if not data:
                    continue

                block = np.frombuffer(data, dtype=np.float32)
                mono = track.downmixer.process(block)
                if mono.size == 0:
                    continue

                track.level = dsp.rms(mono)
                track.window.append(mono)

                with track.lock:
                    writer = track.writer
                    if writer is None:
                        continue
                    now = time.perf_counter()
                    if track.first_block_at is None:
                        track.first_block_at = now
                        origin = now
                        writer.write(mono)
                        track.frames += len(mono)
                        self._emit(track, mono)
                        continue

                    # --- drift correction against the wall clock (H3) -------
                    deficit = drift_deficit(now - origin, rate, track.frames, len(mono))
                    if deficit > 0:
                        silence = np.zeros(deficit, dtype=np.float32)
                        writer.write(silence)
                        track.frames += deficit
                        track.inserted += deficit
                        self._emit(track, silence)
                    elif deficit < 0:
                        track.surplus = max(track.surplus, -deficit)

                    writer.write(mono)
                    track.frames += len(mono)
                    self._emit(track, mono)
        except Exception as exc:                # pragma: no cover
            track.error = f"Capture thread '{track.device.name}': {exc}"
        finally:
            track.level = 0.0


# ----------------------------------------------------------------------
def load_track(track_result, target_rate=dsp.TARGET_RATE):
    """Read a raw track, compensate the start offset and resample.

    The start offset is prepended as silence: a track whose stream started
    later missed the beginning and must move back on the common timeline.
    """
    if track_result is None or not os.path.exists(track_result.path):
        return np.zeros(0, dtype=np.float32)

    data, rate = sf.read(track_result.path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    resampled = dsp.resample(data, rate, target_rate)
    pad = int(round(track_result.start_offset_s * target_rate))
    if pad > 0:
        resampled = np.concatenate([np.zeros(pad, dtype=np.float32), resampled])
    return resampled
