"""Signal processing: downmix, resampling, level measurement.

Deliberately free of GUI, file and hardware dependencies so this part is
testable with synthetic signals (see tests/test_dsp.py). The algorithmic
audit findings H2 and H7 lived exactly here.
"""

import math

import numpy as np
import scipy.signal as sps

TARGET_RATE = 16000          # whisper.cpp and ElevenLabs both expect 16 kHz
SILENCE_FLOOR = 1e-6         # below this: treat as digital silence


# ----------------------------------------------------------------------
# Downmix
# ----------------------------------------------------------------------
class ActiveChannelDownmixer:
    """Mix multi-channel blocks to mono, ignoring permanently silent channels.

    Why this is needed: the WASAPI loopback of a typical headset reports eight
    channels. Stereo content then occupies two of them and six are digitally
    silent. A naive mean(axis=1) throws away 10*log10(8/2) = 6 dB of level.

    The previous version decided this per block from instantaneous energy - on
    short pauses or hard-panned material the channel set could change between
    two blocks and the level jumped. Channel activity is now accumulated over
    time: a channel that has ever carried signal stays in the mix.
    """

    def __init__(self, channels, rel_threshold_db=-30.0):
        self.channels = max(1, int(channels))
        self.rel_threshold = 10.0 ** (rel_threshold_db / 20.0)
        self._energy = np.zeros(self.channels, dtype=np.float64)
        self._active = np.ones(self.channels, dtype=bool)

    @property
    def active_channels(self):
        return [i for i, active in enumerate(self._active) if active]

    def process(self, block):
        """block: 1-D interleaved float32. Returns 1-D mono float32."""
        if self.channels == 1:
            return np.asarray(block, dtype=np.float32)

        usable = (len(block) // self.channels) * self.channels
        if usable == 0:
            return np.zeros(0, dtype=np.float32)

        frames = np.asarray(block[:usable], dtype=np.float32).reshape(-1, self.channels)

        # Accumulate energy (sum of squares, monotonically increasing)
        self._energy += np.sum(frames.astype(np.float64) ** 2, axis=0)

        loudest = self._energy.max()
        if loudest > 0:
            # Energy is quadratic -> square the threshold as well
            self._active = self._energy >= loudest * (self.rel_threshold ** 2)
            if not self._active.any():
                self._active[:] = True

        return frames[:, self._active].mean(axis=1).astype(np.float32)


def downmix_active(data, channels, rel_threshold_db=-30.0):
    """One-shot variant for complete recordings (no streaming state)."""
    return ActiveChannelDownmixer(channels, rel_threshold_db).process(data)


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------
def resample(x, src_rate, dst_rate=TARGET_RATE):
    """Exact polyphase resampling in a single pass.

    Deliberately NOT called block by block: resample_poly filters every call
    independently, so applying it per block introduces a discontinuity at
    every block boundary (with 1024-sample blocks that is 47 clicks per
    second). The recording is therefore written to disk at its native rate and
    resampled in one go when the recording ends.
    """
    x = np.asarray(x, dtype=np.float32)
    if src_rate == dst_rate or len(x) == 0:
        return x
    divisor = math.gcd(int(dst_rate), int(src_rate))
    up, down = int(dst_rate) // divisor, int(src_rate) // divisor
    return sps.resample_poly(x, up, down).astype(np.float32)


# ----------------------------------------------------------------------
# Level measurement
# ----------------------------------------------------------------------
def rms(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def to_db(amplitude, floor_db=-90.0):
    if amplitude <= SILENCE_FLOOR:
        return floor_db
    return max(floor_db, 20.0 * math.log10(amplitude))


def frame_rms(x, frame_len):
    """RMS per frame. Returns a 1-D array (empty if x is shorter than one frame)."""
    x = np.asarray(x, dtype=np.float32)
    count = len(x) // frame_len
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frames = x[:count * frame_len].reshape(count, frame_len)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)).astype(np.float32)


def quietest_split(x, rate, earliest, latest, frame_s=0.05):
    """Sample index in [earliest, latest] with the least energy.

    Live transcription has to cut the running recording into chunks somewhere.
    Cutting through a word costs that word in both chunks, cutting into a pause
    costs nothing - so the boundary is moved to the quietest 50 ms of the
    search range instead of landing on a fixed sample number.

    Falls back to `latest` when the range is too short to search.
    """
    x = np.asarray(x, dtype=np.float32)
    earliest = max(0, int(earliest))
    latest = min(len(x), int(latest))
    frame = max(1, int(rate * frame_s))
    if latest - earliest < 2 * frame:
        return latest

    levels = frame_rms(x[earliest:latest], frame)
    if levels.size == 0:
        return latest
    return earliest + int(np.argmin(levels)) * frame + frame // 2


def reference_level(x, rate=TARGET_RATE, percentile=95.0):
    """Typical speech level of a track: percentile of the 100 ms frame RMS.

    Core of the fix for audit finding H2. Speaker attribution no longer
    compares the absolute levels of the two tracks - those depend on the gain
    sliders (-8 dB mic against +10 dB system means an 18 dB systematic bias) -
    but each level relative to that track's OWN reference. Any constant factor
    therefore cancels out.

    The 95th percentile rather than the maximum, so a single cough or mouse
    click cannot move the reference point.
    """
    levels = frame_rms(x, max(1, int(rate * 0.1)))
    if levels.size == 0:
        return rms(x)
    speech = levels[levels > SILENCE_FLOOR]
    if speech.size == 0:
        return 0.0
    return float(np.percentile(speech, percentile))


def segment_rms(x, t_start, t_end, rate=TARGET_RATE):
    """RMS within the time window [t_start, t_end) in seconds."""
    if x is None or len(x) == 0:
        return 0.0
    i_start = max(0, int(t_start * rate))
    i_end = min(len(x), int(math.ceil(t_end * rate)))
    if i_start >= i_end:
        return 0.0
    return rms(x[i_start:i_end])


# ----------------------------------------------------------------------
# Level adjustment
# ----------------------------------------------------------------------
def apply_gain(x, gain_db):
    if gain_db == 0.0:
        return np.asarray(x, dtype=np.float32)
    return (np.asarray(x, dtype=np.float32) * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def limit_peak(x, ceiling=0.95):
    """Only scales down, never up - prevents clipping on write."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak > ceiling and peak > 0:
        return (x * (ceiling / peak)).astype(np.float32)
    return x


def normalize_for_asr(x, target_rms=0.06, ceiling=0.95, reference=None):
    """Bring a track to an even level for speech recognition.

    whisper performs noticeably worse on very quiet material. Because both
    tracks are transcribed separately, each may be normalised independently -
    unlike in the previous version this no longer affects speaker attribution,
    which works on levels relative to each track.

    `reference` overrides the level this piece is measured against. Pass the
    whole track's reference when normalising a section of it: a short quiet
    excerpt measured on its own gets the full +32 dB, which lifts room noise
    to speech level - and whisper answers amplified noise with hallucinations.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if reference is None:
        reference = reference_level(x)
    if reference <= SILENCE_FLOOR:
        return x
    factor = min(target_rms / reference, 40.0)   # at most +32 dB, else noise
    return limit_peak(x * factor, ceiling)
