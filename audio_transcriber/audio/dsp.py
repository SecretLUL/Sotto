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

# Telling speech from the noise floor in reference_level()
NOISE_FLOOR_PERCENTILE = 10.0
SPEECH_ABOVE_FLOOR = 10.0 ** (10.0 / 20.0)    # a frame must be 10 dB over the floor
MIN_SPEECH_FRAMES = 10                        # ... and there must be a second of them


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


def chunk_frames(src_rate, dst_rate=TARGET_RATE, seconds=30.0):
    """A chunk length for resample_stream(): about `seconds` of audio, rounded
    to a multiple of the down-sampling factor so every chunk's output lines up
    with the output grid of the whole signal."""
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    down = src_rate // math.gcd(dst_rate, src_rate)
    return max(down, int(seconds * src_rate) // down * down)


def resample_stream(chunks, src_rate, dst_rate=TARGET_RATE, context_s=1.0):
    """Resample consecutive mono chunks, yielding the resampled chunks.

    The result is what resample() gives for the whole signal - up to floating
    point - but only a few chunks are in memory at a time. resample() warns
    against calling resample_poly block by block, and rightly: every call
    filters on its own, which leaves a click at each boundary. Here every chunk
    is filtered together with a second of its neighbours' audio on either side
    and only its own part is kept, so the filter always sees real samples where
    the whole-signal run would. At the very start and end there are no
    neighbours, exactly as in the whole-signal run.

    Every chunk but the last must be a multiple of the down-sampling factor
    long (chunk_frames() gives such a length). That keeps each chunk's output
    on the global output grid; without it the samples would drift against the
    whole-signal result.
    """
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    iterator = iter(chunks)

    def next_chunk():
        for chunk in iterator:
            chunk = np.asarray(chunk, dtype=np.float32)
            if len(chunk):
                return chunk
        return None

    if src_rate == dst_rate:
        chunk = next_chunk()
        while chunk is not None:
            yield chunk
            chunk = next_chunk()
        return

    divisor = math.gcd(dst_rate, src_rate)
    up, down = dst_rate // divisor, src_rate // divisor
    context = max(1, int(context_s * src_rate) // down) * down

    carry = np.zeros(0, dtype=np.float32)
    current = next_chunk()
    while current is not None:
        following = next_chunk()
        last = following is None
        if not last and len(current) % down:
            raise ValueError(f"every chunk but the last must be a multiple of "
                             f"{down} frames long, got {len(current)}")
        right = np.zeros(0, dtype=np.float32) if last else following[:context]
        resampled = sps.resample_poly(np.concatenate([carry, current, right]),
                                      up, down)
        start = len(carry) * up // down
        count = (-(-len(current) * up // down) if last
                 else len(current) * up // down)
        yield resampled[start:start + count].astype(np.float32)
        carry = current[-context:]
        current = following


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

    That percentile breaks down on a track on which little is said - a webinar
    you only listen to, a call in which the other side does the talking. Below
    roughly 5 % speech it lands in the room noise, and the noise then counts as
    "typical speech": normalize_for_asr() lifts it by the full +32 dB to speech
    level and the silence filter in diarize.py can no longer tell it from a
    spoken line, so whisper's hallucinations over the noise survive. The
    reference is therefore never below the median of the frames that stand
    clearly out of the noise floor (10 dB above its 10th percentile, at least
    a second of them). With enough speech that median is lower than the
    percentile and nothing changes; with little speech it replaces it.
    """
    levels = frame_rms(x, max(1, int(rate * 0.1)))
    if levels.size == 0:
        return rms(x)
    levels = levels[levels > SILENCE_FLOOR]
    if levels.size == 0:
        return 0.0
    reference = float(np.percentile(levels, percentile))

    floor = float(np.percentile(levels, NOISE_FLOOR_PERCENTILE))
    speech = levels[levels > floor * SPEECH_ABOVE_FLOOR]
    if speech.size >= MIN_SPEECH_FRAMES:
        reference = max(reference, float(np.median(speech)))
    return reference


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
