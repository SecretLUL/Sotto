"""Reading audio files as mono without holding the whole file in memory.

The loaders used to read the complete file at its native rate, average the
channels into a second copy and resample into a third: about 1.8 GB at the peak
for an hour of 44.1 kHz stereo (eight times the 230 MB the result takes) and
about 1 GB per hour for each raw recording track. A three hour podcast or
meeting did not fit on a modest machine. Here the file is read in chunks and
only the result - which is what has to be kept - grows with its length.
"""

import numpy as np
import soundfile as sf

from . import dsp


def read_mono_resampled(path, target_rate=dsp.TARGET_RATE, chunk_s=10.0):
    """The file as one mono float32 array at `target_rate`.

    The same samples as averaging the channels of the whole file and calling
    dsp.resample() on it (the old way), computed chunk by chunk - see
    dsp.resample_stream() for why that leaves no click at the chunk boundaries.
    The result is allocated once if the file says how long it is, so it is not
    held twice while it is being assembled.

    Raises whatever soundfile raises for a file it cannot read, as sf.read()
    did; the caller decides about a fallback.
    """
    with sf.SoundFile(path) as source:
        rate = source.samplerate
        announced = source.frames if source.frames and source.frames > 0 else 0
        return _assemble(_resampled_chunks(source, target_rate, chunk_s),
                         -(-announced * target_rate // rate))


def iter_mono_resampled(path, target_rate, chunk_s=10.0):
    """The same samples as read_mono_resampled(), handed out chunk by chunk.

    For a consumer that writes them straight on - the listening copy of a
    recording at its full rate - so not even the result is held in memory.
    """
    with sf.SoundFile(path) as source:
        yield from _resampled_chunks(source, target_rate, chunk_s)


def _resampled_chunks(source, target_rate, chunk_s):
    rate = source.samplerate
    frames = dsp.chunk_frames(rate, target_rate, chunk_s)

    def chunks():
        while True:
            block = source.read(frames, dtype="float32", always_2d=True)
            if len(block) == 0:
                return
            yield block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]

    return dsp.resample_stream(chunks(), rate, target_rate)


def _assemble(parts, expected_length):
    """Join the resampled chunks, writing into one array when its size is known.

    Some formats announce a length that is only an estimate; if the audio turns
    out longer, the rest is simply appended, and if it is shorter the array is
    cut to what was there.
    """
    result = np.empty(expected_length, dtype=np.float32) if expected_length else None
    position = 0
    overflow = None
    for part in parts:
        if overflow is not None:
            overflow.append(part)
        elif result is not None and position + len(part) <= len(result):
            result[position:position + len(part)] = part
            position += len(part)
        else:
            overflow = [result[:position] if result is not None
                        else np.zeros(0, dtype=np.float32), part]

    if overflow is not None:
        return np.concatenate(overflow)
    if result is None:
        return np.zeros(0, dtype=np.float32)
    return result if position == len(result) else result[:position].copy()
