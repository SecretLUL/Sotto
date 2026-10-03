"""Chunked reading and resampling: the same samples, a fraction of the memory."""

import os
import shutil
import tempfile
import tracemalloc
import unittest

import numpy as np
import soundfile as sf

from audio_transcriber.audio import dsp, stream

TARGET = dsp.TARGET_RATE


def signal(rate, seconds, seed=1):
    """Two tones plus noise - enough to show a filter edge if there is one."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    clean = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t)
    return (clean + rng.normal(0, 0.05, len(t))).astype(np.float32)


def chunked(x, size):
    for start in range(0, len(x), size):
        yield x[start:start + size]


class TestResampleStream(unittest.TestCase):
    RATES = (44100, 48000, 22050, 11025, 96000, 8000)

    def _stream(self, x, rate, seconds=2.0, context_s=0.25):
        size = dsp.chunk_frames(rate, TARGET, seconds)
        return np.concatenate(list(dsp.resample_stream(
            chunked(x, size), rate, TARGET, context_s=context_s)))

    def test_it_matches_resampling_the_whole_signal_at_once(self):
        for rate in self.RATES:
            with self.subTest(rate=rate):
                x = signal(rate, 9.3)              # not a multiple of the chunk
                expected = dsp.resample(x, rate, TARGET)
                result = self._stream(x, rate)
                self.assertEqual(len(result), len(expected))
                self.assertLess(float(np.max(np.abs(result - expected))), 1e-5)

    def test_there_is_no_click_where_the_chunks_meet(self):
        """resample() warns against calling resample_poly block by block: every
        call filters on its own and leaves a discontinuity at each boundary."""
        rate = 44100
        x = signal(rate, 9.3)
        size = dsp.chunk_frames(rate, TARGET, 2.0)
        expected = dsp.resample(x, rate, TARGET)

        naive = np.concatenate([dsp.resample(chunk, rate, TARGET)
                                for chunk in chunked(x, size)])
        chunked_properly = self._stream(x, rate)

        self.assertGreater(float(np.max(np.abs(naive[:len(expected)] - expected))),
                           1e-3, "the naive way was supposed to click")
        self.assertLess(float(np.max(np.abs(chunked_properly - expected))), 1e-5)

    def test_a_single_short_chunk_works(self):
        x = signal(48000, 0.01)
        np.testing.assert_allclose(
            np.concatenate(list(dsp.resample_stream([x], 48000, TARGET))),
            dsp.resample(x, 48000, TARGET), atol=1e-6)

    def test_the_same_rate_passes_through(self):
        x = signal(TARGET, 3.0)
        result = np.concatenate(list(dsp.resample_stream(chunked(x, 4000), TARGET, TARGET)))
        np.testing.assert_array_equal(result, x)

    def test_nothing_in_nothing_out(self):
        self.assertEqual(list(dsp.resample_stream([], 44100, TARGET)), [])
        self.assertEqual(list(dsp.resample_stream([np.zeros(0)], 44100, TARGET)), [])

    def test_a_chunk_off_the_output_grid_is_refused(self):
        """It would shift every following sample against the whole-signal run."""
        off_grid = [np.zeros(1000), np.zeros(1000)]       # 44100 -> 16000 needs 441s
        with self.assertRaises(ValueError):
            list(dsp.resample_stream(off_grid, 44100, TARGET))

    def test_chunk_frames_are_multiples_of_the_downsampling_factor(self):
        self.assertEqual(dsp.chunk_frames(44100, TARGET, 30.0) % 441, 0)
        self.assertEqual(dsp.chunk_frames(48000, TARGET, 30.0) % 3, 0)
        self.assertGreaterEqual(dsp.chunk_frames(44100, TARGET, 0.001), 441)
        self.assertAlmostEqual(dsp.chunk_frames(48000, TARGET, 30.0) / 48000, 30.0,
                               delta=0.01)


class TestReadMonoResampled(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _old_way(self, path):
        data, rate = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return dsp.resample(data, rate, TARGET)

    def test_a_stereo_file_gives_what_the_old_way_gave(self):
        rate = 44100
        left, right = signal(rate, 12.0, seed=1), signal(rate, 12.0, seed=2)
        path = os.path.join(self.dir, "stereo.wav")
        sf.write(path, np.column_stack([left, right]), rate, subtype="PCM_16")

        result = stream.read_mono_resampled(path, TARGET, chunk_s=2.0)
        expected = self._old_way(path)
        self.assertEqual(len(result), len(expected))
        self.assertLess(float(np.max(np.abs(result - expected))), 1e-5)

    def test_a_mono_file_at_48k_gives_what_the_old_way_gave(self):
        path = os.path.join(self.dir, "mono.wav")
        sf.write(path, signal(48000, 7.7), 48000, subtype="PCM_16")
        result = stream.read_mono_resampled(path, TARGET, chunk_s=2.0)
        expected = self._old_way(path)
        self.assertEqual(len(result), len(expected))
        self.assertLess(float(np.max(np.abs(result - expected))), 1e-5)

    def test_the_default_chunk_size_works_across_several_chunks(self):
        path = os.path.join(self.dir, "long.wav")
        sf.write(path, signal(TARGET, 70.0), TARGET, subtype="PCM_16")
        result = stream.read_mono_resampled(path)
        np.testing.assert_allclose(result, self._old_way(path), atol=1e-6)

    def test_an_empty_file_gives_an_empty_array(self):
        path = os.path.join(self.dir, "empty.wav")
        sf.write(path, np.zeros(0, dtype=np.float32), 44100, subtype="PCM_16")
        self.assertEqual(len(stream.read_mono_resampled(path)), 0)

    def test_a_file_that_is_not_audio_raises_like_soundfile_does(self):
        path = os.path.join(self.dir, "junk.wav")
        with open(path, "wb") as handle:
            handle.write(b"this is not audio")
        with self.assertRaises(Exception):
            stream.read_mono_resampled(path)

    def _write_stereo(self, name, seconds, rate=44100):
        path = os.path.join(self.dir, name)
        rng = np.random.default_rng(3)
        with sf.SoundFile(path, "w", samplerate=rate, channels=2,
                          subtype="PCM_16") as handle:
            for _ in range(seconds):
                handle.write(rng.normal(0, 0.1, (rate, 2)).astype(np.float32))
        return path

    @staticmethod
    def _peak(function):
        """Peak of the memory allocated while `function` runs, and its result."""
        tracemalloc.start()
        try:
            result = function()
            return tracemalloc.get_traced_memory()[1], result
        finally:
            tracemalloc.stop()

    def test_the_memory_needed_does_not_grow_with_the_length_of_the_file(self):
        """Measured on real files: about 1.8 GB per hour of 44.1 kHz stereo for the
        old way, eight times what the result itself takes. Now it is the result
        plus a working set that is the same for a minute and for an hour.

        Small chunks make the working set small against the result, so that a
        second copy of the result (joining the pieces with np.concatenate while
        all of them are still held) would show."""
        short = self._write_stereo("short.wav", 60)
        long = self._write_stereo("long.wav", 240)

        def read(path):
            return lambda: stream.read_mono_resampled(path, chunk_s=2.0)

        short_peak, short_result = self._peak(read(short))
        long_peak, long_result = self._peak(read(long))

        # Three minutes more audio may cost what its resampled samples take, plus
        # a little slack - not the stereo original, the mono mix or a second copy
        # of the result.
        extra_result = long_result.nbytes - short_result.nbytes
        self.assertLess(long_peak - short_peak, 1.25 * extra_result,
                        f"{(long_peak - short_peak) / 1e6:.1f} MB more for "
                        f"{extra_result / 1e6:.1f} MB more result")

    def test_the_default_chunk_size_keeps_the_working_set_small(self):
        path = self._write_stereo("long.wav", 240)
        old_peak, _ = self._peak(lambda: self._old_way(path))
        new_peak, result = self._peak(lambda: stream.read_mono_resampled(path))

        self.assertLess(new_peak - result.nbytes, 40e6,
                        "what is held besides the result should be a few chunks")
        self.assertLess(new_peak, 0.3 * old_peak,
                        f"chunked {new_peak / 1e6:.0f} MB vs whole {old_peak / 1e6:.0f} MB")


class TestAssemble(unittest.TestCase):
    """The result is written into one array sized from what the file announces,
    which for some formats is only an estimate."""

    @staticmethod
    def _parts(*lengths):
        start, parts = 0, []
        for length in lengths:
            parts.append(np.arange(start, start + length, dtype=np.float32))
            start += length
        return parts

    def _expect(self, announced, *lengths):
        result = stream._assemble(iter(self._parts(*lengths)), announced)
        np.testing.assert_array_equal(result, np.arange(sum(lengths), dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)

    def test_an_exact_announcement(self):
        self._expect(13, 5, 5, 3)

    def test_a_file_longer_than_announced_keeps_everything(self):
        self._expect(8, 5, 5, 3)

    def test_a_file_shorter_than_announced_is_cut_to_what_was_there(self):
        self._expect(20, 5, 5, 3)

    def test_a_file_that_announces_nothing(self):
        self._expect(0, 5, 5, 3)

    def test_nothing_gives_an_empty_array(self):
        for announced in (0, 10):
            with self.subTest(announced=announced):
                result = stream._assemble(iter([]), announced)
                self.assertEqual(len(result), 0)
                self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
