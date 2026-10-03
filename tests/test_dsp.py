"""Tests for the signal processing layer."""

import math
import unittest

import numpy as np

from audio_transcriber.audio import dsp


class TestDownmix(unittest.TestCase):
    def test_ignores_silent_surround_channels(self):
        """An 8-channel loopback carrying stereo must not lose 6 dB.

        Regression for the audit finding about the naive mean(axis=1): a real
        loopback device reports 8 channels, six of which are digitally silent
        during stereo playback.
        """
        frames = 4800
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 0.2, frames).astype(np.float32)

        multi = np.zeros((frames, 8), dtype=np.float32)
        multi[:, 0] = signal
        multi[:, 1] = signal
        interleaved = multi.reshape(-1)

        mono = dsp.downmix_active(interleaved, 8)
        naive = multi.mean(axis=1)

        self.assertAlmostEqual(dsp.rms(mono), dsp.rms(signal), places=5)
        # The naive route loses 20*log10(8/2) = 12 dB
        loss_db = 20 * math.log10(dsp.rms(mono) / dsp.rms(naive))
        self.assertAlmostEqual(loss_db, 12.0, places=1)

    def test_channel_activity_is_sticky(self):
        """A channel that once carried signal stays in the mix - otherwise the
        level jumps on every pause in speech."""
        mixer = dsp.ActiveChannelDownmixer(2)
        loud = np.zeros((512, 2), dtype=np.float32)
        loud[:, 0] = 0.5
        loud[:, 1] = 0.5
        mixer.process(loud.reshape(-1))
        self.assertEqual(mixer.active_channels, [0, 1])

        # A block where only channel 0 carries signal
        partial = np.zeros((512, 2), dtype=np.float32)
        partial[:, 0] = 0.5
        mixer.process(partial.reshape(-1))
        self.assertEqual(mixer.active_channels, [0, 1])

    def test_mono_passthrough(self):
        data = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        np.testing.assert_allclose(dsp.downmix_active(data, 1), data)

    def test_incomplete_frame_is_discarded(self):
        # 7 samples across 2 channels -> only 3 complete frames
        data = np.arange(7, dtype=np.float32)
        self.assertEqual(len(dsp.downmix_active(data, 2)), 3)


class TestResample(unittest.TestCase):
    def test_length_and_frequency_preserved(self):
        rate, target, freq, seconds = 48000, 16000, 1000.0, 1.0
        t = np.arange(int(rate * seconds)) / rate
        sine = np.sin(2 * np.pi * freq * t).astype(np.float32)

        out = dsp.resample(sine, rate, target)
        self.assertAlmostEqual(len(out) / target, seconds, places=2)

        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.fft.rfftfreq(len(out), 1 / target)[int(np.argmax(spectrum))]
        self.assertAlmostEqual(peak_hz, freq, delta=5.0)

    def test_44100_to_16000(self):
        out = dsp.resample(np.zeros(44100, dtype=np.float32), 44100, 16000)
        self.assertAlmostEqual(len(out), 16000, delta=2)

    def test_identity(self):
        data = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(dsp.resample(data, 16000, 16000), data)


class TestLevels(unittest.TestCase):
    def _speech_like(self, rate=16000, seconds=10.0, level=0.2):
        """Speech-like signal: loud passages with silence in between."""
        count = int(rate * seconds)
        signal = np.zeros(count, dtype=np.float32)
        rng = np.random.default_rng(1)
        for start in range(0, count, rate * 2):
            end = min(count, start + rate)
            signal[start:end] = rng.normal(0, level, end - start)
        return signal

    def test_reference_level_ignores_silence(self):
        """The reference measures the speech level, not the mean including
        pauses - the basis of the gain-neutral attribution."""
        signal = self._speech_like(level=0.2)
        reference = dsp.reference_level(signal)
        self.assertGreater(reference, 0.15)
        self.assertLess(reference, 0.30)
        # The naive overall RMS would sit much lower because of the pauses
        self.assertLess(dsp.rms(signal), reference)

    def test_reference_level_scales_linearly(self):
        """Core assumption of the diarization: a constant factor (the gain
        slider) shifts segment level and reference level equally, so it
        cancels out in the ratio."""
        signal = self._speech_like()
        for factor in (0.1, 0.5, 2.0, 8.0):
            self.assertAlmostEqual(
                dsp.reference_level(signal * factor) / factor,
                dsp.reference_level(signal),
                places=5)

    def test_segment_rms_window(self):
        rate = 16000
        signal = np.zeros(rate * 4, dtype=np.float32)
        signal[rate:rate * 2] = 0.5
        self.assertAlmostEqual(dsp.segment_rms(signal, 1.0, 2.0, rate), 0.5, places=3)
        self.assertAlmostEqual(dsp.segment_rms(signal, 2.5, 3.5, rate), 0.0, places=6)

    def test_segment_rms_out_of_range(self):
        signal = np.ones(1000, dtype=np.float32)
        self.assertEqual(dsp.segment_rms(signal, 10.0, 11.0, 16000), 0.0)
        self.assertEqual(dsp.segment_rms(None, 0.0, 1.0), 0.0)


def burst_slices(count, speech_share, rate=16000):
    """Where sparse_track() puts its 8 s speech bursts."""
    burst = 8 * rate
    bursts = max(1, int(speech_share * count) // burst)
    starts = [int((k + 0.5) * count / bursts) - burst // 2 for k in range(bursts)]
    return [slice(start, start + burst) for start in starts]


def sparse_track(speech_share, noise_db=-58.0, speech_db=-22.0, minutes=10,
                 rate=16000, seed=3):
    """Constant room noise plus 8 s speech bursts adding up to `speech_share`."""
    rng = np.random.default_rng(seed)
    count = minutes * 60 * rate
    signal = rng.normal(0, 10 ** (noise_db / 20), count).astype(np.float32)
    for window in burst_slices(count, speech_share, rate):
        signal[window] += rng.normal(0, 10 ** (speech_db / 20),
                                     window.stop - window.start).astype(np.float32)
    return signal


class TestReferenceOfSparseTracks(unittest.TestCase):
    """A track on which little is said - you listen to a webinar, the other side
    does the talking.

    The 95th percentile of all frames then lands in the room noise, which
    became the 'typical speech level': the noise was lifted by the full +32 dB
    and could no longer be told apart from speech.
    """

    def test_the_reference_is_the_speech_level_however_little_is_said(self):
        for share in (0.40, 0.20, 0.10, 0.05, 0.03, 0.01):
            with self.subTest(speech_share=share):
                reference_db = 20 * np.log10(dsp.reference_level(sparse_track(share)))
                self.assertAlmostEqual(reference_db, -22.0, delta=3.0)

    def test_speech_ends_up_at_the_target_level_however_little_is_said(self):
        """normalize_for_asr() aims at 0.06 RMS for SPEECH. Measured against the
        noise instead, the gain came out far too high - the closing peak limiter
        then scaled the whole track back down and left the speech at four times
        the target."""
        for share in (0.30, 0.03):
            with self.subTest(speech_share=share):
                signal = sparse_track(share)
                lifted = dsp.normalize_for_asr(signal)
                window = burst_slices(len(signal), share)[0]
                off_target_db = 20 * np.log10(dsp.rms(lifted[window]) / 0.06)
                self.assertAlmostEqual(off_target_db, 0.0, delta=3.0)
                self.assertLess(dsp.rms(lifted[:16000 * 3]), 0.002,
                                "room noise must stay far below the speech")

    def test_dense_speech_keeps_exactly_the_old_reference(self):
        """Nothing changes where the percentile worked: the median of the clear
        speech is below it as soon as there is enough speech."""
        for share in (0.40, 0.60, 0.90):
            with self.subTest(speech_share=share):
                signal = sparse_track(share, minutes=4)
                levels = dsp.frame_rms(signal, 1600)
                self.assertEqual(dsp.reference_level(signal),
                                 float(np.percentile(levels[levels > dsp.SILENCE_FLOOR], 95.0)))

    def test_pure_noise_is_left_alone(self):
        noise = np.random.default_rng(4).normal(0, 0.01, 16000 * 30).astype(np.float32)
        self.assertGreater(dsp.reference_level(noise), 0.0095)
        self.assertLess(dsp.reference_level(noise), 0.0108)

    def test_a_single_click_does_not_make_a_track_speech(self):
        """Fewer than a second of loud frames is a cough or a mouse click."""
        rng = np.random.default_rng(5)
        signal = rng.normal(0, 0.001, 16000 * 60).astype(np.float32)
        signal[16000 * 10:16000 * 10 + 4800] += 0.5          # three frames
        self.assertLess(dsp.reference_level(signal), 0.01)


class TestGain(unittest.TestCase):
    def test_limit_peak_never_amplifies(self):
        quiet = np.full(100, 0.1, dtype=np.float32)
        np.testing.assert_allclose(dsp.limit_peak(quiet), quiet)

    def test_limit_peak_reduces_clipping(self):
        loud = np.full(100, 2.0, dtype=np.float32)
        self.assertAlmostEqual(float(np.max(dsp.limit_peak(loud))), 0.95, places=5)

    def test_apply_gain_db(self):
        data = np.ones(10, dtype=np.float32)
        np.testing.assert_allclose(dsp.apply_gain(data, 6.0), 1.995, rtol=1e-3)
        np.testing.assert_allclose(dsp.apply_gain(data, -6.0), 0.5012, rtol=1e-3)

    def test_normalize_for_asr_raises_quiet_track(self):
        rng = np.random.default_rng(2)
        quiet = rng.normal(0, 0.004, 16000 * 3).astype(np.float32)
        out = dsp.normalize_for_asr(quiet, target_rms=0.06)
        self.assertGreater(dsp.reference_level(out), 0.04)
        self.assertLessEqual(float(np.max(np.abs(out))), 0.95 + 1e-6)

    def test_normalize_leaves_digital_silence_alone(self):
        silence = np.zeros(16000, dtype=np.float32)
        np.testing.assert_array_equal(dsp.normalize_for_asr(silence), silence)

    def test_a_given_reference_replaces_the_measured_one(self):
        """An excerpt normalised on its own gets a gain of its own.

        Live transcription hands the closing pass a tail of a few seconds.
        Measured alone, a quiet tail is lifted to full speech level - whisper
        answers amplified room noise with invented sentences. Passing the whole
        track's reference gives the excerpt exactly the track's gain.
        """
        rng = np.random.default_rng(3)
        loud = rng.normal(0, 0.2, 16000 * 3).astype(np.float32)
        quiet_tail = rng.normal(0, 0.004, 16000).astype(np.float32)

        alone = dsp.normalize_for_asr(quiet_tail)
        with_track = dsp.normalize_for_asr(
            quiet_tail, reference=dsp.reference_level(loud))

        self.assertGreater(dsp.reference_level(alone),
                           10 * dsp.reference_level(with_track))
        # Exactly the gain the given reference asks for - no measuring of its
        # own, so the excerpt keeps its place in the track's dynamics.
        factor = 0.06 / dsp.reference_level(loud)
        np.testing.assert_allclose(with_track, quiet_tail * factor, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
