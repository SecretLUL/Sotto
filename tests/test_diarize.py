"""Tests for speaker attribution (audit finding H2 and crosstalk)."""

import unittest

import numpy as np

from audio_transcriber import diarize
from audio_transcriber.audio import dsp
from audio_transcriber.transcribe.base import Segment

RATE = dsp.TARGET_RATE


def track(bursts, seconds=20.0, rate=RATE, seed=0):
    """Build a track of noise bursts: bursts = [(start_s, end_s, amplitude)]."""
    rng = np.random.default_rng(seed)
    signal = np.zeros(int(seconds * rate), dtype=np.float32)
    for start, end, amplitude in bursts:
        i_start, i_end = int(start * rate), int(end * rate)
        signal[i_start:i_end] = rng.normal(0, amplitude, i_end - i_start)
    return signal


class TestGainInvariance(unittest.TestCase):
    """The core of audit finding H2.

    In the previous version determine_speaker() compared the absolute levels
    of both tracks AFTER the gain sliders had been applied. With the settings
    actually found on disk (mic -8 dB, system +10 dB) that produced an 18 dB
    bias in favour of the other party.
    """

    def setUp(self):
        # You speak from 1-3 s, the other party from 5-7 s.
        self.mic = track([(1.0, 3.0, 0.20)], seed=1)
        self.sys = track([(5.0, 7.0, 0.20)], seed=2)
        self.mic_segments = [Segment(1.0, 3.0, "Hello, how are you?", "mic")]
        self.sys_segments = [Segment(5.0, 7.0, "Thanks, doing well.", "sys")]

    def _labels(self, mic_factor=1.0, sys_factor=1.0):
        report = diarize.merge(self.mic_segments, self.sys_segments,
                               mic_audio=self.mic * mic_factor,
                               sys_audio=self.sys * sys_factor)
        return [(line.speaker, line.text) for line in report.lines]

    def test_baseline(self):
        self.assertEqual(self._labels(), [
            (diarize.LABEL_SELF, "Hello, how are you?"),
            (diarize.LABEL_OTHER, "Thanks, doing well."),
        ])

    def test_result_is_independent_of_gain_settings(self):
        """Exactly the configuration found in the original settings.json:
        mic_gain_db = -8.0, loop_gain_db = +10.0."""
        expected = self._labels()
        mic_factor = 10 ** (-8.0 / 20)
        sys_factor = 10 ** (10.0 / 20)
        self.assertEqual(self._labels(mic_factor, sys_factor), expected)

    def test_extreme_gain_difference_still_correct(self):
        expected = self._labels()
        for mic_factor, sys_factor in [(0.01, 10.0), (10.0, 0.01),
                                       (0.1, 0.1), (5.0, 5.0)]:
            with self.subTest(mic=mic_factor, sys=sys_factor):
                self.assertEqual(self._labels(mic_factor, sys_factor), expected)


class TestSparseTrack(unittest.TestCase):
    """You listen, the other side talks: the microphone carries room noise and a
    couple of spoken lines. Whisper makes up sentences over such noise; whether
    the filter drops them used to depend on how much was said on that track."""

    def test_made_up_lines_over_the_noise_go_and_spoken_ones_stay(self):
        seconds = 600
        rng = np.random.default_rng(7)
        mic = rng.normal(0, 10 ** (-58 / 20), seconds * RATE).astype(np.float32)
        for start, end in ((150.0, 156.0), (450.0, 456.0)):    # 2 % speech
            i_start, i_end = int(start * RATE), int(end * RATE)
            mic[i_start:i_end] += rng.normal(0, 10 ** (-22 / 20),
                                             i_end - i_start).astype(np.float32)
        system = track([(50.0, 90.0, 0.2)], seconds=seconds, seed=2)

        mic_segments = [
            Segment(150.2, 155.8, "Yes, I agree with that.", "mic"),
            Segment(300.0, 303.0, "Thank you for watching.", "mic"),   # made up
            Segment(450.2, 455.8, "One more question.", "mic"),
        ]
        sys_segments = [Segment(50.5, 89.0, "The other side talking.", "sys")]

        report = diarize.merge(mic_segments, sys_segments,
                               mic_audio=mic, sys_audio=system)

        mine = [line.text for line in report.lines if line.track == "mic"]
        self.assertEqual(mine, ["Yes, I agree with that.", "One more question."])
        self.assertEqual(report.dropped_silence, 1)


class TestBleedSuppression(unittest.TestCase):
    def test_speaker_bleed_into_microphone_is_dropped(self):
        """Speaker playback: the other party's voice reaches the microphone
        quietly and is recognised there a second time.

        This is the only physically possible crosstalk direction - the
        loopback only captures what the computer plays back, so your own voice
        cannot show up there.
        """
        # Both tracks also contain genuine speech so that the per-track
        # reference level reflects the actual speech level.
        mic = track([(1.0, 3.0, 0.25), (6.0, 8.0, 0.02)], seed=3)
        sys = track([(6.0, 8.0, 0.22)], seed=4)
        report = diarize.merge(
            [Segment(1.0, 3.0, "This is my own statement.", "mic"),
             Segment(6.0, 8.0, "This is what the other person says.", "mic")],
            [Segment(6.0, 8.0, "This is what the other person says.", "sys")],
            mic_audio=mic, sys_audio=sys)

        self.assertEqual(len(report.lines), 2)
        self.assertEqual(report.lines[0].speaker, diarize.LABEL_SELF)
        self.assertEqual(report.lines[1].speaker, diarize.LABEL_OTHER)
        self.assertGreaterEqual(report.dropped_bleed + report.dropped_duplicate, 1)

    def test_tie_break_favours_system_track(self):
        """If a track contains nothing but crosstalk, its own reference level
        sits at crosstalk level and the normalisation says nothing. The
        physical crosstalk direction then decides."""
        mic = track([(1.0, 3.0, 0.02)], seed=5)      # crosstalk only
        sys = track([(1.0, 3.0, 0.25)], seed=6)
        report = diarize.merge(
            [Segment(1.0, 3.0, "Identical sentence.", "mic")],
            [Segment(1.0, 3.0, "Identical sentence.", "sys")],
            mic_audio=mic, sys_audio=sys)

        self.assertEqual(len(report.lines), 1)
        self.assertEqual(report.lines[0].speaker, diarize.LABEL_OTHER)

    def test_simultaneous_speech_keeps_both(self):
        """If both speak at once at comparable level with different text, both
        must survive."""
        mic = track([(2.0, 4.0, 0.20)], seed=5)
        sys = track([(2.0, 4.0, 0.18)], seed=6)
        report = diarize.merge(
            [Segment(2.0, 4.0, "I do not think that is right.", "mic")],
            [Segment(2.0, 4.0, "Hold on, I have to disagree there.", "sys")],
            mic_audio=mic, sys_audio=sys)
        self.assertEqual(len(report.lines), 2)
        self.assertEqual({line.speaker for line in report.lines},
                         {diarize.LABEL_SELF, diarize.LABEL_OTHER})


class TestHallucinationFilter(unittest.TestCase):
    def test_segment_without_signal_is_dropped(self):
        """whisper likes to invent text during pauses ('Subtitles by ...').
        Without energy on its own track the segment is dropped."""
        mic = track([(1.0, 3.0, 0.25)], seed=7)
        sys = np.zeros(int(20 * RATE), dtype=np.float32)
        report = diarize.merge(
            [Segment(1.0, 3.0, "A real sentence.", "mic"),
             Segment(12.0, 14.0, "Subtitles by the community", "mic")],
            [], mic_audio=mic, sys_audio=sys)

        self.assertEqual([line.text for line in report.lines], ["A real sentence."])
        self.assertEqual(report.dropped_silence, 1)

    def test_without_audio_nothing_is_filtered(self):
        """Without a reference signal the filter must not fire blindly."""
        report = diarize.merge(
            [Segment(0.0, 1.0, "A", "mic")], [Segment(2.0, 3.0, "B", "sys")])
        self.assertEqual(len(report.lines), 2)
        self.assertEqual(report.dropped_silence, 0)


class TestWhereItIsSpoken(unittest.TestCase):
    """whisper hands the silence before a sentence to that sentence.

    Found in a real recording: a video started 12 s in and its first line came
    back as 0.00 -> 14.14 s, and the answer after it was dated to the end of
    the line before it - 13 s instead of 21 s.
    """

    def test_a_line_is_dated_where_it_is_spoken(self):
        sys = track([(12.0, 14.0, 0.2), (15.0, 19.0, 0.2)], seconds=25.0, seed=12)
        report = diarize.merge(
            [], [Segment(0.0, 14.14, "My last ace up the sleeve.", "sys")],
            mic_audio=np.zeros_like(sys), sys_audio=sys)
        self.assertEqual(len(report.lines), 1)
        self.assertGreaterEqual(report.lines[0].start, 11.85)
        self.assertLessEqual(report.lines[0].start, 12.0)

    def test_a_short_answer_after_a_long_turn_is_not_crosstalk(self):
        """Measured over the whole window it was mostly the other side's turn,
        so a genuine answer was dropped as crosstalk."""
        mic = track([(18.5, 19.0, 0.2), (25.0, 28.0, 0.2)], seconds=30.0, seed=13)
        sys = track([(0.5, 18.0, 0.2)], seconds=30.0, seed=14)
        report = diarize.merge(
            [Segment(0.0, 19.2, "Yes.", "mic")],
            [Segment(0.5, 18.0, "A long explanation.", "sys")],
            mic_audio=mic, sys_audio=sys)
        self.assertEqual(report.dropped_bleed, 0)
        yes = [line for line in report.lines if line.text == "Yes."]
        self.assertEqual(len(yes), 1)
        self.assertAlmostEqual(yes[0].start, 18.4, delta=0.1)

    def test_crosstalk_is_not_where_your_line_starts(self):
        """A sensitive microphone hears the headphones: the video sits on the
        microphone track too, quietly, while you say nothing."""
        mic = track([(1.0, 10.0, 0.2), (12.0, 20.0, 0.2 * 10 ** (-16 / 20)),
                     (21.5, 25.5, 0.2)], seconds=30.0, seed=15)
        sys = track([(12.0, 20.0, 0.2)], seconds=30.0, seed=16)
        report = diarize.merge(
            [Segment(12.97, 26.09, "Did that work?", "mic")],
            [Segment(12.0, 20.0, "The video.", "sys")],
            mic_audio=mic, sys_audio=sys)
        line = [line for line in report.lines if line.track == "mic"][0]
        self.assertAlmostEqual(line.start, 21.4, delta=0.1)

    def test_a_click_is_not_where_a_line_starts(self):
        mic = track([(5.0, 5.05, 0.3), (8.0, 10.0, 0.2)], seed=17)
        report = diarize.merge([Segment(4.0, 10.0, "After the click.", "mic")], [],
                               mic_audio=mic, sys_audio=np.zeros_like(mic))
        self.assertAlmostEqual(report.lines[0].start, 7.9, delta=0.1)

    def test_a_line_made_up_over_a_pause_is_still_dropped(self):
        """The silence check measures the whole window. Narrowed to the cough
        in it, a line whisper invented over the pause would pass."""
        mic = track([(1.0, 3.0, 0.2), (10.0, 10.3, 0.2 * 10 ** (-15 / 20))],
                    seed=18)
        report = diarize.merge(
            [Segment(1.0, 3.0, "A real sentence.", "mic"),
             Segment(5.0, 15.0, "Thank you for watching.", "mic")],
            [], mic_audio=mic, sys_audio=np.zeros_like(mic))
        self.assertEqual([line.text for line in report.lines], ["A real sentence."])
        self.assertEqual(report.dropped_silence, 1)


class TestOrderingAndRendering(unittest.TestCase):
    def test_chronological_order(self):
        mic = track([(1.0, 2.0, 0.2), (7.0, 8.0, 0.2)], seed=8)
        sys = track([(4.0, 5.0, 0.2)], seed=9)
        report = diarize.merge(
            [Segment(1.0, 2.0, "One", "mic"), Segment(7.0, 8.0, "Three", "mic")],
            [Segment(4.0, 5.0, "Two", "sys")],
            mic_audio=mic, sys_audio=sys)
        self.assertEqual([line.text for line in report.lines], ["One", "Two", "Three"])

    def test_render_format(self):
        mic = track([(65.0, 67.0, 0.2)], seconds=80.0, seed=10)
        report = diarize.merge([Segment(65.0, 67.0, "After one minute", "mic")],
                               [], mic_audio=mic,
                               sys_audio=np.zeros(int(80 * RATE), dtype=np.float32))
        self.assertEqual(diarize.render(report), "[01:05] [You]: After one minute")

    def test_multiple_participants_on_system_track(self):
        """When ElevenLabs supplies speaker ids on the system track the
        participants are told apart instead of collapsing into one speaker."""
        sys = track([(1.0, 2.0, 0.2), (3.0, 4.0, 0.2)], seed=11)
        first = Segment(1.0, 2.0, "My name is Anna.", "sys", speaker_hint="speaker_0")
        second = Segment(3.0, 4.0, "And I am Ben.", "sys", speaker_hint="speaker_1")
        report = diarize.merge([], [first, second], mic_audio=None, sys_audio=sys)
        self.assertEqual([line.speaker for line in report.lines],
                         ["[Participant A]", "[Participant B]"])


if __name__ == "__main__":
    unittest.main()
