"""Merging two independently transcribed tracks into one transcript.

Replaces determine_speaker() of the previous version and fixes H2, H3, H7 and
the crosstalk part of the audit.

Why two tracks at all?
    The previous version mixed microphone and system audio into one stereo
    file and ran whisper over it. whisper downmixes to mono internally, so the
    track separation was already lost by the time recognition happened and was
    guessed afterwards from an RMS comparison. Each track is now recognised on
    its own, which makes the attribution known rather than estimated.

What is still decided here:
    * Crosstalk (bleed): your own voice also appears on the system track (and
      the other way round when using speakers). Such segments are discarded
      based on the level relationship between the two tracks.
    * Hallucinations during silence: segments without meaningful energy on
      their own track are dropped.
    * Duplicate recognition: the same sentence on both tracks - the louder
      track wins.

All level comparisons use levels relative to each track. That removes the gain
sliders from the equation entirely - in the previous version -8 dB on the
microphone against +10 dB on the system produced an 18 dB systematic bias in
favour of the other party.
"""

import math
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from .audio import dsp

LABEL_SELF = "[You]"
LABEL_OTHER = "[Participant]"

# A segment counts as crosstalk if it is quieter on its own track than on the
# opposite track by more than this margin (both normalised).
BLEED_MARGIN_DB = 8.0

# Below this fraction of the track's typical speech level a segment counts as
# "no signal" - the typical case of a whisper hallucination during a pause.
SILENCE_FLOOR_RATIO = 0.06

# From this text similarity onwards two overlapping segments count as the same.
DUPLICATE_RATIO = 0.72

# Below this level difference a duplicate counts as undecided.
DUPLICATE_TIE_DB = 3.0

# On a tie the system track wins. Reason: crosstalk only flows one way
# physically. The loopback captures exclusively what the computer plays back,
# so your own voice cannot appear there at all. Sound from speakers, on the
# other hand, does end up in the microphone. If the same sentence shows up on
# both tracks, the source is therefore the system track.
#
# This rule is necessary because the per-track normalisation says nothing
# precisely when a track contains ONLY crosstalk: its own reference level then
# sits at crosstalk level and the ratio is 0 dB.
DUPLICATE_TIE_WINNER = "sys"


@dataclass
class Line:
    start: float
    end: float
    speaker: str
    text: str
    track: str

    def format(self):
        from .transcribe.base import format_timestamp
        return f"{format_timestamp(self.start)} {self.speaker}: {self.text}"


@dataclass
class DiarizationReport:
    lines: list
    dropped_silence: int = 0
    dropped_bleed: int = 0
    dropped_duplicate: int = 0

    def summary(self):
        parts = []
        if self.dropped_silence:
            parts.append(f"{self.dropped_silence} segment(s) without signal dropped")
        if self.dropped_bleed:
            parts.append(f"{self.dropped_bleed} crosstalk segment(s) dropped")
        if self.dropped_duplicate:
            parts.append(f"{self.dropped_duplicate} duplicate(s) merged")
        return "; ".join(parts)


class _TrackContext:
    """Pre-computed level information for one track."""

    def __init__(self, audio, rate=dsp.TARGET_RATE):
        self.audio = audio
        self.rate = rate
        self.reference = dsp.reference_level(audio, rate) if audio is not None else 0.0

    def normalized_db(self, start, end):
        """Segment level in dB relative to THIS track's typical speech level."""
        if self.audio is None or len(self.audio) == 0 or self.reference <= dsp.SILENCE_FLOOR:
            return None
        level = dsp.segment_rms(self.audio, start, end, self.rate)
        if level <= dsp.SILENCE_FLOOR:
            return -120.0
        return 20.0 * math.log10(level / self.reference)

    def ratio(self, start, end):
        if self.audio is None or len(self.audio) == 0 or self.reference <= dsp.SILENCE_FLOOR:
            return None
        return dsp.segment_rms(self.audio, start, end, self.rate) / self.reference

    def span(self, start, end, other=None, bleed_margin_db=BLEED_MARGIN_DB):
        """Where the speech of a segment is on this track (dsp.speech_span),
        leaving out what is only crosstalk from the `other` track."""
        return dsp.speech_span(self.audio, start, end, self.reference, self.rate,
                               other=other.audio if other else None,
                               other_reference=other.reference if other else 0.0,
                               bleed_margin_db=bleed_margin_db)


# ----------------------------------------------------------------------
def merge(mic_segments, sys_segments, mic_audio=None, sys_audio=None,
          rate=dsp.TARGET_RATE, bleed_margin_db=BLEED_MARGIN_DB,
          silence_floor=SILENCE_FLOOR_RATIO, min_segment_s=0.15):
    """Merge both tracks into one chronological transcript."""
    mic_ctx = _TrackContext(mic_audio, rate)
    sys_ctx = _TrackContext(sys_audio, rate)

    report = DiarizationReport(lines=[])
    candidates = []

    for segments, own, other, label, track in (
        (mic_segments or [], mic_ctx, sys_ctx, LABEL_SELF, "mic"),
        (sys_segments or [], sys_ctx, mic_ctx, LABEL_OTHER, "sys"),
    ):
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue

            own_ratio = own.ratio(*_window(segment, min_segment_s))
            if own_ratio is not None and own_ratio < silence_floor:
                report.dropped_silence += 1
                continue

            # From here on the segment is where it is actually spoken. whisper
            # hands a long pause to the sentence after it: the line was dated
            # to the end of the previous one, and a short answer after a long
            # turn of the other party was measured mostly over that turn - and
            # dropped as crosstalk. The silence check above keeps the whole
            # window: narrowed to a cough, a line made up over the pause would
            # pass it. Only the microphone hears the other side, so only there
            # does crosstalk have to be told apart (see DUPLICATE_TIE_WINNER).
            start, end = own.span(segment.start, segment.end,
                                  other if track == "mic" else None,
                                  bleed_margin_db)
            segment = replace(segment, start=start, end=end)
            win_start, win_end = _window(segment, min_segment_s)

            own_db = own.normalized_db(win_start, win_end)
            other_db = other.normalized_db(win_start, win_end)
            if own_db is not None and other_db is not None:
                if own_db < other_db - bleed_margin_db:
                    report.dropped_bleed += 1
                    continue

            candidates.append((segment, label, track,
                               own_db if own_db is not None else 0.0))

    # --- Remove duplicate recognitions ---------------------------------
    candidates.sort(key=lambda item: (item[0].start, item[0].end))
    kept = []
    for candidate in candidates:
        segment, label, track, level = candidate
        duplicate_of = None
        for index, (other_seg, _label, other_track, _level) in enumerate(kept):
            if other_track == track:
                continue
            if not _overlaps(segment, other_seg):
                continue
            if _similar(segment.text, other_seg.text):
                duplicate_of = index
                break
        if duplicate_of is None:
            kept.append(candidate)
            continue

        report.dropped_duplicate += 1
        rival_level = kept[duplicate_of][3]
        rival_track = kept[duplicate_of][2]

        if level - rival_level > DUPLICATE_TIE_DB:
            kept[duplicate_of] = candidate                 # clearly louder
        elif rival_level - level > DUPLICATE_TIE_DB:
            pass                                           # keep what we have
        elif track == DUPLICATE_TIE_WINNER and rival_track != DUPLICATE_TIE_WINNER:
            kept[duplicate_of] = candidate                 # tie -> the source

    # --- Several participants on the system track ----------------------
    hints = {segment.speaker_hint for segment, _label, track, _level in kept
             if track == "sys" and getattr(segment, "speaker_hint", "")}
    hint_names = {}
    if len(hints) > 1:
        for index, hint in enumerate(sorted(hints)):
            hint_names[hint] = f"[Participant {chr(ord('A') + index)}]"

    for segment, label, track, _level in sorted(kept, key=lambda item: item[0].start):
        speaker = label
        hint = getattr(segment, "speaker_hint", "")
        if track == "sys" and hint in hint_names:
            speaker = hint_names[hint]
        report.lines.append(Line(start=segment.start, end=segment.end,
                                 speaker=speaker, text=segment.text.strip(),
                                 track=track))

    return report


def render(report, header=None):
    """Format the report as a text transcript."""
    out = []
    if header:
        out.append(header)
    for line in report.lines:
        out.append(line.format())
    return "\n".join(out)


# ----------------------------------------------------------------------
def _window(segment, min_segment_s):
    """The window a segment's level is measured over. Slightly wider than the
    segment: its boundaries are accurate to about 100 ms, and too narrow a
    window measures the wrong place."""
    return (max(0.0, segment.start - 0.05),
            max(segment.end, segment.start + min_segment_s) + 0.05)


def _overlaps(a, b, tolerance=0.6):
    return a.start < b.end + tolerance and b.start < a.end + tolerance


def _normalize_text(text):
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).split()


def _similar(a, b, threshold=DUPLICATE_RATIO):
    tokens_a, tokens_b = _normalize_text(a), _normalize_text(b)
    if not tokens_a or not tokens_b:
        return False
    return SequenceMatcher(None, " ".join(tokens_a),
                           " ".join(tokens_b)).ratio() >= threshold
