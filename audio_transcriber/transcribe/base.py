"""Shared interface for both transcription backends."""

import abc
import re
from dataclasses import dataclass

# Accepts MM:SS(.mmm) and HH:MM:SS(.mmm).
# Fixes H7: the previous version discarded the milliseconds through a
# non-capturing group, so the analysis window for speaker attribution could be
# off by up to a full second. They are now evaluated.
TIMESTAMP_RE = re.compile(
    r'^\s*\['
    r'(?P<sh>\d{1,3}):(?P<sm>\d{2})(?::(?P<ss>\d{2}))?(?P<sf>\.\d+)?'
    r'\s*-->\s*'
    r'(?P<eh>\d{1,3}):(?P<em>\d{2})(?::(?P<es>\d{2}))?(?P<ef>\.\d+)?'
    r'\]\s*(?P<text>.*)$'
)


@dataclass
class Segment:
    """One recognised speech segment from a single track."""
    start: float
    end: float
    text: str
    track: str = ""          # "mic" | "sys"
    # Speaker id within a track, as returned by ElevenLabs. Several people can
    # speak on the system track; the hint allows telling them apart.
    speaker_hint: str = ""

    @property
    def duration(self):
        return max(0.0, self.end - self.start)


def parse_timestamp(hours, minutes, seconds, fraction):
    """whisper.cpp emits HH:MM:SS.mmm; shorter forms occur in other builds.
    Without a seconds field the first two groups are MM:SS."""
    if seconds is None:
        total = int(hours) * 60 + int(minutes)
    else:
        total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return total + (float(fraction) if fraction else 0.0)


def parse_line(line):
    """Turn a whisper output line into (start, end, text) or None."""
    match = TIMESTAMP_RE.match(line.strip())
    if not match:
        return None
    groups = match.groupdict()
    start = parse_timestamp(groups["sh"], groups["sm"], groups["ss"], groups["sf"])
    end = parse_timestamp(groups["eh"], groups["em"], groups["es"], groups["ef"])
    return start, end, groups["text"].strip()


def format_timestamp(seconds):
    """[MM:SS], or [HH:MM:SS] from one hour onwards.

    Fixes N4: the previous version computed mins = t // 60 and formatted with
    02d, so a 90 minute recording ended at [90:12] instead of [01:30:12].
    """
    total = int(max(0, seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def format_clock(seconds):
    """'12:30', or '01:02:03' from one hour onwards - without the brackets."""
    return format_timestamp(seconds)[1:-1]


class TranscriptionError(RuntimeError):
    pass


class Backend(abc.ABC):
    """A backend transcribes ONE mono track and returns segments.

    Speaker attribution is deliberately not the backend's job - it happens in
    diarize.py based on two independently transcribed tracks.
    """

    name = "backend"

    @abc.abstractmethod
    def transcribe(self, wav_path, language="de", log=None, track=""):
        """Returns list[Segment]. Raises TranscriptionError on failure."""

    @staticmethod
    def _log(log, message):
        if log is not None:
            log(message)
