"""What happens after stop: prepare tracks, transcribe, build the transcript.

Fixes K1, K4 and H5:
  * The saved transcript ALWAYS comes from a complete pass over the recorded
    audio. In the previous version the rolling live preview text was written
    as the final result, which made the ElevenLabs path unreachable (K1) and
    cut the end off the transcript (K4: measured 16.6 % on a real recording).
  * All post-processing ran on the GUI thread. It now runs in a worker and the
    interface stays responsive.

Since then the waiting time itself is attacked: LiveTranscriber recognises the
audio in chunks while it is still being captured, and the closing pass only
has to catch up with the tail. The saved transcript still covers every second
of the recording exactly once - the live segments and the tail meet at
covered_s and are merged through the same diarize pass as before.
"""

import os
import shutil
import threading
from dataclasses import replace

import numpy as np
import soundfile as sf

from . import diarize, paths
from .audio import capture, dsp
from .events import Failed, Finished, Log, Progress, Status
from .paths import TMP_DIR
from .transcribe.base import TranscriptionError, format_timestamp
from .transcribe.elevenlabs import ElevenLabsBackend
from .transcribe.whispercpp import WhisperCppBackend

# --- Live transcription ------------------------------------------------
# whisper works on 30 s windows internally, so a chunk of that length costs no
# context that a single pass would have had.
LIVE_CHUNK_S = 30.0

# The chunk boundary is moved into the quietest spot of these last seconds.
LIVE_SPLIT_SEARCH_S = 3.0

# Run-up the closing pass gets before the point live transcription reached.
# Its segments are dropped again; it only exists so whisper does not start
# cold on a short piece of audio.
LIVE_TAIL_CONTEXT_S = 5.0

# ... and never less than a full whisper window. whisper decodes in 30 s
# windows, so a shorter excerpt is segmented differently from the same audio
# inside a longer file. Measured on a 62 s recording with a 5 s tail: the
# closing "Yeah, tschüssi!" came back as one eight-second segment with the
# wrong text, which the energy filter then dropped as signal-free. With a full
# window it is recognised exactly as a single pass recognises it.
LIVE_TAIL_WINDOW_S = 30.0

# Give up once this much audio has piled up untranscribed: the chosen model is
# slower than real time and the buffer would grow without bound. Nothing is
# lost - the closing pass has the raw tracks on disk and continues at
# covered_s.
LIVE_MAX_PENDING_S = 300.0

# Below this the live window holds too little material to be worth a pass.
LIVE_MIN_PREVIEW_S = 2.0

LIVE_HEADER = ("⚡ Live transcription — these lines are already recognised and "
               "go into the final transcript.")
PREVIEW_HEADER = ("⚡ Live preview of the last 30 seconds — the final "
                  "transcript is produced when you stop.")


def build_backend(settings, greedy=False, live=False):
    """Create the backend matching the current selection."""
    if settings.uses_cloud() and not live:
        return ElevenLabsBackend(api_key=settings.api_key,
                                 model_id=settings.elevenlabs_model_id)
    model = settings.live_model_name() if live else settings.model_name()
    if model is None:
        model = settings.live_model_name()
    return WhisperCppBackend(
        model_name=model,
        threads=settings.threads(),
        use_vad=settings.use_vad,
        allow_gpu=settings.use_gpu,
        greedy=greedy,
    )


def live_transcription_possible(settings):
    """Whether the recording can be transcribed while it runs.

    The cloud backend cannot: every chunk would be a separate paid request,
    and the final transcript has to come from one and the same recogniser -
    mixing whisper chunks with an ElevenLabs tail would show in the text.
    """
    return not settings.uses_cloud()


class _Worker:
    """What the closing pass of a recording and the transcription of an
    uploaded file have in common: a thread, a cancel flag with the backends to
    cancel along with it, and scratch files that are gone when the run ends.

    The scratch files are named by paths.scratch_name() - ASCII only, because
    whisper-cli cannot open a file whose name was typed in Turkish or Arabic -
    and removed when the run ends, whether it succeeded, failed or was
    cancelled. They used to be named after the recording and cleaned up on
    success only (and, with "keep raw tracks", not even then).

    A subclass implements _work(); it is given what run_async() was given.
    """

    thread_name = "worker"
    # Heading of the error shown when anything but a TranscriptionError ends
    # the run - the last line of defence, so the window never stays locked.
    failure_title = "Unexpected error during processing"

    def __init__(self, bridge, settings, backend_factory=None):
        self.bridge = bridge
        self.settings = settings
        # Injectable so the flow can be tested without a real AI backend.
        self.backend_factory = backend_factory or (lambda s: build_backend(s))
        self._backends = []
        self._cancelled = threading.Event()
        self._scratch = []

    def cancel(self):
        self._cancelled.set()
        for backend in list(self._backends):
            try:
                backend.cancel()
            except Exception:
                pass

    def _check_cancelled(self):
        if self._cancelled.is_set():
            raise TranscriptionError("Processing cancelled.")

    # ------------------------------------------------------------------
    def _start(self, *args):
        thread = threading.Thread(target=self._run, args=args,
                                  name=self.thread_name, daemon=True)
        thread.start()
        return thread

    def _run(self, *args):
        try:
            self._process(*args)
        except TranscriptionError as exc:
            self.bridge.post(Failed(message=str(exc)))
        except Exception as exc:                      # last line of defence
            self.bridge.post_exception(self.failure_title, exc)

    def _process(self, *args):
        try:
            self._work(*args)
        finally:
            self._remove_scratch()

    def _work(self, *args):
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _transcribe(self, path, kind):
        backend = self.backend_factory(self.settings)
        self._backends.append(backend)
        try:
            return backend.transcribe(
                path,
                language=self.settings.language,
                log=lambda message: self.bridge.post(Log(message)),
                track=kind,
                progress=lambda message: self.bridge.post(Progress(message)),
            )
        finally:
            if backend in self._backends:
                self._backends.remove(backend)

    def _scratch_path(self, *labels, suffix=".wav"):
        path = os.path.join(TMP_DIR, paths.scratch_name(*labels, suffix=suffix))
        self._scratch.append(path)
        return path

    def _remove_scratch(self):
        for path in self._scratch:
            _try_remove(path)
        self._scratch = []


class Finalizer(_Worker):
    """Runs the post-processing of a recording in a worker thread."""

    thread_name = "finalize"

    def __init__(self, bridge, settings, backend_factory=None, live=None):
        super().__init__(bridge, settings, backend_factory)
        # LiveTranscriber of this recording, if one ran.
        self.live = live

    def cancel(self):
        self._cancelled.set()
        if self.live is not None:
            self.live.cancel()
        super().cancel()

    def run_async(self, recording, base_name):
        return self._start(recording, base_name)

    # ------------------------------------------------------------------
    def _work(self, recording, base_name):
        bridge, settings = self.bridge, self.settings
        out_dir = settings.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)

        for warning in recording.warnings:
            bridge.post(Log(f"Note: {warning}\n"))

        bridge.post(Status("Preparing audio tracks…", "orange"))

        # --- 1. Load raw tracks, align them, resample to 16 kHz ---------
        mic = capture.load_track(recording.mic)
        system = capture.load_track(recording.sys)

        if len(mic) == 0 and len(system) == 0:
            raise TranscriptionError("No audio data was captured.")

        length = max(len(mic), len(system))
        mic = _pad_to(mic, length)
        system = _pad_to(system, length)

        # --- 2. Audible mixdown (with the gain sliders) -----------------
        mix_path = os.path.join(out_dir, f"{base_name}.wav")
        stereo = np.column_stack([
            dsp.limit_peak(dsp.apply_gain(mic, settings.mic_gain_db)),
            dsp.limit_peak(dsp.apply_gain(system, settings.loop_gain_db)),
        ])
        sf.write(mix_path, stereo, dsp.TARGET_RATE, subtype="PCM_16")
        bridge.post(Log(f"Recording saved: {os.path.basename(mix_path)} "
                        f"({length / dsp.TARGET_RATE:.1f} s)\n"))

        # --- 3. Normalise the tracks for recognition --------------------
        # Independent normalisation is safe now: speaker attribution works on
        # levels relative to each track.
        os.makedirs(TMP_DIR, exist_ok=True)

        live_segments, tail_start = self._live_results(recording, length)

        asr_paths = {}
        tail_audio_start = {}
        for kind, audio in (("mic", mic), ("sys", system)):
            # Everything before tail_start has already been recognised while
            # recording; transcribing it again would only duplicate lines. The
            # run-up before the cut is transcribed anyway and thrown away
            # afterwards - whisper started cold on a few seconds of audio
            # hallucinates instead of recognising.
            begin = 0.0
            if tail_start[kind]:
                begin = max(0.0, min(tail_start[kind] - LIVE_TAIL_CONTEXT_S,
                                     len(audio) / float(dsp.TARGET_RATE)
                                     - LIVE_TAIL_WINDOW_S))
            tail_audio_start[kind] = begin
            tail = audio[min(len(audio), int(begin * dsp.TARGET_RATE)):]
            if len(tail) == 0 or dsp.reference_level(tail) <= dsp.SILENCE_FLOOR:
                continue
            path = self._scratch_path(kind, suffix=".asr.wav")
            # Measured against the whole track, not against the excerpt: a
            # quiet tail normalised on its own comes back as amplified noise.
            sf.write(path, dsp.normalize_for_asr(
                tail, reference=dsp.reference_level(audio)),
                dsp.TARGET_RATE, subtype="PCM_16")
            asr_paths[kind] = path

        if not asr_paths and not any(live_segments.values()):
            raise TranscriptionError(
                "Both tracks are practically silent - there is nothing to "
                "transcribe. Are the right devices selected and is system "
                "audio actually playing?")

        # --- 4. Transcription -------------------------------------------
        segments = {kind: list(items) for kind, items in live_segments.items()}
        # Live results are per track, so they only fit the two-pass path - and
        # that path is the better one anyway once the work is already done.
        if settings.separate_tracks or any(live_segments.values()):
            for kind, path in asr_paths.items():
                self._check_cancelled()
                label = "your track" if kind == "mic" else "the system track"
                bridge.post(Status(f"Transcribing {label}…", "purple"))
                bridge.post(Log(f"\n--- Transcription: {label} ---\n"))
                found = _shift_all(self._transcribe(path, kind),
                                   tail_audio_start[kind])
                # Drop the run-up again: those seconds are already in the live
                # segments, and keeping both would print them twice. Decided on
                # the middle of a segment, not its start - on the short tail
                # whisper likes to merge the last sentences into one segment
                # that begins just before the cut, and dropping that on its
                # start swallowed the end of the recording.
                segments[kind].extend(
                    segment for segment in found
                    if (segment.start + segment.end) / 2.0 >= tail_start[kind])
        else:
            # Single pass over the mixdown (faster, but the attribution has to
            # be estimated again).
            bridge.post(Status("Transcribing mixdown…", "purple"))
            merged_path = self._scratch_path("mixdown", suffix=".asr.wav")
            sf.write(merged_path, dsp.normalize_for_asr((mic + system) * 0.5),
                     dsp.TARGET_RATE, subtype="PCM_16")
            for segment in self._transcribe(merged_path, "mic"):
                own = dsp.segment_rms(mic, segment.start, segment.end)
                other = dsp.segment_rms(system, segment.start, segment.end)
                segments["mic" if own >= other else "sys"].append(segment)

        # --- 5. Merge the tracks ----------------------------------------
        bridge.post(Status("Building transcript…", "orange"))
        report = diarize.merge(segments["mic"], segments["sys"],
                               mic_audio=mic, sys_audio=system)
        summary = report.summary()
        if summary:
            bridge.post(Log(f"\nQuality filter: {summary}.\n"))

        text = diarize.render(report)
        if not text.strip():
            text = "[No spoken text was recognised.]"

        txt_path = os.path.join(out_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")

        # --- 6. The recorded originals ------------------------------------
        # The normalised copies made for recognition are scratch files and go
        # in every case (_process removes them, success or not). The recorded
        # tracks are what "keep raw tracks" is about: they move next to the
        # transcript, where a DAW can find them - not into the hidden temp
        # folder, where the next run under the same name would overwrite them
        # and the check for interrupted recordings would take them for one.
        for kind, track in (("mic", recording.mic), ("sys", recording.sys)):
            if track is None or not os.path.exists(track.path):
                continue
            if settings.keep_raw_tracks:
                self._keep_raw_track(track, kind, out_dir, base_name)
            else:
                _try_remove(track.path)

        bridge.post(Finished(text=text, txt_path=txt_path, audio_path=mix_path))

    def _keep_raw_track(self, track, kind, out_dir, base_name):
        label = "microphone track" if kind == "mic" else "system track"
        target = paths.raw_track_path(out_dir, base_name, kind)
        try:
            _move_file(track.path, target)
        except OSError as exc:
            # The transcript is done; failing the run over this would throw
            # that away. The track stays where it was and is found again by
            # the check for unfinished recordings.
            self.bridge.post(Log(f"⚠ The {label} could not be kept ({exc}); "
                                 f"it stays at {track.path}.\n"))
            return
        offset = ""
        if track.start_offset_s > 0.005:
            offset = (f", starts {track.start_offset_s:.2f} s after the other "
                      f"track")
        self.bridge.post(Log(
            f"Kept the {label}: {os.path.basename(target)} "
            f"({track.rate / 1000:g} kHz mono{offset})\n"))

    # ------------------------------------------------------------------
    def _live_results(self, recording, length):
        """What live transcription already produced, on the final timeline.

        Returns (segments per track, point where the closing pass continues).
        Live times are positions in that track's own raw file; load_track()
        prepends the start offset as silence, so both the segments and the cut
        move by exactly that offset.
        """
        segments = {"mic": [], "sys": []}
        tail_start = {"mic": 0.0, "sys": 0.0}
        if self.live is None:
            return segments, tail_start

        self.bridge.post(Status("Finishing live transcription…", "purple"))
        covered, recognised = self.live.finish()

        for kind in segments:
            track = getattr(recording, kind, None)
            offset = track.start_offset_s if track is not None else 0.0
            segments[kind] = _shift_all(recognised.get(kind, []), offset)
            tail_start[kind] = covered.get(kind, 0.0) + offset

        if any(segments.values()):
            # Only tracks that were actually recorded count: a missing one
            # sits at 0 and would make the report look far worse than it is.
            present = [tail_start[kind] for kind in segments
                       if getattr(recording, kind, None) is not None]
            self.bridge.post(Log(
                f"\nLive transcription already covered "
                f"{_clock(min(present) if present else 0.0)} of "
                f"{_clock(length / float(dsp.TARGET_RATE))} - only the rest "
                f"still has to run.\n"))
        return segments, tail_start


# ----------------------------------------------------------------------
class LiveTranscriber:
    """Recognises the running recording chunk by chunk.

    Stopping used to mean waiting for whisper to work through the whole
    recording; this moves that work into the recording itself. The closing
    pass then only handles what is left over - see Finalizer._live_results.

    Timeline: the engine tap hands over exactly the samples that go into the
    raw track file, including the silence the drift correction inserts. A
    sample position therefore means the same instant on both sides, and
    covered_s is a position in that file.

    Deliberately one chunk at a time: two whisper processes would fight over
    the same cores and both finish later than one would.
    """

    def __init__(self, bridge, settings, engine, backend_factory=None,
                 chunk_s=LIVE_CHUNK_S, max_pending_s=LIVE_MAX_PENDING_S,
                 preview=True):
        self.bridge = bridge
        self.settings = settings
        self.engine = engine
        self.backend_factory = backend_factory or (lambda s: build_backend(s))
        self.chunk_s = chunk_s
        self.max_pending_s = max_pending_s
        self.preview = preview

        self.segments = {"mic": [], "sys": []}
        self.covered_s = {"mic": 0.0, "sys": 0.0}

        self._base_name = "live"
        self._reference = {"mic": 0.0, "sys": 0.0}   # worker thread only
        self._pending = {"mic": [], "sys": []}
        self._pending_frames = {"mic": 0, "sys": 0}
        self._rates = {}
        self._turn = 0
        self._chunk_index = 0
        self._sealed = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._cancelled = threading.Event()
        self._backend = None
        self._thread = None

    # ------------------------------------------------------------------
    def start(self, base_name):
        self._base_name = base_name
        self._thread = threading.Thread(target=self._run, name="live-transcribe",
                                        daemon=True)
        self._thread.start()
        self.engine.set_tap(self._on_block)

    def close(self):
        """No more audio arrives - the recording has stopped."""
        self._closed.set()
        self.engine.set_tap(None)
        self._wake.set()

    def cancel(self):
        self._cancelled.set()
        self.close()
        backend = self._backend
        if backend is not None:
            try:
                backend.cancel()
            except Exception:
                pass

    def finish(self, timeout=180.0):
        """Let the chunk in flight finish, then freeze the result.

        Returns (covered_s, segments). Once the finalizer has cut the tail at
        covered_s nothing may move it any more, or tail and live segments would
        cover the same audio twice - hence the seal.
        """
        self.close()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                # Slower than expected: drop what is in flight. The closing
                # pass redoes that stretch, it starts at covered_s anyway.
                self.cancel()
                thread.join(10.0)
        with self._lock:
            self._sealed = True
            return (dict(self.covered_s),
                    {kind: list(items) for kind, items in self.segments.items()})

    # ------------------------------------------------------------------
    def _on_block(self, kind, samples, rate, position=0):
        """Capture thread: buffer only. No disk, no resampling, no blocking."""
        if self._closed.is_set() or kind not in self._pending:
            return
        with self._lock:
            first = kind not in self._rates
            self._rates[kind] = rate
            self._pending[kind].append(samples)
            self._pending_frames[kind] += len(samples)
            backlog = self._pending_frames[kind] / float(rate)

        if first and position:
            # covered_s is handed to the closing pass as "the file is done up
            # to here". That only holds if the tap was attached before the
            # first block was written.
            self._drop("Live transcription attached after the recording had "
                       "already started - the whole recording is transcribed "
                       "after you press stop.")
            return

        if backlog > self.max_pending_s:
            self._drop(f"{self.settings.model_name()} is slower than real time - "
                       f"live transcription stops here, the rest is "
                       f"transcribed after you press stop.")
            return
        self._wake.set()

    def _drop(self, reason):
        """Stop live transcription for this recording.

        Nothing is lost: everything from covered_s on is still on disk in the
        raw tracks and goes through the closing pass.
        """
        if self._closed.is_set():
            return
        self.close()
        with self._lock:
            self._pending = {"mic": [], "sys": []}
            self._pending_frames = {"mic": 0, "sys": 0}
        self.bridge.post(Log(f"\n⚠ {reason}\n"))

    # ------------------------------------------------------------------
    def _run(self):
        try:
            self._backend = self.backend_factory(self.settings)
            prepare = getattr(self._backend, "prepare", None)
            if prepare is not None:
                prepare(log=lambda message: self.bridge.post(Log(message)))
        except Exception as exc:
            self._drop(f"Live transcription unavailable: {exc}")
            return

        while not self._cancelled.is_set():
            chunk = self._next_chunk()
            if chunk is None:
                if self._closed.is_set():
                    return
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            try:
                self._handle_chunk(*chunk)
            except Exception as exc:
                if not self._cancelled.is_set():
                    self._drop(f"Live transcription stopped: {exc}")
                return

    def _next_chunk(self):
        """Cut the next full chunk out of the buffer, or None if none is ready.

        The cut lands on the quietest point of the last seconds instead of a
        fixed sample number, so a boundary does not fall in the middle of a
        word. What is left stays buffered for the next round.
        """
        if self._closed.is_set():
            return None

        with self._lock:
            # Alternate which track is served first, otherwise a recogniser
            # that is barely fast enough would only ever get to the microphone.
            order = ("mic", "sys") if self._turn == 0 else ("sys", "mic")
            for kind in order:
                rate = self._rates.get(kind)
                if not rate or self._pending_frames[kind] < self.chunk_s * rate:
                    continue

                audio = np.concatenate(self._pending[kind])
                target = int(self.chunk_s * rate)
                # Never search back past half a chunk: with a short chunk_s the
                # window would otherwise cover the whole buffer and the quietest
                # frame could sit right at the start, cutting chunks to nothing.
                earliest = max(target // 2,
                               target - int(LIVE_SPLIT_SEARCH_S * rate))
                cut = max(1, dsp.quietest_split(audio, rate, earliest, target))
                remainder = audio[cut:]
                self._pending[kind] = [remainder] if len(remainder) else []
                self._pending_frames[kind] = len(remainder)
                self._turn ^= 1
                return kind, audio[:cut], rate, self.covered_s[kind]
        return None

    def _handle_chunk(self, kind, audio, rate, start_s):
        duration = len(audio) / float(rate)
        mono = dsp.resample(audio, rate, dsp.TARGET_RATE)

        reference = dsp.reference_level(mono)
        if reference <= dsp.SILENCE_FLOOR:
            # Nothing on this track - the normal case for the microphone while
            # the other side is talking. Skip the recogniser entirely.
            self._advance(kind, duration, [])
            return

        # Never amplify a quiet chunk more than the loudest one so far. Judged
        # on its own, a chunk of pure room noise gets the full +32 dB and comes
        # back from whisper as invented speech - and unlike the preview, these
        # segments end up in the saved transcript.
        self._reference[kind] = max(self._reference[kind], reference)

        # Not named after the recording: see paths.scratch_name().
        path = os.path.join(TMP_DIR, paths.scratch_name(
            kind, "live", str(self._chunk_index)))
        self._chunk_index += 1
        os.makedirs(TMP_DIR, exist_ok=True)
        try:
            sf.write(path, dsp.normalize_for_asr(mono,
                                                 reference=self._reference[kind]),
                     dsp.TARGET_RATE, subtype="PCM_16")
            found = self._backend.transcribe(
                path, language=self.settings.language, track=kind)
        finally:
            _try_remove(path)

        self._advance(kind, duration, _shift_all(found, start_s))

    def _advance(self, kind, duration, segments):
        with self._lock:
            if self._sealed:
                return
            self.covered_s[kind] += duration
            self.segments[kind].extend(segments)
        if segments and self.preview:
            self._post_preview()

    def _post_preview(self):
        with self._lock:
            lines = [
                (segment.start,
                 f"{format_timestamp(segment.start)} {label}: {segment.text}")
                for kind, label in (("mic", diarize.LABEL_SELF),
                                    ("sys", diarize.LABEL_OTHER))
                for segment in self.segments[kind]
            ]
        lines.sort()
        self.bridge.post(_preview_event(
            "\n".join(text for _start, text in lines), header=LIVE_HEADER))


# ----------------------------------------------------------------------
class FileFinalizer(_Worker):
    """Runs post-processing and transcription for an uploaded audio file in a worker thread."""

    thread_name = "file-finalize"
    failure_title = "Unexpected error during file processing"

    def run_async(self, file_path, base_name=None):
        if not base_name:
            # The full file name: safe_output_name() removes the one known
            # extension itself. Stripping it here as well cut a second,
            # dot-separated piece off 'Team Meeting 2026.03.10.wav'.
            base_name = paths.safe_output_name(os.path.basename(file_path))
        return self._start(file_path, base_name)

    def _work(self, file_path, base_name):
        bridge, settings = self.bridge, self.settings
        out_dir = settings.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(TMP_DIR, exist_ok=True)

        bridge.post(Status("Loading audio file…", "orange"))
        bridge.post(Log(f"Opening audio file: {os.path.basename(file_path)}…\n"))

        from .audio.loader import load_audio_file
        audio = load_audio_file(file_path, target_rate=dsp.TARGET_RATE)

        if len(audio) == 0 or dsp.reference_level(audio) <= dsp.SILENCE_FLOOR:
            raise TranscriptionError("The selected audio file is silent or contains no audible speech.")

        # Save audio file to output folder as 16 kHz PCM WAV
        mix_path = os.path.join(out_dir, f"{base_name}.wav")
        duration_s = len(audio) / float(dsp.TARGET_RATE)
        if _same_file(file_path, mix_path):
            # The file picked for upload IS the output audio - a recording
            # chosen from the output folder. Converting it would replace it
            # with a mono 16 kHz copy, which for the app's own recordings
            # destroys the two channels (microphone left, system right) they
            # consist of. It stays exactly as it is.
            bridge.post(Log(f"Audio file loaded: {os.path.basename(mix_path)} "
                            f"({duration_s:.1f} s) - left untouched\n"))
        else:
            sf.write(mix_path, dsp.limit_peak(audio), dsp.TARGET_RATE,
                     subtype="PCM_16")
            bridge.post(Log(f"Audio file loaded: {os.path.basename(mix_path)} "
                            f"({duration_s:.1f} s)\n"))

        # Save normalized WAV into TMP_DIR for ASR
        asr_path = self._scratch_path("file", suffix=".asr.wav")
        sf.write(asr_path, dsp.normalize_for_asr(audio), dsp.TARGET_RATE, subtype="PCM_16")

        self._check_cancelled()

        bridge.post(Status("Transcribing audio file…", "purple"))
        bridge.post(Log("\n--- Transcription ---\n"))
        segments = self._transcribe(asr_path, "file")

        self._check_cancelled()

        bridge.post(Status("Building transcript…", "orange"))
        lines = []
        for seg in segments:
            speaker_str = f" {seg.speaker_hint}:" if getattr(seg, "speaker_hint", "") else ""
            lines.append(f"{format_timestamp(seg.start)}{speaker_str} {seg.text}")

        text = "\n".join(lines)
        if not text.strip():
            text = "[No spoken text was recognised.]"

        txt_path = os.path.join(out_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")

        bridge.post(Finished(text=text, txt_path=txt_path, audio_path=mix_path))


# ----------------------------------------------------------------------
class LivePreview:
    """Continuously transcribes only the most recent time window.

    The read-along mode for cases where LiveTranscriber cannot run: with the
    cloud backend, or when live transcription is switched off. Nothing of this
    reaches the saved transcript - it is a display, deliberately fed by a small
    model, and the same audio is recognised properly by the closing pass.

    Fixes K4: the previous version re-transcribed the entire recording so far
    on every pass (quadratic effort). Cost is constant here because only a
    30 second window is processed.
    """

    def __init__(self, bridge, settings, engine, interval_s=6.0):
        self.bridge = bridge
        self.settings = settings
        self.engine = engine
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._backend = None
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="live-preview",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        """Ask the preview to end. Never waits for it.

        This runs on the GUI thread. The thread used to be joined for up to five
        seconds - a frozen window whenever it was in the middle of a model
        download. It is a daemon and everything it can be busy with is
        cancellable now: the whisper process is killed and a running download
        looks at the cancel flag between blocks.
        """
        self._stop.set()
        backend = self._backend
        if backend is not None:
            try:
                backend.cancel()
            except Exception:
                pass
        self._thread = None

    def _loop(self):
        preview_path = os.path.join(TMP_DIR, ".live_preview.wav")
        os.makedirs(TMP_DIR, exist_ok=True)
        self._backend = build_backend(self.settings, greedy=True, live=True)
        if self._stop.is_set():
            return          # stop() came first and had no backend to cancel yet

        try:
            self._backend.prepare(log=lambda m: self.bridge.post(Log(m)))
        except Exception as exc:
            if not self._stop.is_set():          # a cancel is not worth a line
                self.bridge.post(Log(f"Live preview unavailable: {exc}\n"))
            return

        reported = False
        while not self._stop.wait(self.interval_s):
            try:
                window = live_mixdown(*self.engine.live_window())
                if window is None:
                    continue
                mono, mic, system = window
                sf.write(preview_path, mono, dsp.TARGET_RATE, subtype="PCM_16")

                segments = self._backend.transcribe(
                    preview_path, language=self.settings.language, track="mic")
                if self._stop.is_set():
                    break

                lines = []
                for segment in segments:
                    own = dsp.segment_rms(mic, segment.start, segment.end)
                    other = dsp.segment_rms(system, segment.start, segment.end)
                    label = diarize.LABEL_SELF if own >= other else diarize.LABEL_OTHER
                    lines.append(f"{label}: {segment.text}")
                if lines:
                    self.bridge.post(_preview_event("\n".join(lines),
                                                    header=PREVIEW_HEADER))
            except Exception as exc:
                # The preview must never endanger the recording - but silently
                # skipping every pass is what let a broken window calculation
                # look like "the preview does nothing" for so long. Say it once.
                if not reported:
                    reported = True
                    self.bridge.post(Log(f"\n⚠ Live preview failed: {exc}\n"))
                continue

        _try_remove(preview_path)


def _preview_event(text, header=""):
    from .events import LivePreview as LivePreviewEvent
    return LivePreviewEvent(text=text, header=header)


def live_mixdown(mic, system, min_s=LIVE_MIN_PREVIEW_S):
    """Mono mixdown of the two live windows, or None if there is too little.

    Returns (mono, mic, system) with both tracks padded to the same length, so
    the caller can still attribute segments by comparing their levels.

    Watch the emptiness checks: `len(mic or [])` looks harmless but raises
    ValueError on any array with more than one element - the ambiguous truth
    value. That single expression silently disabled the whole live preview,
    because the caller's loop catches every exception and moves on to the next
    pass, where it failed again.
    """
    mic = _as_audio(mic)
    system = _as_audio(system)
    length = max(len(mic), len(system))
    if length < int(dsp.TARGET_RATE * min_s):
        return None

    mic = _pad_to(mic, length)
    system = _pad_to(system, length)
    return dsp.normalize_for_asr((mic + system) * 0.5), mic, system


def _as_audio(audio):
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


def _clock(seconds):
    """'12:30' - format_timestamp without the transcript brackets."""
    return format_timestamp(seconds).strip("[]")


def _shift_all(segments, offset):
    """Move segments onto another timeline (chunk start, track start offset)."""
    if not offset:
        return list(segments)
    return [replace(segment, start=segment.start + offset,
                    end=segment.end + offset) for segment in segments]


# ----------------------------------------------------------------------
def _pad_to(audio, length):
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros(length - len(audio), dtype=np.float32)])


def _try_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _move_file(source, target):
    """Rename, or copy and delete when the target is on another drive."""
    try:
        os.replace(source, target)
    except OSError:
        shutil.copyfile(source, target)
        _try_remove(source)


def _same_file(first, second):
    """True if both paths are one existing file (links, case and short names
    included). False when either does not exist - the normal case for a
    target that is about to be written."""
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False
