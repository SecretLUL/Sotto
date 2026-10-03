"""whisper.cpp backend (local, subprocess).

Fixes H10, H9, M5, M12:
  * stderr was opened as a PIPE but never read. That works as long as whisper
    logs little (measured: 2872 bytes); once the 64 KB pipe buffer is full the
    child blocks on write and the parent waits forever on stdout - the classic
    pipe deadlock. stderr is now always drained in parallel.
  * The process was never terminated on stop and kept running in the
    background. There is now cancel().
  * '-t' was never set: whisper used 4 of 12 cores. Measured on the
    development machine (79 s of audio, model small): -t 4 = 15.6 s,
    -t 10 = 10.7 s.
  * '-np' keeps diagnostic output out of the transcript.

On '--vad': measured against a real recording, the VAD path of this build
merges speech regions that are far apart into a single segment (microphone
track: 9 instead of 21 segments, the first spanning 1.79 s to 41.83 s) and
loses the last ~20 seconds. Accurate timestamps are decisive for speaker
attribution, so VAD is off by default. Hallucinations during pauses are
filtered by diarize.py using track energy instead.

On '-ng': the Vulkan backend of this build crashes reproducibly with
0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) on the AMD GPU it was tested on -
even with -nfa and without beam search. GPU use therefore stays disabled by
default; the switch is exposed through allow_gpu once a working build exists.
"""

import os
import subprocess
import threading

from .. import paths
from . import binaries
from .base import Backend, Segment, TranscriptionError, parse_line

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Number of identical consecutive segments before truncating. Two are kept
# because "yes, yes" can be genuine speech (N8).
MAX_CONSECUTIVE_REPEATS = 2


class WhisperCppBackend(Backend):
    name = "whisper.cpp"

    def __init__(self, model_name, threads=None, use_vad=False,
                 allow_gpu=False, greedy=False, max_len=45):
        self.model_name = model_name
        self.threads = threads
        self.use_vad = use_vad
        self.allow_gpu = allow_gpu
        self.greedy = greedy
        self.max_len = max_len
        self._proc = None
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def prepare(self, progress=None, log=None):
        """Fetch the binary, the model and optionally the VAD model."""
        exe = binaries.ensure_whisper_binary(progress=progress, log=log)
        model = binaries.ensure_model(self.model_name, progress=progress, log=log)
        vad = None
        if self.use_vad:
            try:
                vad = binaries.ensure_vad_model(progress=progress, log=log)
            except binaries.DownloadError as exc:
                # VAD is an improvement, not a requirement - do not fail on it.
                self.use_vad = False
                self._log(log, f"Note: VAD model unavailable ({exc}). "
                               f"Continuing without VAD.\n")
        return exe, model, vad

    # ------------------------------------------------------------------
    def build_command(self, exe, model, wav_path, language, vad_model=None):
        command = [
            exe,
            "-m", model,
            "-f", wav_path,
            "-l", language,
            "-t", str(self.threads or 4),
            "-ml", str(self.max_len),
            "-sow",
            "-np",          # results only on stdout
            "-sns",         # suppress non-speech tokens
        ]
        if not self.allow_gpu:
            command.append("-ng")
        if self.greedy:
            # Live preview: greedy instead of beam search. Measured 10.7 s -> 8.3 s.
            command += ["-bs", "1", "-bo", "1"]
        if self.use_vad and vad_model:
            command += ["--vad", "-vm", vad_model]
        return command

    # ------------------------------------------------------------------
    def transcribe(self, wav_path, language="de", log=None, track="",
                   progress=None):
        if not os.path.exists(wav_path):
            raise TranscriptionError(f"Audio file not found: {wav_path}")

        self._cancelled.clear()
        exe, model, vad = self.prepare(progress=progress, log=log)
        # whisper-cli.exe reads its arguments in the ANSI code page; a folder
        # called after a Turkish or Arabic user would make it fail with "input
        # file not found" (see paths.ansi_safe_path). The executable itself
        # is started through the wide-character API and needs no such care.
        command = self.build_command(
            exe, paths.ansi_safe_path(model), paths.ansi_safe_path(wav_path),
            language, paths.ansi_safe_path(vad) if vad else None)

        try:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise TranscriptionError(
                f"whisper-cli.exe could not be started: {exc}") from exc

        with self._lock:
            self._proc = proc

        # ALWAYS drain stderr in parallel, otherwise deadlock (H10).
        stderr_lines = []

        def drain_stderr():
            try:
                for line in proc.stderr:
                    line = line.strip()
                    if line:
                        stderr_lines.append(line)
            except Exception:
                pass

        err_thread = threading.Thread(target=drain_stderr, daemon=True,
                                      name="whisper-stderr")
        err_thread.start()

        segments = []
        last_text = None
        repeats = 0
        try:
            for line in proc.stdout:
                parsed = parse_line(line)
                if parsed is None:
                    continue
                start, end, text = parsed
                if not text:
                    continue

                if text == last_text:
                    repeats += 1
                    if repeats >= MAX_CONSECUTIVE_REPEATS:
                        continue        # decoder repetition loop
                else:
                    last_text = text
                    repeats = 0

                segments.append(Segment(start=start, end=end, text=text, track=track))
        finally:
            proc.wait()
            err_thread.join(timeout=2.0)
            # Close the pipes explicitly: the live preview starts this process
            # every minute, so open handles would pile up over a session.
            for pipe in (proc.stdout, proc.stderr):
                try:
                    pipe.close()
                except Exception:
                    pass
            with self._lock:
                self._proc = None

        if self._cancelled.is_set():
            raise TranscriptionError("Transcription was cancelled.")

        if proc.returncode != 0:
            detail = "\n".join(stderr_lines[-12:]) or f"Exit code {proc.returncode}"
            raise TranscriptionError(
                f"whisper.cpp failed with code {proc.returncode}:\n{detail}")

        return segments

    # ------------------------------------------------------------------
    def cancel(self):
        """Terminate a running subprocess (e.g. when the recording stops)."""
        self._cancelled.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass
