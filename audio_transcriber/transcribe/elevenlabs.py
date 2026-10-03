"""ElevenLabs Scribe backend (cloud).

Fixes M8, H8 and the core of K1:
  * The previous version read the entire WAV into RAM and built a bytearray
    from it - roughly 700 MB peak for an hour of audio. The body is streamed
    now.
  * No timeout: a stalled connection blocked the thread indefinitely.
  * The multipart boundary was hard-coded; it is now generated randomly.
  * Words were joined with ' '.join() although the API returns its own
    'spacing' tokens, which produced double spaces and detached punctuation.
    The text is now assembled from the tokens.
  * model_id 'scribe_v2' is attempted first; if the API rejects the model the
    backend falls back to 'scribe_v1' instead of failing without explanation.

Robustness of the request itself:
  * cancel() works at any moment - during the upload and the minutes the server
    needs to transcribe, which is when it matters. It used to take effect only
    once the answer had arrived. The request runs in a helper thread that the
    caller watches, so cancelling does not depend on a blocked socket call
    being woken up (on Windows it is not, reliably); the connection is closed
    as well, which ends the helper early where the platform allows it.
  * HTTP 429, 5xx and connection errors before the upload is complete are
    retried (twice, after a pause). Once the whole file has been sent, a lost
    answer is NOT retried: the file may have been transcribed - and billed -
    already, and sending an hour of audio again is no remedy.
  * The wait for the answer scales with the audio: a socket timeout counts per
    operation, and a fixed 15 minutes is too short for a long file.
  * The upload is FLAC (lossless, documented as supported) instead of WAV and
    roughly half the size; if converting fails the WAV is sent.
"""

import http.client
import json
import os
import secrets
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import soundfile as sf

from .base import Backend, Segment, TranscriptionError

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
CONNECT_TIMEOUT = 30           # connecting, and every single send
RESPONSE_TIMEOUT = 900         # at least this long for the answer ...
RESPONSE_PER_AUDIO_SECOND = 0.5     # ... and half a second per second of audio
RETRY_DELAYS = (2.0, 6.0)      # pauses before the second and third attempt
FALLBACK_MODEL = "scribe_v1"
MAX_WORDS_PER_SEGMENT = 18
SENTENCE_ENDINGS = (".", "?", "!", "…")
PROGRESS_INTERVAL_S = 0.5
CANCEL_POLL_S = 0.2            # how often a waiting request looks for cancel()


class _Retryable(TranscriptionError):
    """A failure that says nothing about the file: try the request again."""


class _ChainedBody:
    """File-like object chaining byte blocks and an open file.

    http.client reads from it block by block, so the audio file never lands in
    memory as a whole. The two hooks let the caller see the upload: on_progress
    gets the number of bytes handed over so far, on_end fires once when the
    last block has been read - and since http.client sends every block before
    it asks for the next one, everything has been sent by then.
    """

    def __init__(self, chunks, on_progress=None, on_end=None):
        self._chunks = list(chunks)
        self._index = 0
        self._handed_over = 0
        self._on_progress = on_progress
        self._on_end = on_end
        self._ended = False

    def read(self, size=-1):
        if size is None or size < 0:
            size = 1 << 20
        data = self._read_block(size)
        if data:
            self._handed_over += len(data)
            if self._on_progress is not None:
                self._on_progress(self._handed_over)
        elif not self._ended:
            self._ended = True
            if self._on_end is not None:
                self._on_end()
        return data

    def _read_block(self, size):
        while self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            if isinstance(chunk, (bytes, bytearray)):
                if chunk:
                    data = bytes(chunk[:size])
                    rest = chunk[size:]
                    self._chunks[self._index] = rest
                    if not rest:
                        self._index += 1
                    return data
                self._index += 1
                continue
            data = chunk.read(size)
            if data:
                return data
            try:
                chunk.close()
            except Exception:
                pass
            self._index += 1
        return b""

    def close(self):
        for chunk in self._chunks:
            if hasattr(chunk, "close"):
                try:
                    chunk.close()
                except Exception:
                    pass


class _TrackingHandlers:
    """urllib handlers that report every connection they open.

    urllib gives no access to the socket before the answer arrives, which made
    cancel() a no-op for the whole upload and the server's processing time.
    These subclasses open exactly the connections urllib would - proxies and
    tunnelling included - but hand them to `track` first.
    """

    def __init__(self, track):
        self.track = track

    def opener(self):
        track = self.track

        class Http(urllib.request.HTTPHandler):
            def http_open(self, request):
                return self.do_open(
                    lambda host, **kw: _tracked(http.client.HTTPConnection,
                                                track, host, **kw), request)

        class Https(urllib.request.HTTPSHandler):
            def https_open(self, request):
                extra = {}
                context = getattr(self, "_context", None)
                if context is not None:
                    extra["context"] = context
                return self.do_open(
                    lambda host, **kw: _tracked(http.client.HTTPSConnection,
                                                track, host, **kw),
                    request, **extra)

        return urllib.request.build_opener(Http(), Https())


def _tracked(connection_class, track, host, **kwargs):
    connection = connection_class(host, **kwargs)
    track(connection)
    return connection


def _to_flac(wav_path, flac_path):
    """Lossless FLAC copy of a WAV file, converted block by block."""
    with sf.SoundFile(wav_path) as source:
        with sf.SoundFile(flac_path, "w", samplerate=source.samplerate,
                          channels=source.channels, format="FLAC",
                          subtype="PCM_16") as target:
            for block in source.blocks(blocksize=1 << 16, dtype="int16"):
                target.write(block)


def _response_timeout(path):
    """How long to wait for the transcript once the whole file is uploaded."""
    try:
        seconds = sf.info(path).duration
    except Exception:
        seconds = os.path.getsize(path) / 32000.0     # 16 kHz mono, 16 bit
    return max(RESPONSE_TIMEOUT, RESPONSE_PER_AUDIO_SECOND * seconds)


class ElevenLabsBackend(Backend):
    name = "ElevenLabs Scribe"

    def __init__(self, api_key, model_id="scribe_v2", diarize=True,
                 tag_audio_events=True):
        self.api_key = (api_key or "").strip()
        self.model_id = model_id
        self.diarize = diarize
        self.tag_audio_events = tag_audio_events
        self._cancel_event = threading.Event()
        self._connection = None     # live connection, so cancel() can close it
        self._upload_done = False
        self._timeout_after_upload = RESPONSE_TIMEOUT

    @property
    def _cancelled(self):
        return self._cancel_event.is_set()

    # ------------------------------------------------------------------
    def transcribe(self, wav_path, language="de", log=None, track="",
                   progress=None):
        if not self.api_key:
            raise TranscriptionError(
                "No ElevenLabs API key is configured. Enter it in the main "
                "window or set the ELEVENLABS_API_KEY environment variable.")
        if not os.path.exists(wav_path):
            raise TranscriptionError(f"Audio file not found: {wav_path}")

        upload_path, upload_name, content_type = self._prepare_upload(wav_path)
        try:
            size_mb = os.path.getsize(upload_path) / (1 << 20)
            self._log(log, f"Uploading {upload_name} ({size_mb:.1f} MB) to "
                           f"ElevenLabs…\n")
            upload = (upload_path, upload_name, content_type)
            try:
                payload = self._send(upload, language, self.model_id, log, progress)
            except TranscriptionError as exc:
                if (self._looks_like_model_error(str(exc))
                        and self.model_id != FALLBACK_MODEL):
                    self._log(log, f"Model '{self.model_id}' was rejected; "
                                   f"trying '{FALLBACK_MODEL}'…\n")
                    payload = self._send(upload, language, FALLBACK_MODEL, log,
                                         progress)
                    self.model_id = FALLBACK_MODEL
                else:
                    raise
        finally:
            if upload_path != wav_path:
                try:
                    os.remove(upload_path)
                except OSError:
                    pass

        return self._to_segments(payload, track)

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_upload(wav_path):
        """(path to send, name to announce, content type): FLAC if possible."""
        base = os.path.basename(wav_path)
        if base.lower().endswith(".wav"):
            handle, flac_path = tempfile.mkstemp(
                suffix=".flac", dir=os.path.dirname(os.path.abspath(wav_path)))
            os.close(handle)
            try:
                _to_flac(wav_path, flac_path)
                return flac_path, base[:-4] + ".flac", "audio/flac"
            except Exception:
                try:
                    os.remove(flac_path)
                except OSError:
                    pass
        return wav_path, base, "audio/wav"

    # ------------------------------------------------------------------
    def _send(self, upload, language, model_id, log, progress):
        """One request, retried where retrying is safe (see the module docs)."""
        attempts = len(RETRY_DELAYS) + 1
        for attempt in range(1, attempts + 1):
            if self._cancelled:
                raise TranscriptionError("Transcription was cancelled.")
            try:
                return self._attempt(upload, language, model_id, progress)
            except _Retryable as exc:
                if attempt == attempts or self._cancelled:
                    raise
                delay = RETRY_DELAYS[attempt - 1]
                self._log(log, f"{exc}\nTrying again in {delay:g} s "
                               f"(attempt {attempt + 1} of {attempts})…\n")
                if self._cancel_event.wait(delay):
                    raise TranscriptionError("Transcription was cancelled.") from exc

    def _attempt(self, upload, language, model_id, progress):
        path, filename, content_type = upload
        boundary = "----AudioTranscriber" + secrets.token_hex(16)
        fields = {
            "model_id": model_id,
            "tag_audio_events": "true" if self.tag_audio_events else "false",
            "diarize": "true" if self.diarize else "false",
        }
        if language and language != "auto":
            fields["language_code"] = language

        prefix = bytearray()
        for name, value in fields.items():
            prefix += f"--{boundary}\r\n".encode()
            prefix += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            prefix += f"{value}\r\n".encode()

        safe_name = filename.replace('"', "_").replace("\r", "").replace("\n", "")
        prefix += f"--{boundary}\r\n".encode()
        prefix += (f'Content-Disposition: form-data; name="file"; '
                   f'filename="{safe_name}"\r\n').encode()
        prefix += f"Content-Type: {content_type}\r\n\r\n".encode()
        suffix = f"\r\n--{boundary}--\r\n".encode()

        content_length = len(prefix) + os.path.getsize(path) + len(suffix)
        timeout_after_upload = _response_timeout(path)
        self._upload_done = False

        reporter = _ProgressReporter(progress, filename, content_length)

        def on_progress(sent):
            # Also catches a cancel() that came before the connection existed.
            if self._cancelled:
                raise OSError("cancelled")
            reporter(sent)

        handle = open(path, "rb")
        body = _ChainedBody([bytes(prefix), handle, suffix],
                            on_progress=on_progress, on_end=self._on_upload_end)

        request = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(content_length),
                "Accept": "application/json",
                "User-Agent": "AudioTranscriber/2.0",
            },
        )
        self._timeout_after_upload = timeout_after_upload
        opener = _TrackingHandlers(self._track).opener()

        # The request runs in a helper thread that this one watches. Closing
        # the socket does not reliably wake a thread that waits on it: with a
        # timeout set, Windows waits in poll(), and a local shutdown() does not
        # end that wait (seen: a cancelled request that stayed blocked until
        # the server answered). So cancel() cannot depend on it - the helper is
        # simply abandoned (it is a daemon and ends when the server answers or
        # the socket times out) and the caller is free at once.
        outcome = {}

        def run():
            try:
                with opener.open(request, timeout=CONNECT_TIMEOUT) as response:
                    outcome["raw"] = response.read().decode("utf-8", errors="replace")
            except BaseException as exc:             # handed over, not lost
                outcome["error"] = exc
            finally:
                body.close()

        worker = threading.Thread(target=run, name="elevenlabs-request",
                                  daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(CANCEL_POLL_S)
            if self._cancelled:
                connection = self._connection
                if connection is not None:
                    self._close(connection)          # frees it where that works
                raise TranscriptionError("Transcription was cancelled.")
        self._connection = None

        error = outcome.get("error")
        if isinstance(error, urllib.error.HTTPError):
            self._raise_for_status(error)
        if isinstance(error, (urllib.error.URLError, OSError,
                              http.client.HTTPException)):
            self._raise_for_failure(error)
        if error is not None:
            raise error

        raw = outcome["raw"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TranscriptionError(
                f"ElevenLabs response was not valid JSON: {raw[:300]}") from exc

    # -- plumbing ----------------------------------------------------------
    def _track(self, connection):
        self._connection = connection
        if self._cancelled:
            self._close(connection)          # cancelled while connecting

    def _on_upload_end(self):
        """Everything has been sent; from here on the server is working."""
        self._upload_done = True
        connection = self._connection
        sock = getattr(connection, "sock", None)
        if sock is not None:
            try:
                sock.settimeout(self._timeout_after_upload)
            except OSError:
                pass

    def _raise_for_status(self, exc):
        detail = exc.read().decode("utf-8", errors="replace")[:1500]
        hint = ""
        if exc.code in (401, 403):
            hint = ("\nHint: the API key was rejected. Is it still valid "
                    "and enabled for speech to text?")
        elif exc.code == 429:
            hint = "\nHint: rate limit reached - try again later."
        message = f"ElevenLabs responded with HTTP {exc.code}:\n{detail}{hint}"
        if exc.code == 429 or exc.code >= 500:
            raise _Retryable(message) from exc      # the file was not processed
        raise TranscriptionError(message) from exc

    def _raise_for_failure(self, exc):
        if self._cancelled:
            raise TranscriptionError("Transcription was cancelled.") from exc
        reason = getattr(exc, "reason", None) or exc
        if not self._upload_done:
            # Not even delivered: nothing was processed, so trying again is safe.
            if isinstance(exc, urllib.error.URLError):
                raise _Retryable(f"Could not reach ElevenLabs: {reason}") from exc
            raise _Retryable(f"Upload failed: {reason}") from exc
        raise TranscriptionError(
            f"ElevenLabs did not answer after the upload ({reason}). The file "
            f"may have been transcribed - and billed - already, so it is not "
            f"sent again automatically.") from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_model_error(message):
        lowered = message.lower()
        return ("model" in lowered
                and any(code in lowered for code in ("422", "400", "404"))) \
            or "model_id" in lowered

    # ------------------------------------------------------------------
    def _to_segments(self, payload, track):
        """Build sentences with timestamps from the word tokens.

        The API returns tokens of type 'word', 'spacing' and 'audio_event'.
        The text is assembled from the raw tokens so punctuation and spacing
        stay correct; segment boundaries appear at sentence punctuation or
        after MAX_WORDS_PER_SEGMENT words.
        """
        words = payload.get("words") or []
        if not words:
            text = (payload.get("text") or "").strip()
            if not text:
                return []
            return [Segment(start=0.0, end=0.0, text=text, track=track)]

        segments = []
        buffer = []
        word_count = 0
        seg_start = None
        seg_end = 0.0
        speaker = ""

        def flush():
            nonlocal buffer, word_count, seg_start, seg_end, speaker
            text = "".join(buffer).strip()
            if text and seg_start is not None:
                segments.append(Segment(start=seg_start,
                                        end=max(seg_end, seg_start),
                                        text=text, track=track,
                                        speaker_hint=speaker))
            buffer = []
            word_count = 0
            seg_start = None
            seg_end = 0.0        # must reset too, else the next segment
            speaker = ""         # inherits this one's end time

        for token in words:
            kind = token.get("type", "word")
            text = token.get("text", "")
            start = token.get("start")
            end = token.get("end")

            if kind == "spacing":
                if buffer:
                    buffer.append(text or " ")
                continue

            if start is not None and seg_start is None:
                seg_start = float(start)
            if end is not None:
                seg_end = float(end)
            if not speaker:
                speaker = str(token.get("speaker_id") or "")

            buffer.append(text)
            if kind == "word":
                word_count += 1

            stripped = text.rstrip()
            if stripped.endswith(SENTENCE_ENDINGS) or word_count >= MAX_WORDS_PER_SEGMENT:
                flush()

        flush()
        return segments

    # ------------------------------------------------------------------
    def cancel(self):
        """Abort a running request, whatever it is doing right now.

        The waiting caller notices within CANCEL_POLL_S (see _attempt). The
        connection is closed too - shutdown() first, because close() alone does
        not wake a blocked thread on every platform - so the helper thread does
        not outlive the cancel for long where that works.
        """
        self._cancel_event.set()
        connection = self._connection
        if connection is not None:
            self._close(connection)

    @staticmethod
    def _close(connection):
        sock = getattr(connection, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            connection.close()
        except Exception:
            pass


class _ProgressReporter:
    """'Uploading x.flac:  45.0 % (12.0/26.7 MB)' - at most twice a second."""

    def __init__(self, progress, filename, total):
        self._progress = progress
        self._filename = filename
        self._total = total
        self._last = 0.0

    def __call__(self, sent):
        if self._progress is None:
            return
        now = time.monotonic()
        if now - self._last < PROGRESS_INTERVAL_S and sent < self._total:
            return
        self._last = now
        self._progress(
            f"Uploading {self._filename}: {sent / self._total * 100:5.1f} % "
            f"({sent / (1 << 20):.1f}/{self._total / (1 << 20):.1f} MB)")
