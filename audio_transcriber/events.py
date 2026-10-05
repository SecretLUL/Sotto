"""Bridge from worker threads to the Tkinter interface.

Fixes H4/K2 structurally. The previous version scattered roughly 15 calls of
    self.root.after(0, lambda: ...)
across the code, which produced two classes of failure:

  1. Closures over the except variable 'e' - Python deletes it at the end of
     the block, the lambda ran later and raised NameError instead of showing
     the error message.
  2. Workers read Tk widgets directly (model_dropdown.get()), which is not
     thread-safe in Tcl/Tk.

There is now exactly one channel: workers put immutable event objects into a
queue, a single pump on the GUI thread processes them. Only data crosses the
thread boundary, never a callable carrying closure state.
"""

import queue
import traceback
from dataclasses import dataclass


# --- Event types -------------------------------------------------------
@dataclass(frozen=True)
class Log:
    """A line for the transcript pane."""
    text: str


@dataclass(frozen=True)
class Status:
    """Status line: text plus colour."""
    text: str
    color: str = "gray"


@dataclass(frozen=True)
class Progress:
    """Download progress; replaces the last log line."""
    text: str


@dataclass(frozen=True)
class LivePreview:
    """Intermediate transcription result while recording (display only)."""
    text: str
    header: str = ""


@dataclass(frozen=True)
class Finished:
    """Transcription completed successfully."""
    text: str
    txt_path: str
    audio_path: str


@dataclass(frozen=True)
class Failed:
    """Transcription failed.

    message is turned into a string INSIDE the worker - which is exactly why
    a NameError can no longer occur here.
    """
    message: str
    detail: str = ""


@dataclass(frozen=True)
class ModelFetch:
    """How a download started on the Settings tab is getting on."""
    model: str
    step: str             # what is happening, e.g. "Downloading model 'small'…"
    done: int = 0         # bytes of the current file so far
    total: int = 0        # its size; 0 while not known (or while unpacking)
    rate: float = 0.0     # bytes per second


@dataclass(frozen=True)
class ModelFetchEnded:
    """The download started on the Settings tab is over."""
    model: str
    error: str = ""       # empty: the model is ready - or it was cancelled
    cancelled: bool = False


@dataclass(frozen=True)
class UpdateChecked:
    """The check for a new version is done."""
    release: object = None    # an update.Release when a newer one is out
    error: str = ""           # why it could not be checked


@dataclass(frozen=True)
class UpdateFetch:
    """How the download of an update is getting on."""
    step: str             # e.g. "Downloading Sotto v2.1.0…"
    done: int = 0
    total: int = 0
    rate: float = 0.0


@dataclass(frozen=True)
class UpdateFetchEnded:
    """The download of an update is over."""
    tag: str
    staged: str = ""      # the unpacked app folder, ready to install
    error: str = ""
    cancelled: bool = False


class UiBridge:
    """Thread-safe one-way street from workers to the GUI."""

    def __init__(self, root, interval_ms=50):
        self._root = root
        self._queue = queue.Queue()
        self._handlers = {}
        self._interval = interval_ms
        self._running = False
        self._after_id = None

    def on(self, event_type, handler):
        self._handlers[event_type] = handler

    def post(self, event):
        """Callable from any thread."""
        self._queue.put(event)

    def post_exception(self, prefix, exc):
        """Convenience for except blocks - formats to a string right away."""
        self._queue.put(Failed(message=f"{prefix}: {exc}",
                               detail=traceback.format_exc()))

    def start(self):
        self._running = True
        self._pump()

    def stop(self):
        self._running = False
        if self._after_id is not None:
            try:
                self._root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def drain_now(self):
        """Process everything pending immediately (for tests and shutdown)."""
        self._dispatch_pending()

    def _dispatch_pending(self):
        # Cap per pass: a download must not starve the GUI.
        for _ in range(200):
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                return
            handler = self._handlers.get(type(event))
            if handler is None:
                continue
            try:
                handler(event)
            except Exception:
                # A broken handler must not kill the pump.
                traceback.print_exc()

    def _pump(self):
        self._dispatch_pending()
        if self._running:
            self._after_id = self._root.after(self._interval, self._pump)
