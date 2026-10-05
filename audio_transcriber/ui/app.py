"""The main window.

Layout: a header, then cards (Sources, Transcription, Recording, Transcript).
All colours and spacings come from theme.py, the controls from widgets.py.

The interface still contains no audio, network or process logic. It reads
widgets exclusively on the GUI thread, hands snapshots to the workers and
receives results only through UiBridge events.
"""

import os
import threading
import time
import tkinter as tk
import webbrowser
from dataclasses import asdict
from tkinter import filedialog, messagebox, ttk

try:
    # Windows: the WASAPI loopback fork. Everywhere else plain PyAudio - the
    # app then has no loopback capture, but recording and file upload work.
    # capture.py does the same dance when it opens a stream.
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio

from .. import config, paths, pipeline, preflight, update, version
from ..audio import capture
from ..audio import devices as devmod
from ..audio.capture import AudioEngine
from ..events import (Failed, Finished, LivePreview, Log, ModelFetch,
                      ModelFetchEnded, Progress, Status, UiBridge,
                      UpdateChecked, UpdateFetch, UpdateFetchEnded)
from ..transcribe import binaries
from ..transcribe.base import format_clock
from . import chrome, dialogs, icons
from . import theme as T
from . import widgets as W

METER_INTERVAL_MS = 40
# How often the ticker asks whether there are unsaved settings.
SAVE_CHECK_INTERVAL_S = 0.2

# The window, designed for a 96 dpi screen: what it opens at, and the least it
# may be shrunk to. Both are scaled with the display and kept within the screen
# (_size_window); they used to be fixed pixel counts - 900 high at the least,
# which does not fit a 1366x768 laptop at all.
WINDOW_SIZE = (940, 840)
WINDOW_MIN_SIZE = (700, 600)
# What the screen keeps for itself: window frame, title bar, taskbar.
SCREEN_MARGIN = (24, 96)

# How long Start waits for an in-flight device reconfiguration before giving up
# and letting the engine report whatever is actually wrong. configure() joins
# its reader threads with a 2 s timeout each, so this leaves room for both.
DEVICE_READY_TIMEOUT_S = 6.0

# What the key field says about its key. The second is for systems without the
# Windows DPAPI, where the key cannot be stored at all - the hint used to claim
# the encryption there as well.
KEY_HINT_STORED = ("Encrypted with the Windows DPAPI and bound to your user "
                   "account — never stored in clear text.")
KEY_HINT_SESSION = ("This system has no secure place for the key, so it is kept "
                    "only until you close the app. Set the ELEVENLABS_API_KEY "
                    "environment variable to keep it.")


class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sotto")

        T.apply(root)                    # first: it sets the display scaling
        chrome.set_icon(root)
        self._size_window()

        self.settings, warnings = config.load()
        self.pa = pyaudio.PyAudio()
        self.engine = AudioEngine(self.pa, paths.TMP_DIR)
        self.bridge = UiBridge(root)
        self.devices = []
        self.finalizer = None
        self.live = None
        self.live_preview = None
        self.recording_base_name = None
        self.recording_started_at = None
        self._run_settings = None        # the copy of the settings a run works on
        self._prefetch = None            # fetches the model while recording
        self._model_download = None      # the Settings tab's Download button
        self._model_fetch_error = None   # (model, message) of its last failure
        self._saved_settings = None      # what was saved last; None: save anyway
        self._save_checked_at = 0.0
        self._work_thread = None         # the finalizer thread of the current run
        self._processing_base = None     # raw-track name of a recovered recording
        self._unfinished = []            # what find_unfinished() saw last
        self._monitor_thread = None
        self._start_deadline = 0.0
        self._shutting_down = False
        self._meter_after_id = None
        self._icon_refs = {}
        # Updates. The tag this build was made for; None when run from source,
        # which never looks for one.
        self._installed = version.current()
        self._update_release = None      # a newer update.Release, once found
        self._update_checking = False
        self._update_download = None     # the UpdateDownload while it runs
        self._update_staged = None       # the unpacked new version, ready
        self._update_error = None        # why the last download failed
        self._update_dismissed = False   # "Later": no banner this session
        self._installer_started = False

        paths.ensure_dirs()
        self._build()
        self._wire_events()
        self.bridge.start()
        self._refresh_recovery_banner()

        for warning in warnings:
            self.transcript.append(f"⚠ {warning}\n")
        if self.settings.migrated_plaintext_key and config.key_can_be_stored():
            self.transcript.append(
                "→ The key will be stored encrypted the next time you save.\n\n")

        self.refresh_devices()
        # The window now shows what settings.json holds - unless reading it
        # had something to correct or a key still has to be encrypted.
        needs_saving = bool(warnings) or (self.settings.migrated_plaintext_key
                                          and config.key_can_be_stored())
        self._saved_settings = None if needs_saving else self._settings_in_window()
        self._refresh_save_button()
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<F5>", lambda _e: self._toggle_recording())
        # Ctrl+1, 2, 3: the tabs, in the order they are shown.
        for number, page in enumerate(self.notebook.pages(), start=1):
            self.root.bind(f"<Control-Key-{number}>",
                           lambda _e, page=page: self.notebook.select(page))

    # ==================================================================
    # Construction
    # ==================================================================
    def _size_window(self):
        """Open at a size that suits this screen and its display scaling."""
        room = (max(1, self.root.winfo_screenwidth() - T.px(SCREEN_MARGIN[0])),
                max(1, self.root.winfo_screenheight() - T.px(SCREEN_MARGIN[1])))
        width, height = (min(T.px(wanted), available)
                         for wanted, available in zip(WINDOW_SIZE, room))
        least = (min(T.px(WINDOW_MIN_SIZE[0]), width),
                 min(T.px(WINDOW_MIN_SIZE[1]), height))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(*least)

    def _build(self):
        outer = tk.Frame(self.root, bg=T.BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=T.XL, pady=(T.LG, T.LG))

        self._build_header(outer)
        self._build_update_banner(outer)

        # The order of work: record, read, and now and then adjust.
        self.notebook = W.Tabs(outer)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_recorder = tk.Frame(self.notebook, bg=T.BG)
        self.tab_transcript = tk.Frame(self.notebook, bg=T.BG)
        self.tab_settings = tk.Frame(self.notebook, bg=T.BG)

        self.notebook.add(self.tab_recorder, "Recorder", "microphone")
        self.notebook.add(self.tab_transcript, "Transcript", "transcript")
        self.notebook.add(self.tab_settings, "Settings", "settings")

        # Recorder and Settings scroll where the screen cannot give the window
        # their full height (W.Scroller).
        recorder_page = W.Scroller(self.tab_recorder)
        recorder_page.pack(fill=tk.BOTH, expand=True)
        recorder_container = recorder_page.body
        self._build_recovery_banner(recorder_container)
        self._build_sources(recorder_container)
        self._build_record_bar(recorder_container)

        self._build_transcript(self.tab_transcript)

        settings_page = W.Scroller(self.tab_settings)
        settings_page.pack(fill=tk.BOTH, expand=True)
        settings_container = settings_page.body
        self._build_ai(settings_container)
        self._build_output_folder(settings_container)
        self._build_options_card(settings_container)
        self._build_updates_card(settings_container)
        self._build_settings_footer(settings_container)
        self._refresh_subtitle()

    # ------------------------------------------------------------------
    def _build_header(self, parent):
        head = tk.Frame(parent, bg=T.BG)
        head.pack(fill=tk.X, pady=(0, T.LG))

        left = tk.Frame(head, bg=T.BG)
        left.pack(side=tk.LEFT)

        self._icon_refs["logo"] = icons.get_icon("app_logo", size=T.px(44))
        tk.Label(left, image=self._icon_refs["logo"], bg=T.BG).pack(
            side=tk.LEFT, padx=(0, T.MD))

        text_frame = tk.Frame(left, bg=T.BG)
        text_frame.pack(side=tk.LEFT)
        tk.Label(text_frame, text="Sotto", bg=T.BG, fg=T.TEXT,
                 font=T.fonts["display"], anchor="w").pack(anchor="w")
        # What the next run is going to use, as set on the Settings tab
        # (_refresh_subtitle). It used to list every engine there is, whichever
        # was set.
        self.subtitle = tk.Label(text_frame, bg=T.BG, fg=T.TEXT_MUTE,
                                 font=T.fonts["small"], anchor="w")
        self.subtitle.pack(anchor="w", pady=(T.px(2), 0))

        self.status = W.StatusPill(head, bg=T.BG, width=260, height=36)
        self.status.pack(side=tk.RIGHT, anchor="e")

    # ------------------------------------------------------------------
    def _build_recovery_banner(self, parent):
        """A line above the sources while an interrupted recording is waiting.

        Built but not packed: _refresh_recovery_banner() shows it only when
        there is something to recover.
        """
        self.recovery_banner = W.Card(parent)
        body = self.recovery_banner.body
        body.columnconfigure(0, weight=1)
        self.recovery_label = W.WrapLabel(body, bg=T.CARD, fg=T.WARN,
                                          font=T.fonts["small"])
        self.recovery_label.grid(row=0, column=0, sticky="ew")
        self.recover_btn = W.Button(body, text="Recover…", kind="accent",
                                    width=120, height=32,
                                    command=self.offer_recovery)
        self.recover_btn.grid(row=0, column=1, padx=(T.MD, 0))

    # ------------------------------------------------------------------
    def _build_update_banner(self, parent):
        """A line under the header while a new version is out.

        Above the tabs, so it is seen whichever tab is open. Built but not
        packed: _refresh_update_banner() shows it when there is something to
        say. One action button and a "Later"; what they do follows the state
        of the update.
        """
        self.update_banner = W.Card(parent)
        body = self.update_banner.body
        body.columnconfigure(1, weight=1)
        self.update_banner_badge = W.IconBadge(body, "refresh", T.ACCENT)
        self.update_banner_badge.grid(row=0, column=0, rowspan=2, sticky="nw",
                                      padx=(0, T.MD), pady=(T.px(2), 0))
        self.update_title = tk.Label(body, bg=T.CARD, fg=T.TEXT,
                                     font=T.fonts["body_bold"], anchor="w")
        self.update_title.grid(row=0, column=1, sticky="w")
        self.update_detail = W.WrapLabel(body, font=T.fonts["small"])
        self.update_detail.grid(row=1, column=1, sticky="ew", pady=(T.px(2), 0))

        buttons = tk.Frame(body, bg=T.CARD)
        buttons.grid(row=0, column=2, rowspan=2, sticky="e", padx=(T.MD, 0))
        self.update_notes_btn = W.Button(buttons, text="What's new", kind="quiet",
                                         width=100, height=34,
                                         command=self._open_release_page)
        self.update_action_btn = W.Button(buttons, text="Update now",
                                          icon_name="save", kind="accent",
                                          width=130, height=34)
        self.update_action_btn.pack(side=tk.LEFT)
        self.update_later_btn = W.Button(buttons, text="Later", kind="quiet",
                                         width=70, height=34,
                                         command=self._dismiss_update)

        # Only while the download runs.
        self.update_progress = tk.Frame(body, bg=T.CARD)
        self.update_progress.grid(row=2, column=1, columnspan=2, sticky="ew",
                                  pady=(T.SM, 0))
        self.update_progress.columnconfigure(0, weight=1)
        self.update_bar = W.ProgressBar(self.update_progress)
        self.update_bar.grid(row=0, column=0, sticky="ew")
        self.update_percent = tk.Label(self.update_progress, bg=T.CARD,
                                       fg=T.TEXT_DIM, font=T.fonts["mono_small"],
                                       width=6, anchor="e")
        self.update_percent.grid(row=0, column=1, sticky="e", padx=(T.SM, 0))
        self.update_progress.grid_remove()

    # ------------------------------------------------------------------
    def _build_sources(self, parent):
        card = W.Card(parent, title="Sources", icon_name="sources")
        card.pack(fill=tk.X, pady=(0, T.MD))

        self.sources_card = card
        body = card.body
        body.columnconfigure(0, weight=1)

        # Each source in the colour of the voice it carries - the colours of
        # the logo and of the speakers in the transcript.
        self.mic_combo, self.mic_meter, self.mic_gain, self.mic_gain_label = \
            self._source_row(body, 0, "microphone", T.YOU, "Microphone",
                             "your voice", self._on_mic_gain)
        tk.Frame(body, bg=T.BORDER, height=1).grid(
            row=1, column=0, sticky="ew", pady=T.MD)
        self.sys_combo, self.sys_meter, self.sys_gain, self.sys_gain_label = \
            self._source_row(body, 2, "speaker", T.OTHERS, "System audio",
                             "everyone else on the call", self._on_sys_gain)

    def _source_row(self, body, row, icon_name, colour, title, whose, gain_command):
        """One source: what it is, its device, its level and its gain."""
        frame = tk.Frame(body, bg=T.CARD)
        frame.grid(row=row, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        W.IconBadge(frame, icon_name, colour).grid(
            row=0, column=0, rowspan=2, sticky="nw", padx=(0, T.MD), pady=(T.px(2), 0))
        heading = tk.Frame(frame, bg=T.CARD)
        heading.grid(row=0, column=1, sticky="w")
        tk.Label(heading, text=title, bg=T.CARD, fg=T.TEXT,
                 font=T.fonts["body_bold"]).pack(side=tk.LEFT)
        tk.Label(heading, text=f"·  {whose}", bg=T.CARD, fg=T.TEXT_MUTE,
                 font=T.fonts["small"]).pack(side=tk.LEFT, padx=(T.SM, 0))

        combo = ttk.Combobox(frame, state="readonly", style="Dark.TCombobox",
                             font=T.fonts["body"])
        combo.grid(row=1, column=1, sticky="ew", pady=(T.XS, T.SM))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.restart_monitoring())

        meter = W.Meter(frame, width=560, height=14, colour=colour)
        meter.grid(row=2, column=1, sticky="ew")

        gain_row = tk.Frame(frame, bg=T.CARD)
        gain_row.grid(row=3, column=1, sticky="ew", pady=(T.XS, 0))
        gain_row.columnconfigure(1, weight=1)
        tk.Label(gain_row, text="Gain", bg=T.CARD, fg=T.TEXT_MUTE,
                 font=T.fonts["small"], anchor="w").grid(row=0, column=0,
                                                         padx=(0, T.SM))
        slider = W.Slider(gain_row, from_=-20.0, to=20.0, command=gain_command,
                          width=480, colour=colour)
        slider.grid(row=0, column=1, sticky="ew")
        value = tk.Label(gain_row, text="+0.0 dB", bg=T.CARD, fg=T.TEXT_DIM,
                         font=T.fonts["mono_small"], width=9, anchor="e")
        value.grid(row=0, column=2, sticky="e")
        return combo, meter, slider, value

    # ------------------------------------------------------------------
    def _build_ai(self, parent):
        card = W.Card(parent, title="Transcription", icon_name="sparkle")
        card.pack(fill=tk.X, pady=(0, T.MD))
        body = card.body
        body.columnconfigure(0, weight=3, uniform="ai")
        body.columnconfigure(1, weight=2, uniform="ai")

        model_field = W.Field(body, "Model", lambda p: _combo(
            p, [choice[0] for choice in config.MODEL_CHOICES]), icon_name="brain")
        model_field.grid(row=0, column=0, sticky="ew", padx=(0, T.MD))
        self.model_combo = model_field.widget
        self.model_combo.current(config.model_index(self.settings.model))
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: (
            self._refresh_key_state(), self._refresh_model_state(),
            self._refresh_subtitle()))

        lang_field = W.Field(body, "Language", lambda p: _combo(
            p, [choice[0] for choice in config.LANGUAGE_CHOICES]), icon_name="globe")
        lang_field.grid(row=0, column=1, sticky="ew")
        self.lang_combo = lang_field.widget
        # An unknown code keeps the default language, not the first entry -
        # which is automatic detection.
        self.lang_combo.current(_index_of(config.LANGUAGE_CHOICES,
                                          self.settings.language,
                                          fallback=config.Settings.language))
        self.lang_combo.bind("<<ComboboxSelected>>",
                             lambda _e: self._refresh_subtitle())

        # --- API key ---------------------------------------------------
        # Shown only while the cloud engine is chosen (_refresh_key_state):
        # with a local model it was a disabled field taking a third of the card.
        def show_button(parent):
            self.show_key_btn = W.Button(parent, text="Show", icon_name="eye",
                                         kind="ghost", width=84, height=34,
                                         command=self._toggle_key)
            return self.show_key_btn

        self.key_field = W.Field(
            body, "ElevenLabs API key",
            lambda p: ttk.Entry(p, show="•", style="Dark.TEntry",
                                font=T.fonts["body"]),
            hint=KEY_HINT_STORED if config.key_can_be_stored() else KEY_HINT_SESSION,
            icon_name="lock", aside=show_button)
        self.key_field.grid(row=2, column=0, columnspan=2, sticky="ew",
                            pady=(T.MD, 0))
        self.api_entry = self.key_field.widget
        self.api_entry.insert(0, self.settings.api_key)
        self._key_visible = False

        self._build_model_panel(body)

    def _build_model_panel(self, body):
        """Whether the local model is on this computer, and a button to fetch it.

        Shown only while a local model is chosen (_refresh_model_state). A model
        used to come only with the first recording that needed it - 3 GB of
        large-v3 at the start of a meeting - and nothing showed whether it was
        there.
        """
        panel = self.model_panel = tk.Frame(body, bg=T.CARD)
        panel.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(T.MD, 0))
        panel.columnconfigure(1, weight=1)
        tk.Frame(panel, bg=T.BORDER, height=1).grid(
            row=0, column=0, columnspan=3, sticky="ew", pady=(0, T.MD))

        self.model_badge = W.IconBadge(panel, "save", T.TEXT_MUTE)
        self.model_badge.grid(row=1, column=0, rowspan=2, sticky="nw",
                              padx=(0, T.MD), pady=(T.px(2), 0))
        self.model_state_label = tk.Label(panel, bg=T.CARD, fg=T.TEXT,
                                          font=T.fonts["body_bold"], anchor="w")
        self.model_state_label.grid(row=1, column=1, sticky="w")
        self.model_detail = W.WrapLabel(panel, font=T.fonts["small"])
        self.model_detail.grid(row=2, column=1, sticky="ew", pady=(T.px(2), 0))
        self.model_btn = W.Button(panel, text="Download", icon_name="save",
                                  kind="accent", width=130, height=34,
                                  command=self._on_model_button)
        self.model_btn.grid(row=1, column=2, rowspan=2, sticky="e",
                            padx=(T.MD, 0))

        # Only while a download runs.
        self.model_progress = tk.Frame(panel, bg=T.CARD)
        self.model_progress.grid(row=3, column=1, columnspan=2, sticky="ew",
                                 pady=(T.SM, 0))
        self.model_progress.columnconfigure(0, weight=1)
        self.model_bar = W.ProgressBar(self.model_progress)
        self.model_bar.grid(row=0, column=0, sticky="ew")
        self.model_percent = tk.Label(self.model_progress, bg=T.CARD,
                                      fg=T.TEXT_DIM, font=T.fonts["mono_small"],
                                      width=6, anchor="e")
        self.model_percent.grid(row=0, column=1, sticky="e", padx=(T.SM, 0))
        self.model_progress.grid_remove()

    # ------------------------------------------------------------------
    def _build_output_folder(self, parent):
        card = W.Card(parent, title="Output folder", icon_name="folder")
        card.pack(fill=tk.X, pady=(0, T.MD))
        body = card.body
        body.columnconfigure(0, weight=1)

        def buttons(parent):
            frame = tk.Frame(parent, bg=T.CARD)
            W.Button(frame, text="Browse…", icon_name="folder", kind="ghost",
                     width=100, height=34,
                     command=self._browse_output_dir).pack(side=tk.LEFT)
            W.Button(frame, text="Reset", kind="quiet", width=74, height=34,
                     command=self._reset_output_dir).pack(side=tk.LEFT,
                                                          padx=(T.XS, 0))
            return frame

        self.out_dir_field = W.Field(
            body, "Folder",
            lambda p: ttk.Entry(p, style="Dark.TEntry", font=T.fonts["body"]),
            hint="Recordings (.wav) and transcripts (.txt) are saved here.",
            aside=buttons)
        self.out_dir_field.grid(row=0, column=0, sticky="ew")
        self.output_dir_entry = self.out_dir_field.widget
        self.output_dir_entry.insert(0, self.settings.output_dir or paths.OUT_DIR)

    def _browse_output_dir(self):
        current = self.output_dir_entry.get().strip() or self.settings.get_output_dir()
        chosen = filedialog.askdirectory(
            title="Select Output Folder for Transcripts & Audio",
            initialdir=current if os.path.exists(current) else paths.OUT_DIR
        )
        if chosen:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, os.path.normpath(chosen))

    def _reset_output_dir(self):
        self.output_dir_entry.delete(0, tk.END)
        self.output_dir_entry.insert(0, paths.OUT_DIR)

    # ------------------------------------------------------------------
    def _build_options_card(self, parent):
        card = W.Card(parent, title="Processing", icon_name="settings")
        card.pack(fill=tk.X, pady=(0, T.MD))

        body = card.body
        body.columnconfigure((0, 1), weight=1, uniform="option")

        self.live_var = tk.BooleanVar(value=self.settings.live_transcribe)
        self.preview_var = tk.BooleanVar(value=self.settings.live_preview)
        self.separate_var = tk.BooleanVar(value=self.settings.separate_tracks)
        self.vad_var = tk.BooleanVar(value=self.settings.use_vad)
        self.gpu_var = tk.BooleanVar(value=self.settings.use_gpu)
        self.keep_raw_var = tk.BooleanVar(value=self.settings.keep_raw_tracks)

        # Every switch with a line on what it does, next to it - not one
        # paragraph below all six that explained two of them. Two columns; the
        # descriptions wrap to the width there is (W.WrapLabel).
        options = (
            ("Live transcription", self.live_var,
             "Transcribes while you record, so the transcript is ready right "
             "after Stop. Local models only."),
            ("Live preview", self.preview_var,
             "Shows the text in the Transcript tab as it is recognised."),
            ("Separate tracks", self.separate_var,
             "Transcribes your voice and the system audio one after the other, "
             "for exact speakers. Off: one quicker pass over the mix."),
            ("VAD", self.vad_var,
             "Voice activity detection: leaves out silence before transcribing. "
             "Off keeps the timestamps exact."),
            ("Keep raw tracks", self.keep_raw_var,
             "Keeps both recorded tracks next to the transcript, as .mic.wav "
             "and .sys.wav."),
            ("GPU", self.gpu_var,
             "Lets whisper.cpp use the graphics card where its build can; a run "
             "that fails there is repeated on the CPU."),
        )
        indent = T.px(W.Switch.TRACK_W + 10)
        for index, (caption, variable, text) in enumerate(options):
            column = index % 2
            tile = tk.Frame(body, bg=T.CARD)
            tile.grid(row=index // 2, column=column, sticky="new",
                      padx=(0, T.XL) if column == 0 else 0,
                      pady=(T.MD if index >= 2 else 0, 0))
            W.Switch(tile, caption, variable).pack(anchor="w")
            W.WrapLabel(tile, text=text).pack(fill=tk.X, padx=(indent, 0),
                                               pady=(T.px(2), 0))

    def _build_updates_card(self, parent):
        """Which version this is, whether there is a newer one, and the switch
        for asking at every start.

        A slim line rather than a card: with one more card the Settings tab no
        longer fitted a window of the usual size and had to scroll.
        """
        row = tk.Frame(parent, bg=T.BG)
        row.pack(fill=tk.X, pady=(0, T.MD))
        row.columnconfigure(1, weight=1)

        self.update_badge = W.IconBadge(row, "refresh", T.TEXT_MUTE, bg=T.BG)
        self.update_badge.grid(row=0, column=0, rowspan=2, sticky="w",
                               padx=(T.XS, T.MD))
        tk.Label(row, bg=T.BG, fg=T.TEXT, font=T.fonts["body_bold"], anchor="w",
                 text=f"Sotto {self._installed}" if self._installed
                 else "Sotto, from source").grid(row=0, column=1, sticky="sw")
        self.update_info = W.WrapLabel(row, bg=T.BG)
        self.update_info.grid(row=1, column=1, sticky="new")

        controls = tk.Frame(row, bg=T.BG)
        controls.grid(row=0, column=2, rowspan=2, sticky="e", padx=(T.MD, 0))
        self.update_auto_var = tk.BooleanVar(value=self.settings.check_for_updates)
        W.Switch(controls, "Check for updates at start", self.update_auto_var,
                 bg=T.BG).pack(side=tk.LEFT, padx=(0, T.LG))
        self.update_check_btn = W.Button(
            controls, text="Check now", icon_name="refresh", kind="ghost",
            bg=T.BG, width=120, height=34,
            state="normal" if self._installed else "disabled",
            command=lambda: self.check_for_updates(manual=True))
        self.update_check_btn.pack(side=tk.LEFT)

        if self._installed:
            self._show_update_info("Not checked yet. Nothing is downloaded "
                                   "until you say so.")
        else:
            self._show_update_info(update.install_problem())

    def _build_settings_footer(self, parent):
        footer = tk.Frame(parent, bg=T.BG)
        footer.pack(fill=tk.X, pady=(0, T.SM))
        footer.columnconfigure(0, weight=1)
        W.WrapLabel(footer, bg=T.BG,
                    text="Changes apply to the next recording right away; "
                         "Save keeps them for the next start.").grid(
            row=0, column=0, sticky="ew", padx=(T.XS, T.LG))
        self.save_settings_btn = W.Button(footer, text="Save settings",
                                          icon_name="check", kind="accent",
                                          bg=T.BG, width=150, height=38,
                                          command=self.save_settings)
        self.save_settings_btn.grid(row=0, column=1, sticky="e")


    # ------------------------------------------------------------------
    def _build_record_bar(self, parent):
        # A grey dot: a red one in the heading looked like a recording running.
        card = W.Card(parent, title="Recording", icon_name="record",
                      icon_colour=T.TEXT_DIM)
        card.pack(fill=tk.X, pady=(0, T.MD))
        body = card.body
        body.columnconfigure(0, weight=1)

        tk.Label(body, text="File name", bg=T.CARD, fg=T.TEXT_DIM,
                 font=T.fonts["small"], anchor="w").grid(
            row=0, column=0, sticky="w", pady=(0, T.px(4)))
        self.filename_entry = ttk.Entry(body, style="Dark.TEntry",
                                        font=T.fonts["body"])
        self.filename_entry.grid(row=1, column=0, sticky="ew", padx=(0, T.LG))
        self.filename_entry.insert(0, self.settings.filename)

        self.timer_label = tk.Label(body, text="00:00", bg=T.CARD,
                                    fg=T.TEXT_MUTE, font=T.fonts["timer"])
        self.timer_label.grid(row=1, column=1, padx=(0, T.LG))

        self.upload_btn = W.Button(body, text="Upload file", icon_name="upload",
                                   kind="ghost", width=130, height=42,
                                   command=self.upload_and_transcribe)
        self.upload_btn.grid(row=1, column=2, padx=(0, T.SM))

        # Start and Stop share a place: while a recording runs, Stop is the
        # only thing that can be done, and a Stop button that is disabled
        # the rest of the time only looked broken.
        self.start_btn = W.Button(body, text="Start recording", icon_name="record",
                                  kind="record", width=170, height=42,
                                  command=self.start_recording)
        self.start_btn.grid(row=1, column=3, sticky="e")

        self.stop_btn = W.Button(body, text="Stop recording", icon_name="stop",
                                 kind="stop", width=170, height=42, state="disabled",
                                 command=self.stop_recording)
        self.stop_btn.grid(row=1, column=3, sticky="e")
        self.stop_btn.grid_remove()

        W.WrapLabel(body, text="The recording and its transcript are saved "
                               "under this name in the output folder. F5 starts "
                               "and stops a recording.").grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(T.SM, 0))

    def _show_stop(self, show):
        """Stop in place of Start while a recording runs, and back."""
        hidden, shown = ((self.start_btn, self.stop_btn) if show
                         else (self.stop_btn, self.start_btn))
        hidden.grid_remove()
        shown.grid()


    # ------------------------------------------------------------------
    def _build_transcript(self, parent):
        card = W.Card(parent, title="Transcript", icon_name="transcript", stretch=True)
        card.pack(fill=tk.BOTH, expand=True, pady=(0, T.SM))

        toolbar = tk.Frame(card.body, bg=T.CARD)
        toolbar.pack(fill=tk.X, pady=(0, T.SM))

        # Which colour is who, in the words of the transcript.
        for colour, who in ((T.SPEAKER_SELF, "You"), (T.SPEAKER_OTHER, "Participants")):
            tk.Label(toolbar, text="●", bg=T.CARD, fg=colour,
                     font=T.fonts["small"]).pack(side=tk.LEFT)
            tk.Label(toolbar, text=who, bg=T.CARD, fg=T.TEXT_DIM,
                     font=T.fonts["small"]).pack(side=tk.LEFT, padx=(T.px(4), T.LG))

        self.clear_btn = W.Button(toolbar, text="Clear", icon_name="trash", kind="quiet",
                                  width=80, height=32, command=lambda: self.transcript.clear())
        self.clear_btn.pack(side=tk.RIGHT)

        self.save_transcript_btn = W.Button(toolbar, text="Save", icon_name="save",
                                            kind="ghost", width=80, height=32,
                                            command=self._save_transcript)
        self.save_transcript_btn.pack(side=tk.RIGHT, padx=(0, T.SM))

        self.copy_btn = W.Button(toolbar, text="Copy", icon_name="copy", kind="ghost",
                                 width=80, height=32, command=self._copy_transcript)
        self.copy_btn.pack(side=tk.RIGHT, padx=(0, T.SM))

        tk.Frame(card.body, bg=T.BORDER, height=1).pack(fill=tk.X, pady=(0, T.SM))

        self.transcript = W.Transcript(card.body)
        self.transcript.pack(fill=tk.BOTH, expand=True)


    # ==================================================================
    # Events
    # ==================================================================
    def _wire_events(self):
        self.bridge.on(Log, lambda e: self.transcript.append(e.text))
        self.bridge.on(Progress, lambda e: self.transcript.replace_last_line(e.text))
        self.bridge.on(Status, lambda e: self.status.set(e.text, _colour(e.color)))
        self.bridge.on(LivePreview, self._on_live_preview)
        self.bridge.on(Finished, self._on_finished)
        self.bridge.on(Failed, self._on_failed)
        self.bridge.on(ModelFetch, self._on_model_fetch)
        self.bridge.on(ModelFetchEnded, self._on_model_fetch_ended)
        self.bridge.on(UpdateChecked, self._on_update_checked)
        self.bridge.on(UpdateFetch, self._on_update_fetch)
        self.bridge.on(UpdateFetchEnded, self._on_update_fetch_ended)
        # A recording may have fetched the model in the meantime.
        self.notebook.bind("<<NotebookTabChanged>>",
                           lambda _e: self._refresh_model_state())

    def _on_live_preview(self, event):
        self.transcript.set_transcript(event.text, header=event.header,
                                       scroll_to_end=True)

    def _on_finished(self, event):
        self.transcript.set_transcript(event.text)
        self.status.set("done · transcript saved", T.OK)
        self._reset_controls()
        self.notebook.select(self.tab_transcript)
        out_folder = os.path.dirname(event.txt_path)
        messagebox.showinfo(
            "Done",
            f"Transcription complete.\n\n"
            f"Audio:      {os.path.basename(event.audio_path)}\n"
            f"Transcript: {os.path.basename(event.txt_path)}\n\n"
            f"Folder: {out_folder}")


    def _on_failed(self, event):
        self.transcript.append(f"\n[ERROR] {event.message}\n")
        self.status.set("processing failed", T.DANGER)
        self._reset_controls()
        if self._unfinished:
            # A failed run leaves its recorded tracks behind on purpose.
            self.transcript.append(
                "The recorded tracks were kept. Once the problem is fixed, "
                "'Recover…' on the Recorder tab processes them again.\n")
        messagebox.showerror("Error", event.message)

    # ==================================================================
    # Devices
    # ==================================================================
    def refresh_devices(self):
        self.devices = devmod.enumerate_devices(self.pa)
        mics = devmod.microphone_candidates(self.devices)
        outputs = (devmod.playback_candidates(self.devices)
                   + devmod.loopback_devices(self.devices))

        self.mic_combo["values"] = [device.label for device in mics]
        self.sys_combo["values"] = [device.label for device in outputs]

        if not mics:
            self.transcript.append("⚠ No capture device was found.\n")
        if not outputs:
            self.transcript.append("⚠ No playback device was found.\n")

        self._select(self.mic_combo, mics, self.settings.mic_device)
        self._select(self.sys_combo, outputs, self.settings.loop_device)

        self.mic_gain.set(self.settings.mic_gain_db)
        self._on_mic_gain(self.settings.mic_gain_db)
        self.sys_gain.set(self.settings.loop_gain_db)
        self._on_sys_gain(self.settings.loop_gain_db)

        self._refresh_key_state()
        self._refresh_model_state()
        self.restart_monitoring()

    @staticmethod
    def _select(combo, device_list, preferred):
        if not device_list:
            return
        labels = [device.label for device in device_list]
        if preferred in labels:
            combo.set(preferred)
            return
        if preferred and ":" in preferred:
            wanted = preferred.split(":", 1)[1].strip()
            for index, device in enumerate(device_list):
                if device.name == wanted:
                    combo.current(index)
                    return
        combo.current(0)

    def _current_devices(self):
        mic = devmod.by_label(self.devices, self.mic_combo.get())
        chosen = devmod.by_label(self.devices, self.sys_combo.get())
        loopback, reason = devmod.find_loopback_for(self.devices, chosen)
        return mic, loopback, reason

    def restart_monitoring(self):
        if self.engine.is_recording:
            return
        mic, loopback, reason = self._current_devices()
        self.sources_card.set_hint(
            f"System audio captured from: {reason}" if loopback else f"⚠ {reason}")

        def worker():
            for warning in self.engine.configure(mic, loopback):
                self.bridge.post(Log(f"⚠ {warning}\n"))

        thread = threading.Thread(target=worker, name="restart-monitor",
                                  daemon=True)
        self._monitor_thread = thread
        thread.start()

    # ==================================================================
    # Recording
    # ==================================================================
    def _toggle_recording(self):
        if self.engine.is_recording:
            self.stop_recording()
        elif str(self.start_btn["state"]) != "disabled":
            self.start_recording()

    def start_recording(self):
        mic, loopback, _reason = self._current_devices()
        if mic is None and loopback is None:
            messagebox.showwarning("No devices",
                                   "Please select a microphone and a playback device.")
            return

        # Disable straight away so F5 or a second click cannot start twice
        # while we are still waiting for the devices. Upload goes with it: it
        # runs its own pipeline and takes over the buttons and the transcript,
        # which must not happen while a recording is on its way, being made or
        # processed. _reset_controls() lets both go again.
        self.start_btn.config(state="disabled")
        self.upload_btn.config(state="disabled")
        self._start_deadline = time.monotonic() + DEVICE_READY_TIMEOUT_S
        self._start_when_devices_ready()

    def _start_when_devices_ready(self):
        """Begin recording once no device reconfiguration is in flight.

        engine.configure() runs off the GUI thread, and between closing the old
        streams and assigning the new ones the engine has no active track.
        Hitting Start in that window failed with "Neither audio source is
        active" immediately after a perfectly valid device change. We poll
        instead of joining so the interface stays responsive.

        On timeout we fall through deliberately: start_recording then reports
        the engine's real error rather than hiding it behind a spinner.
        """
        if self._shutting_down:
            return

        thread = self._monitor_thread
        busy = ((thread is not None and thread.is_alive())
                or self.engine.is_configuring)
        if busy and time.monotonic() < self._start_deadline:
            self.status.set("preparing devices…", T.WARN)
            self.root.after(80, self._start_when_devices_ready)
            return

        self._begin_recording()

    def _begin_recording(self):
        self._sync_settings_from_ui()
        # The run gets its own copy: the sliders and 'Save settings' go on
        # changing self.settings while it is under way, and the live chunks and
        # the closing pass must be recognised with one and the same model and
        # language.
        self._run_settings = self.settings.snapshot()

        go, heads_up = self._preflight(self._run_settings, recording=True)
        if not go:
            self._allow_new_work()
            return

        base_name = self._resolve_output_conflict(
            paths.safe_output_name(self.filename_entry.get()),
            paths.output_extensions(self.settings.keep_raw_tracks))
        if base_name is None:                       # the user cancelled
            self._allow_new_work()
            return
        self._apply_base_name(base_name)

        # Attach live transcription BEFORE the first block is written: its
        # sample positions are positions in the raw file, and a late start
        # would shift the whole live half of the transcript.
        notes = self._start_live(base_name)

        try:
            self.engine.start_recording(base_name)
        except RuntimeError as exc:
            self._stop_live()
            self._allow_new_work()
            messagebox.showerror("Cannot record", str(exc))
            return

        self.recording_base_name = base_name
        self.recording_started_at = time.monotonic()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._show_stop(True)
        self.mic_combo.config(state="disabled")
        self.sys_combo.config(state="disabled")
        self.timer_label.config(fg=T.REC)
        self.status.set("recording", T.REC, pulse=True)
        self.transcript.clear()
        self.transcript.append("Recording started.\n")
        for note in notes:
            self.transcript.append(note)
        self._show_heads_up(heads_up)

        # Only now that the recording really runs: fetch the model the closing
        # pass needs, if it is not here yet, instead of after the meeting.
        self._prefetch = pipeline.ModelPrefetch(self.bridge, self._run_settings)
        self._prefetch.start()

    def _preflight(self, run, recording):
        """Look for trouble before anything is started.

        Returns (go, notes). Errors are shown and mean no - the run would fail
        after the recording, or never get going. Warnings are asked about.
        Notes are only things worth knowing and come back to be written to the
        log once it has been cleared for the run.
        """
        findings = preflight.check(run, recording=recording)
        errors = [f.text for f in findings if f.level == preflight.ERROR]
        if errors:
            messagebox.showerror("Cannot start", "\n\n".join(errors))
            return False, []
        warnings = [f.text for f in findings if f.level == preflight.WARNING]
        if warnings and not messagebox.askokcancel(
                "Before you start", "\n\n".join(warnings) + "\n\nStart anyway?",
                icon="warning"):
            return False, []
        return True, [f.text for f in findings if f.level == preflight.NOTE]

    def _show_heads_up(self, notes):
        for note in notes:
            self.transcript.append(f"ℹ {note}\n")

    def _start_live(self, base_name):
        """Live transcription if the backend allows it, otherwise the preview.

        Never both: they would run two whisper processes against each other on
        the same cores, and the preview has nothing to add once the real
        transcription is already running along.

        Returns the lines to show once the transcript pane has been cleared -
        this runs before the recording starts, so it must not write there yet.
        """
        wants_live = bool(self.live_var.get())
        wants_preview = bool(self.preview_var.get())
        run = self._run_settings

        if wants_live and pipeline.live_transcription_possible(run):
            self.live = pipeline.LiveTranscriber(self.bridge, run, self.engine,
                                                 preview=wants_preview)
            self.live.start(base_name)
            return [f"Live transcription running with "
                    f"{run.model_name()} - stopping only has to "
                    f"catch up with the tail.\n"]

        notes = []
        if wants_live:
            notes.append("⚠ Live transcription needs a local model; "
                         "ElevenLabs transcribes after you stop.\n")
        if wants_preview:
            self.live_preview = pipeline.LivePreview(self.bridge, run, self.engine)
            self.live_preview.start()
        return notes

    def _stop_live(self):
        """Tear both live paths down without keeping their results."""
        if self.live is not None:
            self.live.cancel()
            self.live = None
        if self.live_preview is not None:
            self.live_preview.stop()
            self.live_preview = None

    def stop_recording(self):
        self.stop_btn.config(state="disabled")
        self._show_stop(False)           # Start stays disabled until it is done
        self.status.set("stopping…", T.WARN)
        self.recording_started_at = None
        self.timer_label.config(fg=T.TEXT_MUTE)

        if self.live_preview is not None:
            self.live_preview.stop()
            self.live_preview = None

        recording = self.engine.stop_recording()
        # Stop buffering right away; the finalizer waits for the chunk that is
        # still in flight, off the GUI thread.
        live, self.live = self.live, None
        if live is not None:
            live.close()

        if not recording.has_audio:
            if live is not None:
                live.cancel()
            recording.discard()          # empty files: nothing to recover later
            messagebox.showwarning("No data", "No audio data was captured.")
            self._reset_controls()
            return

        # The levels are meant to be moved while listening to the meters, and
        # the mixdown they shape is made now; everything else stays as it was
        # when the recording began.
        if self._run_settings is None:
            self._run_settings = self.settings.snapshot()
        self._run_settings.mic_gain_db = self.settings.mic_gain_db
        self._run_settings.loop_gain_db = self.settings.loop_gain_db

        self._start_finalizer(recording, self.recording_base_name, live=live)

    def _start_finalizer(self, recording, base_name, live=None):
        self.finalizer = pipeline.Finalizer(self.bridge, self._run_settings,
                                            live=live)
        self._work_thread = self.finalizer.run_async(recording, base_name)

    def upload_and_transcribe(self):
        # The button is disabled while a recording or any processing is under
        # way; the check keeps a direct call from slipping past it.
        if self.engine.is_recording or str(self.upload_btn["state"]) == "disabled":
            return

        # Before a file is picked: a missing key or an unwritable folder is no
        # reason to make the user choose one first.
        self._sync_settings_from_ui()
        go, heads_up = self._preflight(self.settings.snapshot(), recording=False)
        if not go:
            return

        file_types = [
            ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma *.mp4 *.webm *.opus *.aiff *.m4b *.amr *.caf"),
            ("WAV Audio", "*.wav"),
            ("MP3 Audio", "*.mp3"),
            ("M4A / AAC Audio", "*.m4a *.aac"),
            ("FLAC / OGG Audio", "*.flac *.ogg"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="Select Audio File to Transcribe",
            filetypes=file_types
        )
        if not file_path:
            return

        file_basename = os.path.basename(file_path)   # safe_output_name drops the extension
        base_name = self._resolve_output_conflict(
            paths.safe_output_name(self.filename_entry.get() or file_basename))
        if base_name is None:                       # the user cancelled
            return
        self._apply_base_name(base_name)

        self.start_btn.config(state="disabled")
        self.upload_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.mic_combo.config(state="disabled")
        self.sys_combo.config(state="disabled")
        self.status.set("transcribing file…", T.WARN)
        self.transcript.clear()
        self.transcript.append(f"Processing uploaded file: {os.path.basename(file_path)}\n")
        self._show_heads_up(heads_up)

        self.finalizer = pipeline.FileFinalizer(self.bridge,
                                                self.settings.snapshot())
        self._work_thread = self.finalizer.run_async(file_path, base_name)

    # ------------------------------------------------------------------
    def _resolve_output_conflict(self, base_name, extensions=None):
        """Ask before an existing recording is silently replaced.

        The output files are written as '<base_name>.wav' / '.txt' (and, for a
        recording with "Keep raw tracks", '.mic.wav' / '.sys.wav'), so a second
        run under the same name used to overwrite the first one without a word.
        Returns the base name to write to - possibly numbered or renamed - or
        None when the user cancelled.
        """
        out_dir = self.settings.get_output_dir()
        if not paths.existing_outputs(out_dir, base_name,
                                      extensions or paths.OUTPUT_EXTENSIONS):
            return base_name
        return dialogs.ask_output_conflict(self.root, out_dir, base_name,
                                           extensions)

    def _apply_base_name(self, base_name):
        """Show the name that is actually going to be written on disk."""
        self.filename_entry.delete(0, tk.END)
        self.filename_entry.insert(0, base_name)
        self.settings.filename = base_name

    def _allow_new_work(self):
        """Back to idle after a start that came to nothing (cancelled, refused)."""
        self.start_btn.config(state="normal")
        self.upload_btn.config(state="normal")
        self.status.set("ready", T.TEXT_MUTE)

    def _reset_controls(self):
        self.start_btn.config(state="normal")
        self.upload_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._show_stop(False)
        self.mic_combo.config(state="readonly")
        self.sys_combo.config(state="readonly")
        self.recording_started_at = None
        self.timer_label.config(text="00:00", fg=T.TEXT_MUTE)
        self.mic_meter.reset()
        self.sys_meter.reset()
        # The run is over once Finished / Failed arrives, even if its thread is
        # still tidying up for a moment - and only now can the check for
        # leftover recordings tell a failed run from one in progress.
        self._work_thread = None
        self._processing_base = None
        self._refresh_recovery_banner()

    # ==================================================================
    # Recordings that were interrupted
    # ==================================================================
    def _busy_with_work(self):
        thread = self._work_thread
        return self.engine.is_recording or (thread is not None and thread.is_alive())

    def _in_use(self):
        """Raw-track names of the recording being made or processed right now."""
        if not self._busy_with_work():
            return set()
        return {name for name in (self.recording_base_name, self._processing_base)
                if name}

    def _refresh_recovery_banner(self):
        """Show the banner while recordings are waiting that nothing works on."""
        found = capture.find_unfinished(paths.TMP_DIR, exclude=self._in_use())
        self._unfinished = found
        if not found or self._busy_with_work():
            self.recovery_banner.pack_forget()
            return
        first = found[0]
        more = f" and {len(found) - 1} more" if len(found) > 1 else ""
        self.recovery_label.config(
            text=f"⚠ Unfinished recording: {_shorten(first.base_name)} "
                 f"({format_clock(first.duration_s)}){more}")
        self.recovery_banner.pack(fill=tk.X, pady=(0, T.SM),
                                  before=self.sources_card)

    def offer_recovery(self):
        """Ask what to do with the newest interrupted recording.

        Called shortly after start-up and by the banner's button. Does nothing
        while a recording is being made or processed.
        """
        if self._shutting_down or self._busy_with_work():
            return
        self._refresh_recovery_banner()
        if not self._unfinished:
            return
        newest = self._unfinished[0]
        choice = dialogs.ask_recovery(self.root, newest,
                                      more=len(self._unfinished) - 1)
        if choice == "process":
            self._process_unfinished(newest)
        elif choice == "delete":
            self._delete_unfinished(newest)

    def _process_unfinished(self, unfinished):
        self._sync_settings_from_ui()
        run = self.settings.snapshot()
        go, heads_up = self._preflight(run, recording=False)
        if not go:                                  # the tracks stay where they are
            return
        base_name = self._resolve_output_conflict(
            paths.safe_output_name(unfinished.base_name),
            paths.output_extensions(self.settings.keep_raw_tracks))
        if base_name is None:                       # the user cancelled
            return

        self.start_btn.config(state="disabled")
        self.upload_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.mic_combo.config(state="disabled")
        self.sys_combo.config(state="disabled")
        self.recovery_banner.pack_forget()
        self.status.set("processing recovered recording…", T.WARN)
        self.transcript.clear()
        self.transcript.append(
            f"Processing the recovered recording '{unfinished.base_name}'…\n")
        self._show_heads_up(heads_up)

        self.recording_base_name = base_name
        self._processing_base = unfinished.base_name
        self._run_settings = run
        self._start_finalizer(capture.recording_from_unfinished(unfinished),
                              base_name)

    def _delete_unfinished(self, unfinished):
        if not messagebox.askyesno(
                "Delete recording",
                f"Delete the recorded tracks of '{unfinished.base_name}'?\n\n"
                f"This cannot be undone.", icon="warning"):
            return
        capture.remove_unfinished(unfinished)
        self._refresh_recovery_banner()


    # ==================================================================
    # Settings
    # ==================================================================
    def _sync_settings_from_ui(self):
        self._read_settings_from_ui(self.settings)

    def _read_settings_from_ui(self, settings):
        settings.mic_device = self.mic_combo.get()
        settings.loop_device = self.sys_combo.get()
        settings.mic_gain_db = round(float(self.mic_gain.get()), 1)
        settings.loop_gain_db = round(float(self.sys_gain.get()), 1)
        settings.model = config.model_key(self.model_combo.current())
        settings.language = config.LANGUAGE_CHOICES[max(0, self.lang_combo.current())][1]
        settings.api_key = self.api_entry.get().strip()
        settings.live_transcribe = bool(self.live_var.get())
        settings.live_preview = bool(self.preview_var.get())
        settings.separate_tracks = bool(self.separate_var.get())
        settings.use_vad = bool(self.vad_var.get())
        settings.use_gpu = bool(self.gpu_var.get())
        settings.keep_raw_tracks = bool(self.keep_raw_var.get())
        settings.check_for_updates = bool(self.update_auto_var.get())
        settings.filename = paths.safe_output_name(self.filename_entry.get())
        raw_out = self.output_dir_entry.get().strip()
        settings.output_dir = "" if raw_out == paths.OUT_DIR else raw_out


    def save_settings(self):
        self._sync_settings_from_ui()
        try:
            warnings = config.save(self.settings)
        except Exception as exc:
            messagebox.showerror("Error", f"Settings could not be saved:\n{exc}")
            return
        self.settings.migrated_plaintext_key = False
        # Saving again would not change what is stored - even after a warning:
        # a key that could not be secured would not be the second time either.
        self._saved_settings = self._settings_in_window()
        self._refresh_save_button()
        if warnings:
            # Pressed 'Save' and something did not get saved: say so where the
            # user is looking, not in a log on another tab.
            self.status.set("settings saved, with a warning", T.WARN)
            messagebox.showwarning("Settings saved", "\n\n".join(warnings))
            return
        self.status.set("settings saved", T.OK)

    def _settings_in_window(self):
        """What 'Save settings' would store now, read from the widgets."""
        pending = self.settings.snapshot()
        self._read_settings_from_ui(pending)
        state = asdict(pending)
        state.pop("migrated_plaintext_key", None)
        return state

    def _refresh_save_button(self):
        """'Save settings' in colour only while there is something to save.

        Asked by the ticker a few times a second: the settings live in a dozen
        widgets on two tabs, and asking them all beats wiring every one up.
        _saved_settings is None while saving is always worth it (a key from an
        old settings file still to be encrypted, a file that had to be
        corrected when it was read).
        """
        unsaved = (self._saved_settings is None
                   or self._settings_in_window() != self._saved_settings)
        state = "normal" if unsaved else "disabled"
        if str(self.save_settings_btn["state"]) != state:
            self.save_settings_btn.config(state=state)

    def _refresh_subtitle(self):
        self.subtitle.config(text=_engine_summary(
            config.model_key(self.model_combo.current()),
            config.LANGUAGE_CHOICES[max(0, self.lang_combo.current())][1]))

    def _refresh_key_state(self):
        """The API key only matters for the cloud backend."""
        uses_cloud = config.model_key(self.model_combo.current()) == config.CLOUD_MODEL
        self.api_entry.config(state="normal" if uses_cloud else "disabled")
        self.show_key_btn.config(state="normal" if uses_cloud else "disabled")
        if uses_cloud:
            self.key_field.grid()
        else:
            self.key_field.grid_remove()

    # ------------------------------------------------------------------
    # The local model on the Settings tab
    # ------------------------------------------------------------------
    def _refresh_model_state(self):
        """Show whether the chosen local model is here - and what to do if not."""
        self._refresh_model_labels()
        key = config.model_key(self.model_combo.current())
        if key == config.CLOUD_MODEL:
            self.model_panel.grid_remove()
            return
        self.model_panel.grid()
        if self._model_download is not None:
            return                           # its events keep the panel up to date

        named = f"'{key}'"
        model_here = binaries.model_present(key)
        engine_here = binaries.find_whisper_executable() is not None
        engine_problem = binaries.local_engine_problem()
        error = self._model_fetch_error
        if error is not None and error[0] != key:
            error = None

        if error is not None:
            self._show_model_state("warning", T.DANGER, "Download failed",
                                   error[1], button="Try again")
        elif model_here and engine_here:
            self._show_model_state(
                "check", T.OK, "Downloaded",
                f"The model {named} is on this computer, ready to transcribe "
                f"without an internet connection.")
        elif model_here and engine_problem:
            self._show_model_state("warning", T.WARN, "whisper.cpp is missing",
                                   engine_problem)
        elif model_here:
            self._show_model_state(
                "save", T.WARN, "Almost ready",
                f"The model {named} is here; whisper.cpp itself still has to be "
                f"fetched.", button="Download")
        else:
            self._show_model_state(
                "save", T.WARN, "Not downloaded",
                f"The model {named} is fetched the first time a recording needs "
                f"it - or now, so that nothing has to wait for it then.",
                button="Download")

    def _refresh_model_labels(self):
        """Mark the models that are on this computer in the model list."""
        index = self.model_combo.current()
        labels = [label + ("   ✓ downloaded"
                           if name and binaries.model_present(name) else "")
                  for label, name in config.MODEL_CHOICES]
        if list(self.model_combo["values"]) != labels:
            self.model_combo["values"] = labels
            self.model_combo.current(max(0, index))

    def _show_model_state(self, icon_name, colour, title, detail, button=None):
        self.model_badge.set(icon_name, colour)
        self.model_state_label.config(text=title)
        self.model_detail.config(text=detail)
        self.model_progress.grid_remove()
        if button is None:
            self.model_btn.grid_remove()
            return
        self.model_btn.config(text=button, kind="accent", icon_name="save",
                              state="normal")
        self.model_btn.grid()

    def _on_model_button(self):
        """Download - or, while a download runs, Cancel."""
        download = self._model_download
        if download is not None:
            download.cancel()
            self.model_state_label.config(text="Cancelling…")
            self.model_btn.config(state="disabled")
            return

        key = config.model_key(self.model_combo.current())
        if key == config.CLOUD_MODEL:
            return
        no_room = (None if binaries.model_present(key)
                   else preflight.no_room_for_models([key]))
        if no_room:
            self._model_fetch_error = (key, no_room)
            self._refresh_model_state()
            return

        self._model_fetch_error = None
        self._model_download = pipeline.ModelDownload(self.bridge, key)
        self.model_badge.set("save", T.ACCENT)
        self.model_state_label.config(text=f"Downloading '{key}'…")
        self.model_detail.config(text="Connecting…")
        self.model_btn.config(text="Cancel", kind="ghost", icon_name="stop",
                              state="normal")
        self.model_btn.grid()
        self.model_bar.set(None)
        self.model_percent.config(text="")
        self.model_progress.grid()
        self._model_download.start()

    def _on_model_fetch(self, event):
        download = self._model_download
        if (download is None or download.cancelled
                or event.model != download.model_name):
            return
        if event.step:
            self.model_state_label.config(text=event.step)
        if event.total:
            fraction = min(1.0, event.done / event.total)
            self.model_bar.set(fraction)
            self.model_percent.config(text=f"{fraction * 100:.0f} %")
        else:                                # connecting, unpacking
            self.model_bar.set(None)
            self.model_percent.config(text="")
        self.model_detail.config(text=_transfer_text(event) if event.done else "")

    def _on_model_fetch_ended(self, event):
        self._model_download = None
        if event.error:
            self._model_fetch_error = (event.model, event.error)
            self.status.set("model download failed", T.DANGER)
        elif event.cancelled:
            self.status.set("model download cancelled", T.TEXT_MUTE)
        else:
            self.status.set(f"model '{event.model}' downloaded", T.OK)
        self._refresh_model_state()

    def _toggle_key(self):
        self._key_visible = not self._key_visible
        self.api_entry.config(show="" if self._key_visible else "•")
        self.show_key_btn.configure(
            text="Hide" if self._key_visible else "Show",
            icon_name="eye_off" if self._key_visible else "eye"
        )

    def _copy_transcript(self):
        txt = self.transcript.text.get("1.0", tk.END).strip()
        if txt:
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            self.status.set("transcript copied", T.OK)

    def _save_transcript(self):
        txt = self.transcript.text.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showinfo("Empty transcript", "There is no text in the transcript to save.")
            return

        default_name = self.recording_base_name or paths.safe_output_name(self.filename_entry.get()) or "transcript"
        file_path = filedialog.asksaveasfilename(
            title="Save Transcript As",
            defaultextension=".txt",
            initialfile=f"{default_name}.txt",
            filetypes=[("Text File", "*.txt"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write(txt + "\n")
            self.status.set("transcript saved", T.OK)
            messagebox.showinfo("Saved", f"Transcript successfully saved to:\n{file_path}")
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save transcript:\n{exc}")


    def _on_mic_gain(self, value):
        self.settings.mic_gain_db = float(value)
        self.mic_gain_label.config(text=f"{float(value):+.1f} dB")
        if hasattr(self, "mic_meter") and hasattr(self, "engine"):
            gain_factor = 10.0 ** (self.settings.mic_gain_db / 20.0)
            self.mic_meter.set_level(self.engine.mic_level * gain_factor)

    def _on_sys_gain(self, value):
        self.settings.loop_gain_db = float(value)
        self.sys_gain_label.config(text=f"{float(value):+.1f} dB")
        if hasattr(self, "sys_meter") and hasattr(self, "engine"):
            gain_factor = 10.0 ** (self.settings.loop_gain_db / 20.0)
            self.sys_meter.set_level(self.engine.sys_level * gain_factor)

    # ==================================================================
    # Updates
    # ==================================================================
    def start_update_check(self):
        """Once the window is up (main.py): how the last update went, then ask
        GitHub. Not in the constructor - the tests build the window too, and
        must not go on the network."""
        if self._shutting_down or self._installed is None:
            return
        if update.install_problem() is None:
            self._report_update_result(update.take_result())
            threading.Thread(target=update.cleanup, name="update-cleanup",
                             daemon=True).start()
        if self.settings.check_for_updates:
            self.check_for_updates()

    def _report_update_result(self, result):
        """Say how the update installed at the last close or restart went."""
        if not result:
            return
        if result["ok"]:
            self.transcript.append(f"✓ Sotto was updated to {self._installed}.\n")
            self.status.set(f"updated to {self._installed}", T.OK)
        else:
            self.transcript.append(
                f"⚠ The update to {result['tag']} could not be installed: "
                f"{result['error']} Sotto is still {self._installed}.\n")
            self.status.set("update failed", T.WARN)

    def check_for_updates(self, manual=False):
        """Ask GitHub for a newer release, in the background."""
        if (self._installed is None or self._update_checking
                or self._update_download is not None
                or self._update_staged is not None):
            return
        self._update_checking = True
        if manual:
            self._update_dismissed = False
        self.update_check_btn.config(state="disabled")
        self._show_update_info("Checking…")
        update.UpdateCheck(self.bridge, self._installed).start()

    def _on_update_checked(self, event):
        self._update_checking = False
        self.update_check_btn.config(state="normal")
        if event.error:
            self._show_update_info(f"Could not check for updates: {event.error}",
                                   T.WARN, "warning")
            return
        if event.release is None:
            self._show_update_info("This is the newest version.", T.OK, "check")
            return
        self._update_release = event.release
        self._update_error = None
        self._show_update_info(f"Sotto {event.release.tag} is available.",
                               T.ACCENT)
        self._refresh_update_banner()

    def _show_update_info(self, text, colour=T.TEXT_MUTE, icon_name="refresh"):
        """The line on the Updates card in Settings."""
        self.update_info.config(text=text)
        self.update_badge.set(icon_name, colour)

    def _refresh_update_banner(self):
        """Show the banner in the state the update is in - or hide it."""
        release = self._update_release
        if release is None or self._update_dismissed:
            self.update_banner.pack_forget()
            return
        self.update_progress.grid_remove()
        problem = update.install_problem()

        if self._update_download is not None:
            self._banner("save", T.ACCENT, f"Downloading Sotto {release.tag}…",
                         "Connecting…", "Cancel", self._cancel_update_download,
                         kind="ghost", icon_name="stop", later=False)
            self.update_bar.set(None)
            self.update_percent.config(text="")
            self.update_progress.grid()
        elif self._update_staged is not None:
            self._banner("check", T.OK, f"Sotto {release.tag} is ready to install",
                         "Restart Sotto to finish - it takes a few seconds. Or "
                         "carry on: the update is installed when you close Sotto.",
                         "Restart now", self.restart_to_update, icon_name="refresh")
        elif self._update_error is not None:
            self._banner("warning", T.DANGER, "The update could not be downloaded",
                         self._update_error, "Try again", self._start_update_download,
                         icon_name="refresh")
        elif problem is not None:
            self._banner("refresh", T.ACCENT, f"Sotto {release.tag} is available",
                         f"You have {self._installed}. {problem}", "Download",
                         self._open_release_page)
        else:
            self._banner("refresh", T.ACCENT, f"Sotto {release.tag} is available",
                         f"You have {self._installed}. The update downloads in the "
                         f"background; your recordings, models and settings stay "
                         f"as they are.", "Update now", self._start_update_download,
                         notes=True)
        self.update_banner.pack(fill=tk.X, pady=(0, T.MD), before=self.notebook)

    def _banner(self, badge, colour, title, detail, action, command,
                kind="accent", icon_name="save", notes=False, later=True):
        """Fill the banner: its badge, its text, and which buttons it shows."""
        self.update_banner_badge.set(badge, colour)
        self.update_title.config(text=title)
        self.update_detail.config(text=detail)
        self.update_action_btn.config(text=action, kind=kind, command=command,
                                      icon_name=icon_name, state="normal")
        self.update_notes_btn.pack_forget()
        self.update_later_btn.pack_forget()
        if notes:
            self.update_notes_btn.pack(side=tk.LEFT, padx=(0, T.XS),
                                       before=self.update_action_btn)
        if later:
            self.update_later_btn.pack(side=tk.LEFT, padx=(T.XS, 0))

    def _start_update_download(self):
        release = self._update_release
        if release is None or self._update_download is not None:
            return
        self._update_error = None
        self._update_download = update.UpdateDownload(self.bridge, release,
                                                      update.workspace())
        self._refresh_update_banner()
        self._update_download.start()

    def _cancel_update_download(self):
        if self._update_download is not None:
            self._update_download.cancel()
            self.update_title.config(text="Cancelling…")
            self.update_action_btn.config(state="disabled")

    def _on_update_fetch(self, event):
        download = self._update_download
        if download is None or download.cancelled:
            return
        if event.step:
            self.update_title.config(text=event.step)
        if event.total:
            fraction = min(1.0, event.done / event.total)
            self.update_bar.set(fraction)
            self.update_percent.config(text=f"{fraction * 100:.0f} %")
        else:                                # checksums, verifying, unpacking
            self.update_bar.set(None)
            self.update_percent.config(text="")
        self.update_detail.config(text=_transfer_text(event) if event.done else "")

    def _on_update_fetch_ended(self, event):
        self._update_download = None
        if event.error:
            self._update_error = event.error
            self.status.set("update download failed", T.DANGER)
        elif event.staged:
            self._update_staged = event.staged
            self.status.set(f"{event.tag} ready to install", T.OK)
        self._refresh_update_banner()

    def restart_to_update(self):
        """Quit, let the installer swap the files, and start the new version."""
        if self._update_staged is None:
            return
        if self._busy_with_work():
            messagebox.showinfo(
                "Update",
                "A recording is being made or processed. Restart once it is "
                "done - or simply close Sotto then: the update is installed on "
                "the way out.")
            return
        try:
            update.start_install(self._update_staged, relaunch=True)
        except OSError as exc:
            messagebox.showerror("Update", f"The update could not be started:\n{exc}")
            return
        self._installer_started = True
        self.on_close()

    def _dismiss_update(self):
        """'Later': no banner for the rest of this session."""
        self._update_dismissed = True
        self._refresh_update_banner()

    def _open_release_page(self):
        release = self._update_release
        webbrowser.open(release.page_url if release else update.RELEASES_PAGE)

    # ==================================================================
    # Ticker
    # ==================================================================
    def _tick(self):
        if self._shutting_down:
            return
        # Called directly (the tests do), it would start a second ticker that
        # on_close no longer knows about - and that fires into a closed window.
        if self._meter_after_id is not None:
            self.root.after_cancel(self._meter_after_id)
            self._meter_after_id = None
        mic_gain = 10.0 ** (self.settings.mic_gain_db / 20.0)
        sys_gain = 10.0 ** (self.settings.loop_gain_db / 20.0)
        self.mic_meter.set_level(self.engine.mic_level * mic_gain)
        self.sys_meter.set_level(self.engine.sys_level * sys_gain)
        self.status.tick(METER_INTERVAL_MS / 1000.0)
        self.model_bar.tick(METER_INTERVAL_MS / 1000.0)
        self.update_bar.tick(METER_INTERVAL_MS / 1000.0)
        now = time.monotonic()
        if now - self._save_checked_at >= SAVE_CHECK_INTERVAL_S:
            self._save_checked_at = now
            self._refresh_save_button()
        for kind, message in self.engine.new_stream_errors():
            self._report_stream_error(kind, message)


        if self.recording_started_at is not None:
            elapsed = int(time.monotonic() - self.recording_started_at)
            hours, rest = divmod(elapsed, 3600)
            minutes, seconds = divmod(rest, 60)
            self.timer_label.config(
                text=f"{hours}:{minutes:02d}:{seconds:02d}" if hours
                else f"{minutes:02d}:{seconds:02d}")

        self._meter_after_id = self.root.after(METER_INTERVAL_MS, self._tick)

    def _report_stream_error(self, kind, message):
        """A source stopped delivering audio: say so now, not after Stop.

        The recording carries on with the other track and the dead one just
        ends, so until now nothing in the window told you - the meter going flat
        was the only sign.
        """
        source = "Microphone" if kind == "mic" else "System audio"
        self.transcript.append(f"⚠ {message}\n")
        self.sources_card.set_hint(f"⚠ {source} stopped")
        if self.engine.is_recording:
            self.status.set(f"recording - {source.lower()} stopped", T.WARN,
                            pulse=True)

    # ==================================================================
    # Shutdown
    # ==================================================================
    def _confirm_quit(self):
        """Closing the window mid-recording used to discard it without a word."""
        if self.engine.is_recording:
            what = "A recording is running. Quitting now stops it without a transcript."
        else:
            what = "A recording is still being processed. Quitting now cancels that."
        return messagebox.askyesno(
            "Quit now?",
            f"{what}\n\nThe recorded tracks are kept, and the next start "
            f"offers to process them. Quit anyway?", icon="warning")

    def on_close(self):
        if self._busy_with_work() and not self._confirm_quit():
            return
        self._shutting_down = True
        if self._meter_after_id is not None:
            try:
                self.root.after_cancel(self._meter_after_id)
            except Exception:
                pass
        self.bridge.stop()

        if self.live_preview is not None:
            self.live_preview.stop()
        if self.live is not None:
            self.live.cancel()
        if self._prefetch is not None:
            self._prefetch.cancel()
        if self._model_download is not None:
            self._model_download.cancel()
        if self._update_download is not None:
            self._update_download.cancel()
        if self.finalizer is not None:
            self.finalizer.cancel()

        try:
            self.engine.stop_streams()
        except Exception:
            pass
        try:
            self.pa.terminate()
        except Exception:
            pass

        # A downloaded update that was not installed with "Restart now" is
        # installed on the way out; the installer waits for this process to end.
        if self._update_staged is not None and not self._installer_started:
            try:
                update.start_install(self._update_staged, relaunch=False)
            except OSError:
                pass                     # the next start clears it away

        self.root.destroy()


# ----------------------------------------------------------------------
_STATUS_COLOURS = {
    "red": T.REC, "orange": T.WARN, "purple": T.ACCENT,
    "green": T.OK, "gray": T.TEXT_MUTE, "grey": T.TEXT_MUTE,
}


def _colour(name):
    return _STATUS_COLOURS.get(name, T.TEXT_MUTE)


def _engine_summary(model, language):
    """'whisper.cpp small, on this computer · German': what a run will use."""
    if model == config.CLOUD_MODEL:
        engine = "ElevenLabs Scribe, in the cloud"
    else:
        engine = f"whisper.cpp {model}, on this computer"
    names = {code: label for label, code in config.LANGUAGE_CHOICES}
    spoken = ("language detected automatically" if language == "auto"
              else names.get(language, language))
    return f"{engine} · {spoken}"


def _transfer_text(event):
    """'191 MB of 465 MB · 13.5 MB/s · about 20 s left' for a ModelFetch."""
    if not event.total:
        return f"{_megabytes(event.done)} so far"
    parts = [f"{_megabytes(event.done)} of {_megabytes(event.total)}"]
    if event.rate > 0:
        parts.append(f"{event.rate / (1 << 20):.1f} MB/s")
        if event.done < event.total:
            left = (event.total - event.done) / event.rate
            parts.append(f"about {_eta(left)} left")
    return " · ".join(parts)


def _megabytes(num_bytes):
    """In the units of the model list (and of the download log): 2^20 bytes."""
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GB"
    return f"{num_bytes / (1 << 20):.0f} MB"


def _eta(seconds):
    if seconds < 60:
        return f"{max(1, round(seconds))} s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def _combo(parent, values):
    return ttk.Combobox(parent, state="readonly", style="Dark.TCombobox",
                        font=T.fonts["body"], values=values)


def _index_of(choices, value, fallback=None):
    """Where `value` sits in `choices`; else where `fallback` does, else 0."""
    codes = [code for _label, code in choices]
    for candidate in (value, fallback):
        if candidate in codes:
            return codes.index(candidate)
    return 0


def _shorten(name, limit=40):
    return name if len(name) <= limit else name[:limit - 1] + "…"
