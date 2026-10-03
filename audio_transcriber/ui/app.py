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
from tkinter import filedialog, messagebox, ttk

try:
    # Windows: the WASAPI loopback fork. Everywhere else plain PyAudio - the
    # app then has no loopback capture, but recording and file upload work.
    # capture.py does the same dance when it opens a stream.
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio

from .. import config, paths, pipeline
from ..audio import devices as devmod
from ..audio.capture import AudioEngine
from ..events import (Failed, Finished, LivePreview, Log, Progress, Status,
                      UiBridge)
from . import dialogs, icons
from . import theme as T
from . import widgets as W

METER_INTERVAL_MS = 40

# How long Start waits for an in-flight device reconfiguration before giving up
# and letting the engine report whatever is actually wrong. configure() joins
# its reader threads with a 2 s timeout each, so this leaves room for both.
DEVICE_READY_TIMEOUT_S = 6.0


class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio AI Recorder")
        self.root.geometry("920x980")
        self.root.minsize(840, 900)

        T.apply(root)

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
        self._monitor_thread = None
        self._start_deadline = 0.0
        self._shutting_down = False
        self._meter_after_id = None
        self._icon_refs = {}

        paths.ensure_dirs()
        self._build()
        self._wire_events()
        self.bridge.start()

        for warning in warnings:
            self.transcript.append(f"⚠ {warning}\n")
        if self.settings.migrated_plaintext_key:
            self.transcript.append(
                "→ The key will be stored encrypted the next time you save.\n\n")

        self.refresh_devices()
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<F5>", lambda _e: self._toggle_recording())

    # ==================================================================
    # Construction
    # ==================================================================
    def _build(self):
        outer = tk.Frame(self.root, bg=T.BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=T.XL, pady=(T.LG, T.LG))

        self._build_header(outer)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_recorder = tk.Frame(self.notebook, bg=T.BG)
        self.tab_settings = tk.Frame(self.notebook, bg=T.BG)
        self.tab_transcript = tk.Frame(self.notebook, bg=T.BG)

        self.notebook.add(self.tab_recorder, text="  🎙️ Recorder  ")
        self.notebook.add(self.tab_settings, text="  ⚙️ Settings  ")
        self.notebook.add(self.tab_transcript, text="  📄 Transcript  ")

        # Tab 1: Recorder
        recorder_container = tk.Frame(self.tab_recorder, bg=T.BG)
        recorder_container.pack(fill=tk.BOTH, expand=True, pady=(T.SM, 0))
        self._build_sources(recorder_container)
        self._build_record_bar(recorder_container)

        # Tab 2: Settings
        settings_container = tk.Frame(self.tab_settings, bg=T.BG)
        settings_container.pack(fill=tk.BOTH, expand=True, pady=(T.SM, 0))
        self._build_ai(settings_container)
        self._build_output_folder(settings_container)
        self._build_options_card(settings_container)

        # Tab 3: Transcript
        transcript_container = tk.Frame(self.tab_transcript, bg=T.BG)
        transcript_container.pack(fill=tk.BOTH, expand=True, pady=(T.SM, 0))
        self._build_transcript(transcript_container)

    # ------------------------------------------------------------------
    def _build_header(self, parent):
        head = tk.Frame(parent, bg=T.BG)
        head.pack(fill=tk.X, pady=(0, T.MD))

        left = tk.Frame(head, bg=T.BG)
        left.pack(side=tk.LEFT)

        self._icon_refs["logo"] = icons.get_icon("app_logo", size=42)
        tk.Label(left, image=self._icon_refs["logo"], bg=T.BG).pack(
            side=tk.LEFT, padx=(0, T.MD))

        text_frame = tk.Frame(left, bg=T.BG)
        text_frame.pack(side=tk.LEFT)
        tk.Label(text_frame, text="Audio AI Recorder", bg=T.BG, fg=T.TEXT,
                 font=T.fonts["display"], anchor="w").pack(anchor="w")
        self.subtitle = tk.Label(
            text_frame, bg=T.BG, fg=T.TEXT_MUTE, font=T.fonts["small"], anchor="w",
            text=f"whisper.cpp · {self.settings.threads()} threads · "
                 f"ElevenLabs Scribe")
        self.subtitle.pack(anchor="w", pady=(2, 0))

        self.status = W.StatusPill(head, bg=T.BG, width=260, height=36)
        self.status.pack(side=tk.RIGHT, anchor="e")

    # ------------------------------------------------------------------
    def _build_sources(self, parent):
        card = W.Card(parent, title="Sources", icon_name="sources")
        card.pack(fill=tk.X, pady=(0, T.SM))

        self.sources_card = card
        body = card.body

        self.mic_combo, self.mic_meter, self.mic_gain, self.mic_gain_label = \
            self._source_row(body, "microphone", "Microphone — your voice",
                             self._on_mic_gain, row=0)
        tk.Frame(body, bg=T.BORDER, height=1).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=T.SM)
        self.sys_combo, self.sys_meter, self.sys_gain, self.sys_gain_label = \
            self._source_row(body, "speaker", "Playback — the other voices",
                             self._on_sys_gain, row=2)

        body.columnconfigure(1, weight=1)

    def _source_row(self, body, icon_name, label, gain_command, row):
        head = tk.Frame(body, bg=T.CARD)
        head.grid(row=row, column=0, columnspan=3, sticky="ew")
        head.columnconfigure(1, weight=1)

        self._icon_refs[f"src_{row}"] = icons.get_icon(icon_name, size=26)
        tk.Label(head, image=self._icon_refs[f"src_{row}"], bg=T.CARD).grid(
            row=0, column=0, sticky="w", padx=(0, T.SM))
        tk.Label(head, text=label, bg=T.CARD, fg=T.TEXT_DIM,
                 font=T.fonts["small"], anchor="w").grid(row=0, column=1,
                                                         sticky="w")

        combo = ttk.Combobox(head, state="readonly", style="Dark.TCombobox",
                             font=T.fonts["body"])
        combo.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 6))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.restart_monitoring())

        meter = W.Meter(head, width=560, height=18)
        meter.grid(row=2, column=0, columnspan=3, sticky="ew")


        gain_row = tk.Frame(head, bg=T.CARD)
        gain_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        gain_row.columnconfigure(1, weight=1)
        tk.Label(gain_row, text="Gain", bg=T.CARD, fg=T.TEXT_MUTE,
                 font=T.fonts["tiny"], width=5, anchor="w").grid(row=0, column=0)
        slider = W.Slider(gain_row, from_=-20.0, to=20.0, command=gain_command,
                          width=480)
        slider.grid(row=0, column=1, sticky="ew")
        value = tk.Label(gain_row, text="+0.0 dB", bg=T.CARD, fg=T.TEXT_DIM,
                         font=T.fonts["mono_small"], width=9, anchor="e")
        value.grid(row=0, column=2, sticky="e")
        return combo, meter, slider, value

    # ------------------------------------------------------------------
    def _build_ai(self, parent):
        card = W.Card(parent, title="Transcription Engine", icon_name="sparkle")
        card.pack(fill=tk.X, pady=(0, T.SM))
        body = card.body
        body.columnconfigure(0, weight=3, uniform="ai")
        body.columnconfigure(1, weight=2, uniform="ai")

        model_field = W.Field(body, "Model", lambda p: _combo(
            p, [choice[0] for choice in config.MODEL_CHOICES]), icon_name="brain")
        model_field.grid(row=0, column=0, sticky="ew", padx=(0, T.MD))
        self.model_combo = model_field.widget
        self.model_combo.current(self.settings.model_index)
        self.model_combo.bind("<<ComboboxSelected>>",
                              lambda _e: self._refresh_key_state())

        lang_field = W.Field(body, "Language", lambda p: _combo(
            p, [choice[0] for choice in config.LANGUAGE_CHOICES]), icon_name="globe")
        lang_field.grid(row=0, column=1, sticky="ew")
        self.lang_combo = lang_field.widget
        self.lang_combo.current(_index_of(config.LANGUAGE_CHOICES,
                                          self.settings.language))

        # --- API key ---------------------------------------------------
        key_wrap = tk.Frame(body, bg=T.CARD)
        key_wrap.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(T.MD, 0))
        key_wrap.columnconfigure(0, weight=1)

        self.key_field = W.Field(
            key_wrap, "ElevenLabs API key",
            lambda p: ttk.Entry(p, show="•", style="Dark.TEntry",
                                font=T.fonts["body"]),
            hint="Encrypted with the Windows DPAPI and bound to your user "
                 "account — never stored in clear text.", icon_name="lock")
        self.key_field.grid(row=0, column=0, sticky="ew")
        self.api_entry = self.key_field.widget
        self.api_entry.insert(0, self.settings.api_key)

        self.show_key_btn = W.Button(key_wrap, text="show", icon_name="eye",
                                     kind="ghost", width=84, height=32,
                                     command=self._toggle_key)
        self.show_key_btn.grid(row=0, column=1, sticky="n",
                               padx=(T.SM, 0), pady=(22, 0))
        self._key_visible = False

    # ------------------------------------------------------------------
    def _build_output_folder(self, parent):
        card = W.Card(parent, title="Output Directory", icon_name="folder")
        card.pack(fill=tk.X, pady=(0, T.SM))
        body = card.body
        body.columnconfigure(0, weight=1)

        wrap = tk.Frame(body, bg=T.CARD)
        wrap.grid(row=0, column=0, sticky="ew")
        wrap.columnconfigure(0, weight=1)

        self.out_dir_field = W.Field(
            wrap, "Target Folder",
            lambda p: ttk.Entry(p, style="Dark.TEntry", font=T.fonts["body"]),
            hint="Custom directory where transcripts (.txt) and audio (.wav) will be saved.",
            icon_name="globe"
        )

        self.out_dir_field.grid(row=0, column=0, sticky="ew")
        self.output_dir_entry = self.out_dir_field.widget
        self.output_dir_entry.insert(0, self.settings.output_dir or paths.OUT_DIR)

        browse_btn = W.Button(wrap, text="Browse...", icon_name="upload",
                              kind="quiet", width=100, height=32,
                              command=self._browse_output_dir)
        browse_btn.grid(row=0, column=1, sticky="s", padx=(T.SM, 0), pady=(22, 0))

        reset_btn = W.Button(wrap, text="Reset", kind="ghost",
                             width=74, height=32,
                             command=self._reset_output_dir)
        reset_btn.grid(row=0, column=2, sticky="s", padx=(T.XS, 0), pady=(22, 0))

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
        card = W.Card(parent, title="Processing Options", icon_name="settings")
        card.pack(fill=tk.X, pady=(0, T.SM))

        body = card.body
        body.columnconfigure(0, weight=1)

        options = tk.Frame(body, bg=T.CARD)
        options.grid(row=0, column=0, sticky="ew")

        self.live_var = tk.BooleanVar(value=self.settings.live_transcribe)
        self.preview_var = tk.BooleanVar(value=self.settings.live_preview)
        self.separate_var = tk.BooleanVar(value=self.settings.separate_tracks)
        self.vad_var = tk.BooleanVar(value=self.settings.use_vad)
        self.keep_raw_var = tk.BooleanVar(value=self.settings.keep_raw_tracks)

        W.Switch(options, "Live transcription", self.live_var).grid(
            row=0, column=0, sticky="w")
        W.Switch(options, "Live preview", self.preview_var).grid(
            row=0, column=1, sticky="w", padx=(T.MD, 0))
        W.Switch(options, "Separate tracks", self.separate_var).grid(
            row=0, column=2, sticky="w", padx=(T.MD, 0))
        W.Switch(options, "VAD", self.vad_var).grid(
            row=0, column=3, sticky="w", padx=(T.MD, 0))
        W.Switch(options, "Keep raw tracks", self.keep_raw_var).grid(
            row=1, column=0, sticky="w", pady=(T.SM, 0))

        tk.Label(options,
                 text="Live transcription recognises the recording while it "
                      "runs, so stopping only has to catch up with the last "
                      "few seconds. Local models only.",
                 bg=T.CARD, fg=T.TEXT_MUTE, font=T.fonts["tiny"],
                 anchor="w", justify="left").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(T.SM, 0))

        self.save_settings_btn = W.Button(options, text="Save settings",
                                          kind="ghost", width=150, height=32,
                                          command=self.save_settings)
        self.save_settings_btn.grid(row=0, column=4, sticky="e", padx=(T.MD, 0))
        options.columnconfigure(4, weight=1)


    # ------------------------------------------------------------------
    def _build_record_bar(self, parent):
        card = W.Card(parent)
        card.pack(fill=tk.X, pady=(0, T.SM))
        body = card.body
        body.columnconfigure(0, weight=1)

        name_field = W.Field(body, "File name", lambda p: ttk.Entry(
            p, style="Dark.TEntry", font=T.fonts["body"]))
        name_field.grid(row=0, column=0, sticky="ew", padx=(0, T.LG))
        self.filename_entry = name_field.widget
        self.filename_entry.insert(0, self.settings.filename)

        self.timer_label = tk.Label(body, text="00:00", bg=T.CARD,
                                    fg=T.TEXT_MUTE, font=T.fonts["display"])
        self.timer_label.grid(row=0, column=1, sticky="s", padx=(0, T.LG),
                              pady=(0, 2))

        self.upload_btn = W.Button(body, text="Upload file", icon_name="upload",
                                   kind="quiet", width=130, height=42,
                                   command=self.upload_and_transcribe)
        self.upload_btn.grid(row=0, column=2, sticky="s", padx=(0, T.SM), pady=(0, 1))

        self.start_btn = W.Button(body, text="Start recording", icon_name="record",
                                  kind="record", width=170, height=42,
                                  command=self.start_recording)
        self.start_btn.grid(row=0, column=3, sticky="s", pady=(0, 1))

        self.stop_btn = W.Button(body, text="Stop", icon_name="stop", kind="stop",
                                 width=100, height=42, state="disabled",
                                 command=self.stop_recording)
        self.stop_btn.grid(row=0, column=4, sticky="s", padx=(T.SM, 0),
                           pady=(0, 1))


    # ------------------------------------------------------------------
    def _build_transcript(self, parent):
        card = W.Card(parent, title="Transcript", icon_name="transcript", stretch=True)
        card.pack(fill=tk.BOTH, expand=True)

        toolbar = tk.Frame(card.body, bg=T.CARD)
        toolbar.pack(fill=tk.X, pady=(0, T.XS))

        self.clear_btn = W.Button(toolbar, text="Clear", icon_name="trash", kind="quiet",
                                  width=80, height=28, command=lambda: self.transcript.clear())
        self.clear_btn.pack(side=tk.RIGHT)

        self.save_transcript_btn = W.Button(toolbar, text="Save", icon_name="save",
                                            kind="quiet", width=80, height=28,
                                            command=self._save_transcript)
        self.save_transcript_btn.pack(side=tk.RIGHT, padx=(0, T.XS))

        self.copy_btn = W.Button(toolbar, text="Copy", icon_name="copy", kind="quiet",
                                 width=80, height=28, command=self._copy_transcript)
        self.copy_btn.pack(side=tk.RIGHT, padx=(0, T.XS))

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
        base_name = self._resolve_output_conflict(
            paths.safe_output_name(self.filename_entry.get()))
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
        self.mic_combo.config(state="disabled")
        self.sys_combo.config(state="disabled")
        self.timer_label.config(fg=T.REC)
        self.status.set("recording", T.REC, pulse=True)
        self.transcript.clear()
        self.transcript.append("Recording started.\n")
        for note in notes:
            self.transcript.append(note)

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

        if wants_live and pipeline.live_transcription_possible(self.settings):
            self.live = pipeline.LiveTranscriber(self.bridge, self.settings,
                                                 self.engine,
                                                 preview=wants_preview)
            self.live.start(base_name)
            return [f"Live transcription running with "
                    f"{self.settings.model_name()} - stopping only has to "
                    f"catch up with the tail.\n"]

        notes = []
        if wants_live:
            notes.append("⚠ Live transcription needs a local model; "
                         "ElevenLabs transcribes after you stop.\n")
        if wants_preview:
            self.live_preview = pipeline.LivePreview(self.bridge, self.settings,
                                                     self.engine)
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
            messagebox.showwarning("No data", "No audio data was captured.")
            self._reset_controls()
            return

        self.finalizer = pipeline.Finalizer(self.bridge, self.settings, live=live)
        self.finalizer.run_async(recording, self.recording_base_name)

    def upload_and_transcribe(self):
        # The button is disabled while a recording or any processing is under
        # way; the check keeps a direct call from slipping past it.
        if self.engine.is_recording or str(self.upload_btn["state"]) == "disabled":
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

        self._sync_settings_from_ui()
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

        self.finalizer = pipeline.FileFinalizer(self.bridge, self.settings)
        self.finalizer.run_async(file_path, base_name)

    # ------------------------------------------------------------------
    def _resolve_output_conflict(self, base_name):
        """Ask before an existing recording is silently replaced.

        Both output files are written as '<base_name>.wav' / '.txt', so a
        second run under the same name used to overwrite the first one without
        a word. Returns the base name to write to - possibly numbered or
        renamed - or None when the user cancelled.
        """
        out_dir = self.settings.get_output_dir()
        if not paths.existing_outputs(out_dir, base_name):
            return base_name
        return dialogs.ask_output_conflict(self.root, out_dir, base_name)

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
        self.mic_combo.config(state="readonly")
        self.sys_combo.config(state="readonly")
        self.recording_started_at = None
        self.timer_label.config(text="00:00", fg=T.TEXT_MUTE)
        self.mic_meter.reset()
        self.sys_meter.reset()


    # ==================================================================
    # Settings
    # ==================================================================
    def _sync_settings_from_ui(self):
        settings = self.settings
        settings.mic_device = self.mic_combo.get()
        settings.loop_device = self.sys_combo.get()
        settings.mic_gain_db = round(float(self.mic_gain.get()), 1)
        settings.loop_gain_db = round(float(self.sys_gain.get()), 1)
        settings.model_index = max(0, self.model_combo.current())
        settings.language = config.LANGUAGE_CHOICES[max(0, self.lang_combo.current())][1]
        settings.api_key = self.api_entry.get().strip()
        settings.live_transcribe = bool(self.live_var.get())
        settings.live_preview = bool(self.preview_var.get())
        settings.separate_tracks = bool(self.separate_var.get())
        settings.use_vad = bool(self.vad_var.get())
        settings.keep_raw_tracks = bool(self.keep_raw_var.get())
        settings.filename = paths.safe_output_name(self.filename_entry.get())
        raw_out = self.output_dir_entry.get().strip()
        settings.output_dir = "" if raw_out == paths.OUT_DIR else raw_out


    def save_settings(self):
        self._sync_settings_from_ui()
        try:
            config.save(self.settings)
        except Exception as exc:
            messagebox.showerror("Error", f"Settings could not be saved:\n{exc}")
            return
        self.settings.migrated_plaintext_key = False
        self.status.set("settings saved", T.OK)

    def _refresh_key_state(self):
        """The API key only matters for the cloud backend."""
        uses_cloud = config.MODEL_CHOICES[max(0, self.model_combo.current())][1] is None
        self.api_entry.config(state="normal" if uses_cloud else "disabled")
        self.show_key_btn.config(state="normal" if uses_cloud else "disabled")

    def _toggle_key(self):
        self._key_visible = not self._key_visible
        self.api_entry.config(show="" if self._key_visible else "•")
        self.show_key_btn.configure(
            text="hide" if self._key_visible else "show",
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
    # Ticker
    # ==================================================================
    def _tick(self):
        if self._shutting_down:
            return
        mic_gain = 10.0 ** (self.settings.mic_gain_db / 20.0)
        sys_gain = 10.0 ** (self.settings.loop_gain_db / 20.0)
        self.mic_meter.set_level(self.engine.mic_level * mic_gain)
        self.sys_meter.set_level(self.engine.sys_level * sys_gain)
        self.status.tick(METER_INTERVAL_MS / 1000.0)


        if self.recording_started_at is not None:
            elapsed = int(time.monotonic() - self.recording_started_at)
            hours, rest = divmod(elapsed, 3600)
            minutes, seconds = divmod(rest, 60)
            self.timer_label.config(
                text=f"{hours}:{minutes:02d}:{seconds:02d}" if hours
                else f"{minutes:02d}:{seconds:02d}")

        self._meter_after_id = self.root.after(METER_INTERVAL_MS, self._tick)

    # ==================================================================
    # Shutdown
    # ==================================================================
    def on_close(self):
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

        self.root.destroy()


# ----------------------------------------------------------------------
_STATUS_COLOURS = {
    "red": T.REC, "orange": T.WARN, "purple": T.ACCENT,
    "green": T.OK, "gray": T.TEXT_MUTE, "grey": T.TEXT_MUTE,
}


def _colour(name):
    return _STATUS_COLOURS.get(name, T.TEXT_MUTE)


def _combo(parent, values):
    return ttk.Combobox(parent, state="readonly", style="Dark.TCombobox",
                        font=T.fonts["body"], values=values)


def _index_of(choices, value):
    for index, (_label, code) in enumerate(choices):
        if code == value:
            return index
    return 0
