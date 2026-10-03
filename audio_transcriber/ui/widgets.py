"""Hand-drawn widgets for the dark interface.

Everything is built on tk.Canvas, without an extra library: rounded cards,
switches, sliders, level meters and buttons with hover and pressed states.
"""

import math
import re
import time
import tkinter as tk
from tkinter import ttk

from . import icons
from . import theme as T


# ======================================================================
class Card(tk.Canvas):
    """A rounded surface with an optional heading and icon accent.

    The trick: the rounded polygon sits on the canvas while the actual content
    lives in an embedded frame of the same background colour. Because the
    frame is inset on all sides it never reaches the corners, so the rounding
    stays visible.
    """

    def __init__(self, parent, title=None, hint=None, pad=(16, 13),
                 bg=T.BG, fill=T.CARD, stretch=False, icon_name=None):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, height=64)
        self.padx, self.pady = pad
        self._fill = fill
        self._title = title
        self._hint = hint
        self._icon_name = icon_name
        self._head_h = 26 if title else 0
        self._shape = None
        self._icon_img = None
        self._syncing = False
        # stretch=True: the card fills the available space instead of deriving
        # its height from the content (used for the transcript pane).
        self._stretch = stretch

        self.body = tk.Frame(self, bg=fill)
        self._win = self.create_window(self.padx, self.pady + self._head_h,
                                       anchor="nw", window=self.body)
        if title:
            x_off = self.padx
            if icon_name:
                self._icon_img = icons.get_icon(icon_name, size=16)
                self.create_image(self.padx, self.pady - 1, anchor="nw", image=self._icon_img)
                x_off += 22
            self._title_item = self.create_text(
                x_off, self.pady - 1, anchor="nw", text=title.upper(),
                fill=T.TEXT_DIM, font=T.fonts["section"])
        if hint:
            self._hint_item = self.create_text(
                0, self.pady - 1, anchor="ne", text=hint,
                fill=T.TEXT_MUTE, font=T.fonts["tiny"])

        self.bind("<Configure>", self._sync)
        self.body.bind("<Configure>", self._sync)

    def set_hint(self, text):
        if hasattr(self, "_hint_item"):
            self.itemconfigure(self._hint_item, text=text)

    def _sync(self, _event=None):
        if self._syncing:
            return
        self._syncing = True
        try:
            width = max(self.winfo_width(), 1)
            inner_w = max(1, width - 2 * self.padx)

            if self._stretch:
                height = max(self.winfo_height(), 120)
                self.itemconfigure(
                    self._win, width=inner_w,
                    height=max(1, height - 2 * self.pady - self._head_h))
            else:
                # Derive the height exactly from the content. Do not mix in
                # winfo_height(): that value can come from an earlier layout
                # pass, in which case the card is drawn larger than it is and
                # the bottom edge disappears.
                height = self.body.winfo_reqheight() + 2 * self.pady + self._head_h
                if abs(self.winfo_reqheight() - height) > 1:
                    self.configure(height=height)
                self.itemconfigure(self._win, width=inner_w)

            if hasattr(self, "_hint_item"):
                self.coords(self._hint_item, width - self.padx, self.pady - 1)
            self._draw(width, height)
        finally:
            self._syncing = False

    def _draw(self, width, height):
        if self._shape is not None:
            self.delete(self._shape)
        self._shape = T.round_rect(self, 0.5, 0.5, width - 0.5, height - 0.5,
                                   T.RADIUS_CARD, fill=self._fill,
                                   outline=T.BORDER, width=1)
        self.tag_lower(self._shape)


# ======================================================================
class Button(tk.Canvas):
    """A button with a rounded surface plus hover and pressed states.

    kind: 'accent' | 'record' | 'stop' | 'ghost' | 'quiet'
    """

    _PALETTE = {
        "accent": (T.ACCENT, T.ACCENT_HI, T.ACCENT_LO, "#ffffff", None),
        "record": (T.REC, T.REC_HI, "#d93a48", "#ffffff", None),
        "stop": (T.CARD_HI, "#2e3440", "#20242c", T.TEXT, T.BORDER_HI),
        "ghost": (None, T.CARD_HI, T.BG_DEEP, T.TEXT_DIM, T.BORDER),
        "quiet": (None, T.CARD_HI, T.BG_DEEP, T.TEXT_MUTE, None),
    }

    def __init__(self, parent, text="", command=None, kind="ghost",
                 width=150, height=38, icon="", icon_name=None, bg=T.CARD,
                 radius=None, state="normal"):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=width, height=height)
        self._kind = kind
        self._command = command
        self._text = text
        self._icon = icon
        self._icon_name = icon_name or (icon if icon in icons._RENDERERS else None)
        self._radius = radius if radius is not None else T.RADIUS_CTRL
        self._state = str(state)
        self._hover = False
        self._pressed = False
        self._shape = None
        self._label = None
        self._img_obj = None

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", lambda _e: self._render())
        self._render()

    # -- API compatible with ttk widgets --------------------------------
    def configure(self, cnf=None, **kw):
        state = kw.pop("state", None)
        text = kw.pop("text", None)
        kind = kw.pop("kind", None)
        icon_name = kw.pop("icon_name", None)
        if state is not None:
            self._state = str(state)
            self._hover = self._pressed = False
        if text is not None:
            self._text = text
        if kind is not None:
            self._kind = kind
        if icon_name is not None:
            self._icon_name = icon_name
        result = super().configure(cnf, **kw) if (cnf or kw) else None
        if state is not None or text is not None or kind is not None or icon_name is not None:
            self._render()
        return result

    config = configure

    def __getitem__(self, key):
        if key == "state":
            return self._state
        if key == "text":
            return self._text
        return super().__getitem__(key)

    def invoke(self):
        if self._state != "disabled" and self._command:
            self._command()

    # -- Events ----------------------------------------------------------
    def _on_enter(self, _event):
        if self._state != "disabled":
            self._hover = True
            self.configure(cursor="hand2")
            self._render()

    def _on_leave(self, _event):
        self._hover = self._pressed = False
        self._render()

    def _on_press(self, _event):
        if self._state != "disabled":
            self._pressed = True
            self._render()

    def _on_release(self, event):
        was_pressed = self._pressed
        self._pressed = False
        self._render()
        if was_pressed and self._state != "disabled":
            if 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height():
                self.invoke()

    # -- Rendering -------------------------------------------------------
    def _render(self):
        base, hover, press, fg, border = self._PALETTE[self._kind]
        if self._state == "disabled":
            fill = T.BG_DEEP if base else None
            fg = T.TEXT_MUTE
            border = T.BORDER
        elif self._pressed:
            fill, border = press, border
        elif self._hover:
            fill, border = hover, (T.BORDER_HI if border else None)
        else:
            fill, border = base, border

        width = max(self.winfo_width(), int(self["width"]))
        height = max(self.winfo_height(), int(self["height"]))

        self.delete("all")
        if fill or border:
            self._shape = T.round_rect(
                self, 1, 1, width - 1, height - 1, self._radius,
                fill=fill or self["bg"], outline=border or "", width=1)

        # Check for SVG icon
        if self._icon_name and self._icon_name in icons._RENDERERS:
            icon_sz = min(20, height - 14)
            self._img_obj = icons.get_icon(self._icon_name, size=icon_sz)
            
            font = T.fonts["button"]
            text_w = font.measure(self._text) if self._text else 0
            gap = 8 if self._text else 0
            total_w = icon_sz + gap + text_w
            
            start_x = (width - total_w) / 2
            self.create_image(start_x + icon_sz / 2, height / 2, image=self._img_obj)
            if self._text:
                self.create_text(start_x + icon_sz + gap + text_w / 2, height / 2 + 1,
                                 text=self._text, fill=fg, font=font)
        else:
            label = f"{self._icon}  {self._text}".strip() if self._icon else self._text
            self.create_text(width / 2, height / 2 + 1, text=label, fill=fg,
                             font=T.fonts["button"])


# ======================================================================
class Switch(tk.Canvas):
    """A sliding toggle with a caption."""

    TRACK_W, TRACK_H, KNOB_R = 38, 20, 7

    def __init__(self, parent, text="", variable=None, command=None,
                 bg=T.CARD, width=None):
        self.var = variable if variable is not None else tk.BooleanVar(value=False)
        self._command = command
        self._text = text
        self._hover = False

        font = T.fonts["body"]
        text_w = font.measure(text) if text else 0
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=width or (self.TRACK_W + 10 + text_w + 4),
                         height=max(self.TRACK_H, 22))

        self.bind("<Button-1>", self._toggle)
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        try:
            self.var.trace_add("write", lambda *_: self._render())
        except AttributeError:                      # pragma: no cover
            self.var.trace("w", lambda *_: self._render())
        self._render()

    def _set_hover(self, value):
        self._hover = value
        self.configure(cursor="hand2" if value else "")
        self._render()

    def _toggle(self, _event=None):
        self.var.set(not self.var.get())
        if self._command:
            self._command()

    def _render(self):
        self.delete("all")
        on = bool(self.var.get())
        mid = self.winfo_reqheight() / 2

        track = T.ACCENT if on else T.BG_DEEP
        if self._hover:
            track = T.ACCENT_HI if on else T.BORDER
        T.round_rect(self, 1, mid - self.TRACK_H / 2, self.TRACK_W,
                     mid + self.TRACK_H / 2, self.TRACK_H / 2,
                     fill=track, outline=T.BORDER if not on else "", width=1)

        cx = (self.TRACK_W - self.KNOB_R - 3) if on else (self.KNOB_R + 4)
        self.create_oval(cx - self.KNOB_R, mid - self.KNOB_R,
                         cx + self.KNOB_R, mid + self.KNOB_R,
                         fill="#ffffff" if on else T.TEXT_DIM, outline="")

        if self._text:
            self.create_text(self.TRACK_W + 10, mid, anchor="w", text=self._text,
                             fill=T.TEXT if on else T.TEXT_DIM,
                             font=T.fonts["body"])


# ======================================================================
class Slider(tk.Canvas):
    """A slider with track, fill and knob. Supports dragging and clicking."""

    HEIGHT = 24
    KNOB_R = 7

    def __init__(self, parent, from_=-20.0, to=20.0, value=0.0, command=None,
                 width=280, bg=T.CARD, centered=True):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=width, height=self.HEIGHT)
        self.from_, self.to = from_, to
        self._value = value
        self._command = command
        self._centered = centered      # fill from the centre instead of the left
        self._hover = False
        self._dragging = False
        self._enabled = True

        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<Configure>", lambda _e: self._render())
        self.bind("<MouseWheel>", self._on_wheel)
        self._render()

    # -- API --------------------------------------------------------------
    def get(self):
        return self._value

    def set(self, value, notify=False):
        self._value = max(self.from_, min(self.to, float(value)))
        self._render()
        if notify and self._command:
            self._command(self._value)

    def configure(self, cnf=None, **kw):
        state = kw.pop("state", None)
        if state is not None:
            self._enabled = str(state) != "disabled"
        return super().configure(cnf, **kw) if (cnf or kw) else None

    config = configure

    # -- Events -----------------------------------------------------------
    def _set_hover(self, value):
        self._hover = value
        self.configure(cursor="hand2" if value else "")
        self._render()

    def _x_to_value(self, x):
        usable = max(1, self.winfo_width() - 2 * self.KNOB_R - 2)
        share = (x - self.KNOB_R - 1) / usable
        return self.from_ + max(0.0, min(1.0, share)) * (self.to - self.from_)

    def _on_click(self, event):
        if not self._enabled:
            return
        self._dragging = True
        self.set(self._x_to_value(event.x), notify=True)

    def _on_drag(self, event):
        if self._dragging and self._enabled:
            self.set(self._x_to_value(event.x), notify=True)

    def _on_release(self, _event):
        self._dragging = False

    def _on_wheel(self, event):
        if not self._enabled:
            return
        step = (self.to - self.from_) / 80.0
        self.set(self._value + (step if event.delta > 0 else -step), notify=True)

    # -- Rendering --------------------------------------------------------
    def _render(self):
        self.delete("all")
        width = max(self.winfo_width(), 60)
        mid = self.HEIGHT / 2
        x0, x1 = self.KNOB_R + 1, width - self.KNOB_R - 1
        span = x1 - x0

        T.round_rect(self, x0, mid - 2, x1, mid + 2, 2,
                     fill=T.BG_DEEP, outline="")

        share = (self._value - self.from_) / float(self.to - self.from_)
        knob_x = x0 + share * span

        if self._centered:
            centre = x0 + span * (0.0 - self.from_) / float(self.to - self.from_)
            left, right = min(centre, knob_x), max(centre, knob_x)
            self.create_line(centre, mid - 5, centre, mid + 5,
                             fill=T.BORDER_HI, width=1)
        else:
            left, right = x0, knob_x

        if abs(right - left) > 1:
            T.round_rect(self, left, mid - 2, right, mid + 2, 2,
                         fill=T.ACCENT, outline="")

        radius = self.KNOB_R + (1 if self._hover or self._dragging else 0)
        self.create_oval(knob_x - radius, mid - radius,
                         knob_x + radius, mid + radius,
                         fill="#ffffff" if self._hover or self._dragging else T.TEXT,
                         outline=T.ACCENT if self._dragging else "", width=2)


# ======================================================================
class Meter(tk.Canvas):
    """A level meter with blocks, peak marker and dB readout.

    The ballistics are time based (not tied to the call frequency) and
    0 dBFS corresponds to full scale.
    """

    MIN_DB, MAX_DB = -60.0, 0.0
    DECAY_DB_S = 30.0
    PEAK_HOLD_S = 1.2
    PEAK_FALL_DB_S = 14.0
    READOUT_W = 66

    def __init__(self, parent, width=320, height=14, blocks=44, bg=T.CARD):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=width, height=height)
        self._height = height
        self._block_count = blocks
        self._bar_w = max(20, width - self.READOUT_W)
        self._built_for = 0
        self._blocks = []
        self._peak_item = None
        self._readout = None

        self._db = self.MIN_DB
        self._shown = self.MIN_DB
        self._peak = self.MIN_DB
        self._peak_at = 0.0
        self._last = time.monotonic()

        self._build(width)
        # The meter is stretched by grid/pack, so the blocks have to follow the
        # real width - otherwise the bar ends mid-card and the dB value does
        # not stick to the right edge.
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        if abs(event.width - self._built_for) > 2:
            self._build(event.width)
            self._render()

    def _build(self, width):
        self.delete("all")
        self._built_for = width
        self._bar_w = max(20, width - self.READOUT_W)
        self._blocks = []

        gap = 2
        step = self._bar_w / self._block_count
        for index in range(self._block_count):
            x0 = index * step
            share = (index + 1) / self._block_count
            if share <= 0.68:
                on, off = T.METER_LOW, T.METER_OFF_LOW
            elif share <= 0.88:
                on, off = T.METER_MID, T.METER_OFF_MID
            else:
                on, off = T.METER_HIGH, T.METER_OFF_HIGH
            item = self.create_rectangle(x0, 1, x0 + step - gap, self._height - 1,
                                         fill=off, outline="")
            self._blocks.append((item, x0, on, off))

        self._peak_item = self.create_line(0, 0, 0, self._height, fill="#ffffff",
                                           width=2, state="hidden")
        self._readout = self.create_text(width - 1, self._height / 2, anchor="e",
                                         text="  — dB", fill=T.TEXT_MUTE,
                                         font=T.fonts["mono_small"])

    @property
    def db(self):
        return self._db

    def set_level(self, rms):
        self._db = _to_db(rms, self.MIN_DB)
        self._render()

    def reset(self):
        self._db = self._shown = self._peak = self.MIN_DB
        self._render()

    def _render(self):
        now = time.monotonic()
        dt = max(0.0, min(0.5, now - self._last))
        self._last = now

        self._shown = max(self.MIN_DB, max(self._db, self._shown - self.DECAY_DB_S * dt))
        if self._db >= self._peak:
            self._peak, self._peak_at = self._db, now
        elif now - self._peak_at > self.PEAK_HOLD_S:
            self._peak = max(self.MIN_DB, self._peak - self.PEAK_FALL_DB_S * dt)

        filled = self._x(self._shown)
        for item, x0, on, off in self._blocks:
            self.itemconfigure(item, fill=on if filled >= x0 + 1 else off)

        if self._peak > self.MIN_DB + 1.5:
            x = max(2, self._x(self._peak))
            self.coords(self._peak_item, x, 1, x, self._height - 1)
            self.itemconfigure(self._peak_item, state="normal",
                               fill=T.METER_HIGH if self._peak > -6 else "#ffffff")
        else:
            self.itemconfigure(self._peak_item, state="hidden")

        if self._db > self.MIN_DB + 1:
            colour = T.METER_HIGH if self._db > -3 else (
                T.METER_MID if self._db > -12 else T.TEXT_DIM)
            self.itemconfigure(self._readout, text=f"{self._db:5.1f} dB", fill=colour)
        else:
            self.itemconfigure(self._readout, text="  — dB", fill=T.TEXT_MUTE)

    def _x(self, db):
        share = (db - self.MIN_DB) / (self.MAX_DB - self.MIN_DB)
        return self._bar_w * max(0.0, min(1.0, share))


def _to_db(rms, floor):
    if rms is None or rms <= 1e-6:
        return floor
    return max(floor, min(6.0, 20.0 * math.log10(rms)))


# ======================================================================
class StatusPill(tk.Canvas):
    """Status display: a dot plus text, optionally pulsing."""

    def __init__(self, parent, bg=T.BG, width=230, height=28):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=width, height=height)
        self._text = "ready"
        self._colour = T.TEXT_MUTE
        self._pulse = False
        self._phase = 0.0
        self._render()

    def set(self, text, colour=T.TEXT_MUTE, pulse=False):
        self._text, self._colour, self._pulse = text, colour, pulse
        if not pulse:
            self._phase = 0.0
        self._render()

    def tick(self, dt=0.04):
        if self._pulse:
            self._phase = (self._phase + dt * 2.2) % (2 * math.pi)
            self._render()

    def _render(self):
        self.delete("all")
        width = int(self["width"])
        height = int(self["height"])
        mid = height / 2

        alpha = 1.0
        if self._pulse:
            alpha = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(self._phase))
        dot = T.mix(self["bg"], self._colour, alpha)

        text_w = T.fonts["small"].measure(self._text)
        pill_w = text_w + 32
        x0 = max(0, width - pill_w)

        # Glassmorphic rounded pill background
        T.round_rect(self, x0, 1, width - 1, height - 1, (height - 2) / 2,
                     fill=T.CARD, outline=T.BORDER, width=1)

        radius = 4
        cx = x0 + 14
        self.create_oval(cx - radius, mid - radius, cx + radius, mid + radius,
                         fill=dot, outline="")
        if self._pulse:
            halo = T.mix(self["bg"], self._colour, alpha * 0.25)
            self.create_oval(cx - radius - 4, mid - radius - 4,
                             cx + radius + 4, mid + radius + 4,
                             outline=halo, width=1)
        self.create_text(x0 + 26, mid, anchor="w", text=self._text,
                         fill=T.TEXT_DIM, font=T.fonts["small"])


# ======================================================================
class Transcript(tk.Frame):
    """The transcript pane with speaker colours and a slim scrollbar."""

    def __init__(self, parent, height=14, bg=T.CARD):
        super().__init__(parent, bg=bg)
        self.text = tk.Text(
            self, height=height, wrap="word", relief="flat", bd=0,
            bg=bg, fg=T.TEXT, insertbackground=T.ACCENT,
            selectbackground=T.mix(bg, T.ACCENT, 0.35), selectforeground=T.TEXT,
            font=T.fonts["mono"], padx=2, pady=2, highlightthickness=0,
            spacing1=2, spacing3=2, state=tk.DISABLED, cursor="arrow")
        self.scroll = ttk.Scrollbar(self, orient="vertical",
                                    style="Dark.Vertical.TScrollbar",
                                    command=self.text.yview)
        self.text.configure(yscrollcommand=self.scroll.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))

        self.text.tag_configure("ts", foreground=T.TIMESTAMP)
        self.text.tag_configure("self", foreground=T.SPEAKER_SELF,
                                font=T.fonts["mono"])
        self.text.tag_configure("other", foreground=T.SPEAKER_OTHER,
                                font=T.fonts["mono"])
        self.text.tag_configure("body", foreground=T.TEXT)
        self.text.tag_configure("system", foreground=T.TEXT_MUTE,
                                font=T.fonts["mono_small"])
        self.text.tag_configure("warn", foreground=T.WARN,
                                font=T.fonts["mono_small"])
        self.text.tag_configure("error", foreground=T.DANGER,
                                font=T.fonts["mono_small"])
        self.text.tag_configure("head", foreground=T.ACCENT,
                                font=T.fonts["mono"])

        # True while the last line is a progress line that has no newline yet.
        self._progress_open = False

    # -- API ---------------------------------------------------------------
    def clear(self):
        def action():
            self.text.delete("1.0", tk.END)
            self._progress_open = False
        self._edit(action)

    def append(self, message, tag=None):
        def action():
            self._end_progress_line()
            for line in message.splitlines(keepends=True):
                self._insert_line(line, tag)
            self.text.see(tk.END)
        self._edit(action)

    def set_transcript(self, body, header=None, scroll_to_end=False):
        def action():
            self.text.delete("1.0", tk.END)
            self._progress_open = False
            if header:
                self.text.insert(tk.END, header + "\n\n", "head")
            for line in body.splitlines():
                self._insert_line(line + "\n", None)
            # A growing live transcript should follow the newest line; a
            # finished one starts at the top.
            if scroll_to_end:
                self.text.see(tk.END)
        self._edit(action)

    def replace_last_line(self, message):
        """Update the progress line at the end of the log in place.

        The first call starts that line - on a fresh one if the log did not end
        with a newline, so a half-written line is never overwritten - and the
        following calls replace it. The next append() ends it with a newline;
        without that the text after a finished download ran on straight after
        the last percentage ('...at 20.0 MB/sTranscription started.').

        The deletion stops at 'end-1c' on purpose. Tk also removes the
        PREVIOUS newline when a deletion that starts at a line start reaches
        END, so the old delete(..., END) swallowed the log line above the
        progress ('Downloading model ...') on the first update.
        """
        def action():
            if self._progress_open:
                self.text.delete("end-1c linestart", "end-1c")
            elif self.text.index("end-1c") != self.text.index("end-1c linestart"):
                self.text.insert(tk.END, "\n")
            self._insert_line(message, "system")
            self.text.see(tk.END)
            self._progress_open = True
        self._edit(action)

    # -- Internals ---------------------------------------------------------
    def _end_progress_line(self):
        if self._progress_open:
            self.text.insert(tk.END, "\n")
            self._progress_open = False

    def _insert_line(self, line, tag):
        """Colour timestamps and speakers.

        Handles both forms: '[00:12] [You]: text' from the finished transcript
        and '[You]: text' from the live preview, which has no timestamps.
        """
        if tag is not None:
            self.text.insert(tk.END, line, tag)
            return

        stripped = line.lstrip()
        if stripped.startswith("⚠"):
            self.text.insert(tk.END, line, "warn")
            return
        if stripped.startswith("[ERROR"):
            self.text.insert(tk.END, line, "error")
            return

        rest = line
        had_timestamp = False
        if stripped.startswith("[") and "]" in line:
            end = line.index("]") + 1
            if _looks_like_timestamp(line[:end]):
                self.text.insert(tk.END, line[:end], "ts")
                rest, had_timestamp = line[end:], True

        match = _SPEAKER_RE.match(rest)
        if match:
            speaker = match.group(1)
            style = "self" if "[You]" in speaker else "other"
            self.text.insert(tk.END, speaker, style)
            self.text.insert(tk.END, match.group(2), "body")
            return

        self.text.insert(tk.END, rest, "body" if had_timestamp else "system")

    def _edit(self, action):
        self.text.config(state=tk.NORMAL)
        try:
            action()
        finally:
            self.text.config(state=tk.DISABLED)


_SPEAKER_RE = re.compile(r"^(\s*\[[^\]]+\]\s*:)(.*)$", re.S)


def _looks_like_timestamp(token):
    inner = token.strip("[]")
    parts = inner.split(":")
    return 2 <= len(parts) <= 3 and all(part.isdigit() for part in parts)


# ======================================================================
class Field(tk.Frame):
    """A labelled entry or select field with an optional icon."""

    def __init__(self, parent, label, widget_factory, bg=T.CARD, hint=None, icon_name=None):
        super().__init__(parent, bg=bg)
        lbl_frame = tk.Frame(self, bg=bg)
        lbl_frame.pack(fill=tk.X, pady=(0, 4))
        self._icon_img = None
        if icon_name:
            self._icon_img = icons.get_icon(icon_name, size=15)
            tk.Label(lbl_frame, image=self._icon_img, bg=bg).pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(lbl_frame, text=label, bg=bg, fg=T.TEXT_DIM, font=T.fonts["small"],
                 anchor="w").pack(side=tk.LEFT, fill=tk.X)
        self.widget = widget_factory(self)
        self.widget.pack(fill=tk.X)
        if hint:
            tk.Label(self, text=hint, bg=bg, fg=T.TEXT_MUTE,
                     font=T.fonts["tiny"], anchor="w",
                     justify="left").pack(fill=tk.X, pady=(4, 0))
