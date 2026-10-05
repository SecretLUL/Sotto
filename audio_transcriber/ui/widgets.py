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
                 bg=T.BG, fill=T.CARD, stretch=False, icon_name=None,
                 icon_colour=None):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         height=T.px(64))
        self.padx, self.pady = T.px(pad[0]), T.px(pad[1])
        self._fill = fill
        self._title = title
        self._hint = hint
        self._icon_name = icon_name
        self._head_h = T.px(32) if title else 0
        self._shape = None
        self._icon_img = None
        self._syncing = False
        # stretch=True: the card fills the available space instead of deriving
        # its height from the content (used for the transcript pane).
        self._stretch = stretch

        self.body = tk.Frame(self, bg=fill)
        self._win = self.create_window(self.padx, self.pady + self._head_h,
                                       anchor="nw", window=self.body)
        self._title_end = self.padx
        if title:
            x_off = self.padx
            if icon_name:
                self._icon_img = icons.get_icon(icon_name, size=T.px(18),
                                                fg=icon_colour)
                self.create_image(self.padx, self.pady, anchor="nw", image=self._icon_img)
                x_off += T.px(26)
            self._title_item = self.create_text(
                x_off, self.pady - 1, anchor="nw", text=title,
                fill=T.TEXT, font=T.fonts["card_title"])
            self._title_end = x_off + T.fonts["card_title"].measure(title)

        # The line at the right of the heading. Always there: set_hint() used to
        # do nothing on a card made without one, and the Sources card - whose
        # hint says which output the system audio is taken from - was made
        # without.
        self._hint_text = hint or ""
        self._hint_item = self.create_text(
            0, self.pady + 1, anchor="ne", text=self._hint_text,
            fill=T.TEXT_MUTE, font=T.fonts["tiny"])

        self.bind("<Configure>", self._sync)
        self.body.bind("<Configure>", self._sync)

    def set_hint(self, text):
        self._hint_text = text or ""
        self._fit_hint(self.winfo_width())

    def _fit_hint(self, width):
        """Show the hint at the right edge, cut short before it reaches the heading."""
        text = self._hint_text
        if width > 1:
            room = width - self.padx - self._title_end - T.px(16)
            text = T.ellipsize(T.fonts["tiny"], text, max(0, room))
        self.itemconfigure(self._hint_item, text=text,
                           fill=T.WARN if text.startswith("⚠") else T.TEXT_MUTE)
        self.coords(self._hint_item, max(width, 1) - self.padx, self.pady + 1)

    def _sync(self, _event=None):
        if self._syncing:
            return
        self._syncing = True
        try:
            width = max(self.winfo_width(), 1)
            inner_w = max(1, width - 2 * self.padx)

            if self._stretch:
                height = max(self.winfo_height(), T.px(120))
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

            self._fit_hint(width)
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

    # kind: (fill, fill under the pointer, fill while pressed, text, border).
    # An icon takes the colour of the text.
    _PALETTE = {
        "accent": (T.ACCENT, T.ACCENT_HI, T.ACCENT_LO, T.ON_ACCENT, None),
        "record": (T.REC, T.REC_HI, "#d93a48", "#ffffff", None),
        "stop": (T.REC_TINT, "#3a1a23", "#22101a", T.REC_HI, T.REC),
        "ghost": (None, T.CARD_HI, T.BG_DEEP, T.TEXT_DIM, T.BORDER),
        "quiet": (None, T.CARD_HI, T.BG_DEEP, T.TEXT_DIM, None),
    }

    PAD_X = 14                # air at each side of the content, at 96 dpi

    def __init__(self, parent, text="", command=None, kind="ghost",
                 width=150, height=38, icon="", icon_name=None, bg=T.CARD,
                 radius=None, state="normal"):
        # width and height are designed for a 96 dpi screen (T.px). The width
        # is only the least the button gets: it grows to fit its caption, so a
        # larger font or display scaling can no longer cut the text off.
        self._min_width = T.px(width)
        self._height = T.px(height)
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=self._min_width, height=self._height)
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
        self._fit()
        self._render()

    # -- Size ------------------------------------------------------------
    def _icon_size(self):
        return min(T.px(20), self._height - T.px(14))

    def content_width(self):
        """Width of icon, gap and caption - the least the button has to be."""
        font = T.fonts["button"]
        if self._icon_name and self._icon_name in icons._RENDERERS:
            text_w = font.measure(self._text) if self._text else 0
            return self._icon_size() + (T.px(8) if self._text else 0) + text_w
        label = f"{self._icon}  {self._text}".strip() if self._icon else self._text
        return font.measure(label)

    def _fit(self):
        width = max(self._min_width, self.content_width() + 2 * T.px(self.PAD_X))
        if int(tk.Canvas.cget(self, "width")) != width:
            tk.Canvas.configure(self, width=width)

    # -- API compatible with ttk widgets --------------------------------
    def configure(self, cnf=None, **kw):
        state = kw.pop("state", None)
        text = kw.pop("text", None)
        kind = kw.pop("kind", None)
        icon_name = kw.pop("icon_name", None)
        width = kw.pop("width", None)
        height = kw.pop("height", None)
        if state is not None:
            self._state = str(state)
            self._hover = self._pressed = False
        if text is not None:
            self._text = text
        if kind is not None:
            self._kind = kind
        if icon_name is not None:
            self._icon_name = icon_name
        if width is not None:
            self._min_width = T.px(width)
        if height is not None:
            self._height = T.px(height)
            kw["height"] = self._height
        result = super().configure(cnf, **kw) if (cnf or kw) else None
        if any(value is not None for value in
               (state, text, kind, icon_name, width, height)):
            self._fit()
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
            icon_sz = self._icon_size()
            self._img_obj = icons.get_icon(self._icon_name, size=icon_sz, fg=fg)

            font = T.fonts["button"]
            text_w = font.measure(self._text) if self._text else 0
            gap = T.px(8) if self._text else 0
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

    TRACK_W, TRACK_H, KNOB_R = 38, 20, 7          # at 96 dpi

    def __init__(self, parent, text="", variable=None, command=None,
                 bg=T.CARD, width=None):
        self.var = variable if variable is not None else tk.BooleanVar(value=False)
        self._command = command
        self._text = text
        self._hover = False
        self._track_w, self._track_h, self._knob_r = (
            T.px(value) for value in (self.TRACK_W, self.TRACK_H, self.KNOB_R))

        font = T.fonts["body"]
        text_w = font.measure(text) if text else 0
        natural = self._track_w + T.px(10) + text_w + T.px(4)
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=T.px(width) if width else natural,
                         height=max(self._track_h, T.px(22)))

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
        T.round_rect(self, 1, mid - self._track_h / 2, self._track_w,
                     mid + self._track_h / 2, self._track_h / 2,
                     fill=track, outline=T.BORDER if not on else "", width=1)

        knob = self._knob_r
        cx = (self._track_w - knob - T.px(3)) if on else (knob + T.px(4))
        self.create_oval(cx - knob, mid - knob, cx + knob, mid + knob,
                         fill="#ffffff" if on else T.TEXT_DIM, outline="")

        if self._text:
            self.create_text(self._track_w + T.px(10), mid, anchor="w", text=self._text,
                             fill=T.TEXT if on else T.TEXT_DIM,
                             font=T.fonts["body"])


# ======================================================================
class Slider(tk.Canvas):
    """A slider with track, fill and knob. Supports dragging and clicking."""

    HEIGHT = 24                  # at 96 dpi
    KNOB_R = 7

    def __init__(self, parent, from_=-20.0, to=20.0, value=0.0, command=None,
                 width=280, bg=T.CARD, centered=True, colour=T.ACCENT):
        self._height = T.px(self.HEIGHT)
        self._knob_r = T.px(self.KNOB_R)
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=T.px(width), height=self._height)
        self.from_, self.to = from_, to
        self._value = value
        self._command = command
        self._colour = colour
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
        usable = max(1, self.winfo_width() - 2 * self._knob_r - 2)
        share = (x - self._knob_r - 1) / usable
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
        width = max(self.winfo_width(), T.px(60))
        mid = self._height / 2
        x0, x1 = self._knob_r + 1, width - self._knob_r - 1
        span = x1 - x0
        half = T.px(2)

        T.round_rect(self, x0, mid - half, x1, mid + half, half,
                     fill=T.BG_DEEP, outline="")

        share = (self._value - self.from_) / float(self.to - self.from_)
        knob_x = x0 + share * span

        if self._centered:
            centre = x0 + span * (0.0 - self.from_) / float(self.to - self.from_)
            left, right = min(centre, knob_x), max(centre, knob_x)
            self.create_line(centre, mid - T.px(5), centre, mid + T.px(5),
                             fill=T.BORDER_HI, width=1)
        else:
            left, right = x0, knob_x

        if abs(right - left) > 1:
            T.round_rect(self, left, mid - half, right, mid + half, half,
                         fill=self._colour, outline="")

        radius = self._knob_r + (1 if self._hover or self._dragging else 0)
        self.create_oval(knob_x - radius, mid - radius,
                         knob_x + radius, mid + radius,
                         fill="#ffffff" if self._hover or self._dragging else T.TEXT,
                         outline=self._colour if self._dragging else "", width=2)


# ======================================================================
class Meter(tk.Canvas):
    """A level meter with blocks, peak marker and dB readout.

    The ballistics are time based (not tied to the call frequency) and
    0 dBFS corresponds to full scale. The normal range is drawn in `colour` -
    the colour of the voice the source carries - and the loud end in amber
    and red, as on any meter.
    """

    MIN_DB, MAX_DB = -60.0, 0.0
    DECAY_DB_S = 30.0
    PEAK_HOLD_S = 1.2
    PEAK_FALL_DB_S = 14.0
    READOUT_W = 66

    def __init__(self, parent, width=320, height=14, blocks=44, bg=T.CARD,
                 colour=T.METER_LOW):
        width, height = T.px(width), T.px(height)
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=width, height=height)
        self._height = height
        self._block_count = blocks
        self._colour = colour
        # Room for the readout: at least READOUT_W (at 96 dpi), and always what
        # the widest reading takes in the font it is set in.
        self._readout_w = max(T.px(self.READOUT_W),
                              T.fonts["mono_small"].measure("-60.0 dB") + T.px(8))
        self._bar_w = max(T.px(20), width - self._readout_w)
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
        self._bar_w = max(T.px(20), width - self._readout_w)
        self._blocks = []

        gap = T.px(2)
        step = self._bar_w / self._block_count
        for index in range(self._block_count):
            x0 = index * step
            share = (index + 1) / self._block_count
            if share <= 0.68:
                on, off = self._colour, T.mix(T.METER_BG, self._colour, 0.16)
            elif share <= 0.88:
                on, off = T.METER_MID, T.METER_OFF_MID
            else:
                on, off = T.METER_HIGH, T.METER_OFF_HIGH
            item = self.create_rectangle(x0, 1, x0 + step - gap, self._height - 1,
                                         fill=off, outline="")
            self._blocks.append((item, x0, on, off))

        self._peak_item = self.create_line(0, 0, 0, self._height, fill="#ffffff",
                                           width=T.px(2), state="hidden")
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
        self._min_width = T.px(width)
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=self._min_width, height=T.px(height))
        self._text = "ready"
        self._colour = T.TEXT_MUTE
        self._pulse = False
        self._phase = 0.0
        self._render()

    def set(self, text, colour=T.TEXT_MUTE, pulse=False):
        self._text, self._colour, self._pulse = text, colour, pulse
        if not pulse:
            self._phase = 0.0
        # A longer status ('processing recovered recording...') used to run off
        # the left edge of a canvas of fixed width.
        needed = T.fonts["small"].measure(text) + T.px(32)
        width = max(self._min_width, needed)
        if int(self["width"]) != width:
            self.configure(width=width)
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
        pill_w = text_w + T.px(32)
        x0 = max(0, width - pill_w)

        # Glassmorphic rounded pill background
        T.round_rect(self, x0, 1, width - 1, height - 1, (height - 2) / 2,
                     fill=T.CARD, outline=T.BORDER, width=1)

        radius = T.px(4)
        cx = x0 + T.px(14)
        self.create_oval(cx - radius, mid - radius, cx + radius, mid + radius,
                         fill=dot, outline="")
        if self._pulse:
            halo = T.mix(self["bg"], self._colour, alpha * 0.25)
            self.create_oval(cx - radius - T.px(4), mid - radius - T.px(4),
                             cx + radius + T.px(4), mid + radius + T.px(4),
                             outline=halo, width=1)
        self.create_text(x0 + T.px(26), mid, anchor="w", text=self._text,
                         fill=T.TEXT_DIM, font=T.fonts["small"])


# ======================================================================
class ProgressBar(tk.Canvas):
    """A slim rounded progress bar - the model download on the Settings tab.

    set(fraction) with 0..1 fills it. The fill glides towards the value
    (tick), so ten reports a second read as one movement rather than steps, and
    a sheen runs along it while it is under way. set(None) means "busy, length
    unknown" (unpacking an archive): a segment sweeps across instead.
    tick() is driven by the window's ticker.
    """

    GLIDE = 8.0               # how fast the fill catches up, per second
    SWEEP_S = 1.3             # one pass of the busy segment
    SHEEN_S = 1.8             # one pass of the sheen
    SHEEN_W = 70              # at 96 dpi

    def __init__(self, parent, height=8, bg=T.CARD, colour=T.ACCENT):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         width=T.px(120), height=T.px(height))
        self._colour = colour
        self._target = 0.0
        self._shown = 0.0
        self._phase = 0.0
        self.bind("<Configure>", lambda _e: self._render())
        self._render()

    @property
    def fraction(self):
        """The value last set; None while busy without one."""
        return self._target

    def set(self, fraction):
        if fraction is None:
            self._target = None
        else:
            fraction = max(0.0, min(1.0, float(fraction)))
            # A new file starts at the beginning again: no gliding backwards.
            if self._target is None or fraction < self._shown:
                self._shown = fraction
            self._target = fraction
        self._render()

    def tick(self, dt=0.04):
        if not self.winfo_ismapped():
            return
        self._phase += dt
        if self._target is not None:
            gap = self._target - self._shown
            self._shown = (self._target if abs(gap) < 0.001
                           else self._shown + gap * min(1.0, dt * self.GLIDE))
        self._render()

    def _render(self):
        self.delete("all")
        width = max(self.winfo_width(), 2)
        height = int(self["height"])
        radius = height / 2
        T.round_rect(self, 0, 0, width, height, radius,
                     fill=T.mix(self["bg"], T.BG_DEEP, 0.85),
                     outline=T.BORDER, width=1)

        if self._target is None:
            share = (self._phase / self.SWEEP_S) % 1.0
            eased = 0.5 - 0.5 * math.cos(math.pi * share)
            segment = width * 0.3
            left = -segment + (width + segment) * eased
            x0, x1 = max(0, left), min(width, left + segment)
            if x1 - x0 >= 2:
                T.round_rect(self, x0, 0, x1, height, radius,
                             fill=self._colour, outline="")
            return

        filled = width * self._shown
        if filled < 2:
            return
        T.round_rect(self, 0, 0, filled, height, radius,
                     fill=self._colour, outline="")
        # A lighter top edge gives the fill some body.
        if filled > height:
            self.create_line(radius, 1.5, filled - radius, 1.5, width=1,
                             fill=T.mix(self._colour, "#ffffff", 0.35))
        # The sheen, while there is still something to come.
        if self._shown < 0.999 and filled > 2 * height:
            sheen = T.px(self.SHEEN_W)
            share = (self._phase / self.SHEEN_S) % 1.0
            centre = -sheen + (filled + 2 * sheen) * share
            # Brightest in the middle, fading towards both sides: the widest,
            # faintest band first, the narrow bright one on top.
            steps = 6
            for step in range(steps, 0, -1):
                half = sheen / 2 * step / steps
                x0 = max(radius, centre - half)
                x1 = min(filled - radius, centre + half)
                if x1 - x0 >= 1:
                    self.create_rectangle(
                        x0, 1, x1, height - 1, outline="",
                        fill=T.mix(self._colour, "#ffffff",
                                   0.14 * (1 - (step - 1) / steps)))


# ======================================================================
class Transcript(tk.Frame):
    """The transcript pane with speaker colours and a slim scrollbar.

    The scrollbar shows only while there is something to scroll - before, an
    empty pane had a grey bar its whole height - and an empty pane says what
    will appear in it.
    """

    PLACEHOLDER = ("Nothing here yet. Start a recording or upload a file, and "
                   "the transcript appears here.")

    def __init__(self, parent, height=14, bg=T.CARD):
        super().__init__(parent, bg=bg)
        self.text = tk.Text(
            self, height=height, wrap="word", relief="flat", bd=0,
            bg=bg, fg=T.TEXT, insertbackground=T.ACCENT,
            selectbackground=T.mix(bg, T.ACCENT, 0.35), selectforeground=T.TEXT,
            font=T.fonts["mono"], padx=T.px(2), pady=T.px(2), highlightthickness=0,
            spacing1=T.px(2), spacing3=T.px(2), state=tk.DISABLED, cursor="arrow")
        self.scroll = ttk.Scrollbar(self, orient="vertical",
                                    style="Dark.Vertical.TScrollbar",
                                    command=self.text.yview)
        self.text.configure(yscrollcommand=self._on_scroll)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.placeholder = WrapLabel(self, text=self.PLACEHOLDER, bg=bg,
                                     font=T.fonts["small"], anchor="center",
                                     justify="center")
        self._show_placeholder()

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
            self._show_placeholder()

    def _show_placeholder(self):
        if self.text.compare("end-1c", "==", "1.0"):
            self.placeholder.place(relx=0.5, rely=0.4, anchor="center",
                                   relwidth=0.8)
        else:
            self.placeholder.place_forget()

    def _on_scroll(self, first, last):
        """The scrollbar's set(), showing the bar only while it has a use.

        No back and forth: with the bar the text is narrower and only gets
        longer, without it wider and only shorter. Packed or not is asked of
        the packer - winfo_ismapped() is also false while the tab is hidden.
        """
        self.scroll.set(first, last)
        needed = float(first) > 0.0 or float(last) < 1.0
        packed = bool(self.scroll.winfo_manager())
        if needed and not packed:
            self.scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(T.XS, 0),
                             before=self.text)
        elif packed and not needed:
            self.scroll.pack_forget()


_SPEAKER_RE = re.compile(r"^(\s*\[[^\]]+\]\s*:)(.*)$", re.S)


def _looks_like_timestamp(token):
    inner = token.strip("[]")
    parts = inner.split(":")
    return 2 <= len(parts) <= 3 and all(part.isdigit() for part in parts)


# ======================================================================
class WrapLabel(tk.Label):
    """A label that wraps its text to the width the layout gives it.

    A plain Label asks for the width of its longest line, and in a grid cell
    or a packed row that request wins: a long hint stretched the columns above
    it - the Processing Options card pushed its last switches out of the window
    - or was simply cut off at the edge of its card. This one asks for next to
    nothing and wraps to whatever width it is then given, so it has to be
    stretched: pack(fill=X) or grid(sticky="ew").
    """

    def __init__(self, parent, text="", bg=T.CARD, fg=T.TEXT_MUTE, font=None, **kw):
        kw.setdefault("anchor", "w")
        kw.setdefault("justify", "left")
        super().__init__(parent, text=text, bg=bg, fg=fg,
                         font=font or T.fonts["tiny"], width=1, **kw)
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        # A few pixels of the width are the label's own border and padding.
        width = max(1, event.width - 4)
        if event.width > 1 and width != int(self.cget("wraplength") or 0):
            self.configure(wraplength=width)


# ======================================================================
class Flow(tk.Frame):
    """Children side by side, wrapping onto the next line where the width runs out.

    A row of switches in a grid has a fixed number of columns and so a fixed
    least width: what fitted at 100 % did not at 150 % or with a longer caption,
    and the switches at the end were just outside the window. Here they move
    down instead. Create the children with the Flow as their parent, then add()
    them in order.
    """

    def __init__(self, parent, bg=T.CARD, gap=None, line_gap=None):
        super().__init__(parent, bg=bg, width=1, height=1)
        self._gap = T.MD if gap is None else gap
        self._line_gap = T.SM if line_gap is None else line_gap
        self._items = []
        self._width = 0
        self.bind("<Configure>", self._on_configure)

    def add(self, widget):
        self._items.append(widget)
        self._layout()

    def fit(self, width):
        """Lay the children out for a Flow that is `width` pixels wide."""
        self._width = width
        self._layout()

    def _on_configure(self, event):
        if event.width != self._width:
            self.fit(event.width)

    def _layout(self):
        limit = self._width if self._width > 1 else 0     # 0: not laid out yet
        x = y = line_height = 0
        for item in self._items:
            width, height = item.winfo_reqwidth(), item.winfo_reqheight()
            if limit and x and x + width > limit:
                x, y, line_height = 0, y + line_height + self._line_gap, 0
            item.place(x=x, y=y)
            x += width + self._gap
            line_height = max(line_height, height)
        needed = y + line_height
        # place() does not take part in size negotiation: say how tall this is.
        if needed and needed != int(self.cget("height")):
            self.configure(height=needed)


# ======================================================================
class Scroller(tk.Frame):
    """A page that scrolls where it is taller than the room it is given.

    The tabs were plain frames, and pack squeezes what does not fit: where the
    screen cannot hold the window at full height - at 150 % on a 1080p screen,
    at 100 % on a 1366x768 laptop - the last card of the Settings tab was cut
    off. Here the page keeps its height and a scrollbar shows while needed.
    Build the content into .body.

    While the page scrolls, the wheel scrolls it wherever the pointer is: a
    combo box or slider that passes under the pointer on the way no longer
    catches the wheel and changes its value - a combo box with the keyboard
    focus still does. While everything fits, the widgets keep the wheel.
    """

    def __init__(self, parent, bg=T.BG):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0,
                                width=1, height=1)
        self.body = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window(0, 0, anchor="nw", window=self.body)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical",
                                       style="Dark.Vertical.TScrollbar",
                                       command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrolling = False

        # A bindtag of its own, put in front of those of everything on the
        # page (_claim_wheel), so that the page sees the wheel first.
        self._tag = f"Scroller{id(self)}"
        sequences = ["<MouseWheel>"]
        if self.tk.call("tk", "windowingsystem") == "x11":
            sequences += ["<Button-4>", "<Button-5>"]       # the wheel on X11
        for sequence in sequences:
            self.bind_class(self._tag, sequence, self._on_wheel)

        self.canvas.bind("<Configure>", self._sync)
        self.body.bind("<Configure>", self._sync)

    def _sync(self, _event=None):
        width, room = self.canvas.winfo_width(), self.canvas.winfo_height()
        needed = self.body.winfo_reqheight()
        self.canvas.itemconfigure(self._win, width=width)
        # Ask for the whole page, as the plain frame did: the window still
        # knows what showing all of it would take.
        self.canvas.configure(scrollregion=(0, 0, width, needed),
                              width=self.body.winfo_reqwidth(), height=needed)
        scrolling = 1 < room < needed
        if scrolling != self.scrolling:
            self.scrolling = scrolling
            if scrolling:
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(T.XS, 0),
                                    before=self.canvas)
            else:
                self.scrollbar.pack_forget()
                self.canvas.yview_moveto(0)
        # Every time, so that content added later is covered as well.
        self._claim_wheel(self)

    def _claim_wheel(self, widget):
        tags = widget.bindtags()
        if self._tag not in tags:
            widget.bindtags((self._tag,) + tags)
        for child in widget.winfo_children():
            self._claim_wheel(child)

    def _on_wheel(self, event):
        if not self.scrolling:
            return None                          # nothing to scroll
        widget = event.widget
        if isinstance(widget, ttk.Combobox) \
                and str(widget) == str(self.tk.call("focus")):
            return None                          # it is being used on purpose
        up = event.num == 4 or event.delta > 0
        self.canvas.yview_scroll(-1 if up else 1, "units")
        return "break"


# ======================================================================
class Field(tk.Frame):
    """A labelled entry or select field with an optional icon.

    `aside` makes what belongs next to the field - a button, a frame of them -
    on the same line as the field itself, above the hint. Placed beside the
    whole Field, such buttons sat level with the hint instead.
    """

    def __init__(self, parent, label, widget_factory, bg=T.CARD, hint=None,
                 icon_name=None, aside=None):
        super().__init__(parent, bg=bg)
        lbl_frame = tk.Frame(self, bg=bg)
        lbl_frame.pack(fill=tk.X, pady=(0, T.px(4)))
        self._icon_img = None
        if icon_name:
            self._icon_img = icons.get_icon(icon_name, size=T.px(15))
            tk.Label(lbl_frame, image=self._icon_img, bg=bg).pack(
                side=tk.LEFT, padx=(0, T.px(5)))
        tk.Label(lbl_frame, text=label, bg=bg, fg=T.TEXT_DIM, font=T.fonts["small"],
                 anchor="w").pack(side=tk.LEFT, fill=tk.X)
        row = tk.Frame(self, bg=bg)
        row.pack(fill=tk.X)
        self.aside = None
        if aside is not None:
            # Packed first, so that a narrow window squeezes the field, not it.
            self.aside = aside(row)
            self.aside.pack(side=tk.RIGHT, padx=(T.SM, 0))
        self.widget = widget_factory(row)
        self.widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if hint:
            self.hint = WrapLabel(self, text=hint, bg=bg)
            self.hint.pack(fill=tk.X, pady=(T.px(4), 0))


# ======================================================================
class Tabs(tk.Frame):
    """Pages under a bar of tabs, one page shown at a time.

    It replaces ttk.Notebook, which in the clam theme framed every page with a
    light line and drew the selected tab smaller than the others, so the row
    jumped at every click. Here the tabs are the segments of one bar; all of
    them keep their size, and the selected one is lifted out. add() and
    select() work like the notebook's, which is all the window and the tests
    ask of it.

    Create the pages with the Tabs as their parent.
    """

    HEIGHT = 42                  # at 96 dpi
    PAD_X = 20                   # air at each side of a tab's content
    INSET = 4                    # between the bar's edge and a lifted tab
    ICON = 18

    def __init__(self, parent, bg=T.BG):
        super().__init__(parent, bg=bg)
        self.bar = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0,
                             height=T.px(self.HEIGHT))
        self.bar.pack(fill=tk.X, pady=(0, T.LG))
        self._tabs = []                  # (page, text, icon_name)
        self._current = None
        self._hover = None
        self.bar.bind("<Motion>", self._on_motion)
        self.bar.bind("<Leave>", lambda _e: self._set_hover(None))
        self.bar.bind("<Button-1>", self._on_click)

    # -- API like ttk.Notebook ------------------------------------------
    def add(self, page, text="", icon_name=None):
        self._tabs.append((page, text, icon_name))
        self.bar.configure(width=self._bar_width())
        if self._current is None:
            self.select(page)
        else:
            self._draw()

    def select(self, page=None):
        """Show `page`. Without one: the page that is shown."""
        if page is None or page is self._current:
            return self._current
        if self._current is not None:
            self._current.pack_forget()
        self._current = page
        page.pack(fill=tk.BOTH, expand=True)
        self._draw()
        self.event_generate("<<NotebookTabChanged>>")
        return page

    def pages(self):
        return [page for page, _text, _icon in self._tabs]

    # -- Geometry -------------------------------------------------------
    def _tab_width(self):
        """One width for all: the widest caption decides."""
        font = T.fonts["button"]
        widest = max((font.measure(text) for _page, text, _icon in self._tabs),
                     default=0)
        return widest + T.px(self.ICON) + T.px(8) + 2 * T.px(self.PAD_X)

    def _bar_width(self):
        return len(self._tabs) * self._tab_width() + 2 * T.px(self.INSET)

    def _tab_at(self, x):
        index = int((x - T.px(self.INSET)) // max(1, self._tab_width()))
        return index if 0 <= index < len(self._tabs) and x < self._bar_width() else None

    # -- Events ---------------------------------------------------------
    def _on_motion(self, event):
        self._set_hover(self._tab_at(event.x))

    def _set_hover(self, index):
        if index != self._hover:
            self._hover = index
            self.bar.configure(cursor="hand2" if index is not None else "")
            self._draw()

    def _on_click(self, event):
        index = self._tab_at(event.x)
        if index is not None:
            self.select(self._tabs[index][0])

    # -- Rendering ------------------------------------------------------
    def _draw(self):
        bar = self.bar
        bar.delete("all")
        height, inset = T.px(self.HEIGHT), T.px(self.INSET)
        width = self._tab_width()
        T.round_rect(bar, 0.5, 0.5, self._bar_width() - 0.5, height - 0.5,
                     T.RADIUS_CTRL + inset, fill=T.FIELD, outline=T.BORDER, width=1)

        font = T.fonts["button"]
        icon_size = T.px(self.ICON)
        for index, (page, text, icon_name) in enumerate(self._tabs):
            x0 = inset + index * width
            selected = page is self._current
            hover = index == self._hover and not selected
            if selected:
                T.round_rect(bar, x0, inset, x0 + width, height - inset, T.RADIUS_CTRL,
                             fill=T.CARD_HI, outline=T.BORDER_HI, width=1)
            elif hover:
                T.round_rect(bar, x0, inset, x0 + width, height - inset, T.RADIUS_CTRL,
                             fill=T.CARD, outline="")

            colour = T.TEXT if selected or hover else T.TEXT_DIM
            content = icon_size + T.px(8) + font.measure(text)
            x = x0 + (width - content) / 2
            if icon_name:
                icon_colour = T.ACCENT if selected else (T.TEXT_DIM if hover else T.TEXT_MUTE)
                bar.create_image(x + icon_size / 2, height / 2,
                                 image=icons.get_icon(icon_name, size=icon_size,
                                                      fg=icon_colour))
            bar.create_text(x + icon_size + T.px(8), height / 2, anchor="w",
                            text=text, fill=colour, font=font)


# ======================================================================
class IconBadge(tk.Canvas):
    """An icon on a plate tinted in its colour - the mark of a source row."""

    def __init__(self, parent, icon_name, colour, size=34, bg=T.CARD):
        side = T.px(size)
        super().__init__(parent, width=side, height=side, bg=bg,
                         highlightthickness=0, bd=0)
        self._size = size
        self.set(icon_name, colour)

    def set(self, icon_name, colour):
        """Another icon and colour - the state of the model on the Settings tab."""
        side = int(self["width"])
        self.delete("all")
        T.round_rect(self, 0.5, 0.5, side - 0.5, side - 0.5, side * 0.3,
                     fill=T.mix(self["bg"], colour, 0.16), outline="")
        self._image = icons.get_icon(icon_name, size=T.px(round(self._size * 0.56)),
                                     fg=colour)
        self.create_image(side / 2, side / 2, image=self._image)
