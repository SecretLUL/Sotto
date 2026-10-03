"""Design tokens and ttk theme.

Tkinter looks dated only because its defaults come from the nineties. With a
custom clam-based theme, consistent colour and spacing values and a handful of
hand-drawn widgets (see widgets.py) it is perfectly possible to build a modern
interface without any additional library.

Every colour and spacing value lives here - no colour literals are allowed
anywhere else in the UI code.
"""

import math
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# ----------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------
BG = "#101319"            # window background - also the title bar (chrome.py)
BG_DEEP = "#0a0c10"       # deep background, input fields
CARD = "#181c25"          # card surface
CARD_HI = "#20252f"       # card under the pointer, selected tab
FIELD = "#0f1218"         # input field background
BORDER = "#272d3a"        # card border
BORDER_HI = "#3a4356"     # border on focus/hover

TEXT = "#f1f5f9"          # body text
TEXT_DIM = "#94a3b8"      # labels
TEXT_MUTE = "#64748b"     # hints, footnotes

ACCENT = "#38bdf8"        # primary cyan/sky blue
ACCENT_HI = "#7dd3fc"
ACCENT_LO = "#0284c7"
ON_ACCENT = "#04131d"     # text on an accent surface
REC = "#f43f5e"           # recording crimson
REC_HI = "#fb7185"
REC_TINT = "#2a141b"      # a surface that belongs to a running recording
OK = "#10b981"            # success emerald
WARN = "#f59e0b"          # warning amber
DANGER = "#f43f5e"

# The two voices, as in the logo and the transcript: your own, and everyone
# else's. The microphone's controls are drawn in the one, the system audio's
# in the other.
YOU = "#38bdf8"
OTHERS = "#34d399"

# Level meter
METER_LOW = "#10b981"
METER_MID = "#f59e0b"
METER_HIGH = "#f43f5e"
METER_OFF_LOW = "#142922"
METER_OFF_MID = "#2e2110"
METER_OFF_HIGH = "#33161c"
METER_BG = "#0a0c10"

# Speaker colours in the transcript
SPEAKER_SELF = YOU
SPEAKER_OTHER = OTHERS
TIMESTAMP = "#64748b"

# ----------------------------------------------------------------------
# Spacing (4 point grid) and display scaling
# ----------------------------------------------------------------------
# Every length in the interface is designed for a 96 dpi screen and passes
# through px(). Fonts follow the display by themselves (tk scaling turns points
# into pixels), pixel counts do not: at 150 % the caption of 'Start recording'
# was wider than its 170 px button. apply() sets SCALE and the values below.
SCALE = 1.0

_BASE_SPACING = (6, 10, 14, 18, 26, 34)
_BASE_RADII = (16, 9)

XS, SM, MD, LG, XL, XXL = _BASE_SPACING
RADIUS_CARD, RADIUS_CTRL = _BASE_RADII
RADIUS_PILL = 999


def px(length):
    """A length designed for a 96 dpi screen, in pixels on this one."""
    if not length:
        return 0
    return max(1, int(round(length * SCALE)))


def _set_scale(dpi):
    global SCALE, XS, SM, MD, LG, XL, XXL, RADIUS_CARD, RADIUS_CTRL
    SCALE = max(1.0, dpi / 96.0)
    XS, SM, MD, LG, XL, XXL = (px(value) for value in _BASE_SPACING)
    RADIUS_CARD, RADIUS_CTRL = (px(value) for value in _BASE_RADII)


def _display_dpi(root):
    """Pixels per inch of the screen the window is on (96 if it cannot tell)."""
    try:
        return float(root.winfo_fpixels("1i"))
    except tk.TclError:
        return 96.0


_FAMILY = "Segoe UI"
_MONO = "Consolas"

fonts = {}


def _pick_family(root, *candidates):
    available = set(tkfont.families(root))
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]


def apply(root):
    """Set up fonts, ttk styles and global options."""
    dpi = _display_dpi(root)
    _set_scale(dpi)
    try:
        root.tk.call('tk', 'scaling', max(1.2, dpi / 72.0))
    except Exception:
        pass

    family = _pick_family(root, "Segoe UI Variable Text", _FAMILY, "Helvetica")
    display = _pick_family(root, "Segoe UI Variable Display", _FAMILY, "Helvetica")
    mono = _pick_family(root, "Cascadia Mono", _MONO, "Courier")

    fonts.update({
        "display": tkfont.Font(root=root, family=display, size=20, weight="bold"),
        "title": tkfont.Font(root=root, family=family, size=13, weight="bold"),
        "card_title": tkfont.Font(root=root, family=family, size=11, weight="bold"),
        "section": tkfont.Font(root=root, family=family, size=10, weight="bold"),
        "timer": tkfont.Font(root=root, family=mono, size=18, weight="bold"),
        "body": tkfont.Font(root=root, family=family, size=11),
        "body_bold": tkfont.Font(root=root, family=family, size=11, weight="bold"),
        "small": tkfont.Font(root=root, family=family, size=10),
        "tiny": tkfont.Font(root=root, family=family, size=9),
        "mono": tkfont.Font(root=root, family=mono, size=11),
        "mono_small": tkfont.Font(root=root, family=mono, size=10),
        "button": tkfont.Font(root=root, family=family, size=11, weight="bold"),
    })

    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=TEXT, borderwidth=0,
                    focuscolor=ACCENT, font=fonts["body"])
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD)
    style.configure("TLabel", background=BG, foreground=TEXT, font=fonts["body"])
    style.configure("Card.TLabel", background=CARD, foreground=TEXT)
    style.configure("Dim.TLabel", background=CARD, foreground=TEXT_DIM,
                    font=fonts["small"])
    style.configure("Mute.TLabel", background=CARD, foreground=TEXT_MUTE,
                    font=fonts["tiny"])

    # --- Combo box -----------------------------------------------------
    # clam draws the arrow as a button of its own, framed and grey, beside
    # the field. Here a chevron sits inside the field instead; a read-only
    # combo box opens wherever it is clicked anyway.
    _chevron_element(root, style)
    for name in ("TCombobox", "Dark.TCombobox"):
        style.configure(name,
                        fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                        arrowcolor=TEXT_DIM, bordercolor=FIELD,
                        lightcolor=BORDER, darkcolor=BORDER,
                        selectbackground=FIELD, selectforeground=TEXT,
                        insertcolor=TEXT, padding=(px(12), px(8)),
                        arrowsize=px(14))
        style.map(name,
                  background=[("readonly", FIELD), ("active", FIELD),
                              ("pressed", FIELD), ("disabled", BG_DEEP)],
                  fieldbackground=[("readonly", FIELD), ("disabled", BG_DEEP)],
                  bordercolor=[("disabled", BG_DEEP)],
                  foreground=[("disabled", TEXT_MUTE)],
                  lightcolor=[("focus", ACCENT), ("hover", BORDER_HI)],
                  darkcolor=[("focus", ACCENT), ("hover", BORDER_HI)],
                  arrowcolor=[("disabled", TEXT_MUTE), ("hover", TEXT)])
        style.layout(name, [("Combobox.field", {"sticky": "nswe", "children": [
            ("Sotto.chevron", {"side": "right", "sticky": ""}),
            ("Combobox.padding", {"expand": "1", "sticky": "nswe", "children": [
                ("Combobox.textarea", {"sticky": "nswe"})]})]})])

    # Drop-down list (a classic Tk listbox, only reachable via the option DB)
    root.option_add("*TCombobox*Listbox.background", CARD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.borderWidth", 0)
    root.option_add("*TCombobox*Listbox.highlightThickness", 0)
    root.option_add("*TCombobox*Listbox.font", fonts["body"])

    # --- Entry ---------------------------------------------------------
    style.configure("Dark.TEntry",
                    fieldbackground=FIELD, foreground=TEXT, insertcolor=ACCENT,
                    bordercolor=FIELD, lightcolor=BORDER, darkcolor=BORDER,
                    padding=(px(12), px(8)), selectbackground=ACCENT_LO,
                    selectforeground="#ffffff")
    style.map("Dark.TEntry",
              fieldbackground=[("disabled", BG_DEEP)],
              foreground=[("disabled", TEXT_MUTE)],
              lightcolor=[("focus", ACCENT)],
              darkcolor=[("focus", ACCENT)])

    # --- Scrollbar -----------------------------------------------------
    # clam has no -width: its thumb is as thick as -arrowsize. That was 0 (to
    # hide arrows the layout below does not have anyway), which left a
    # scrollbar 1 px wide.
    style.configure("Dark.Vertical.TScrollbar",
                    background=BORDER, troughcolor=CARD, bordercolor=CARD,
                    arrowcolor=CARD, darkcolor=BORDER, lightcolor=BORDER,
                    arrowsize=px(12))
    style.map("Dark.Vertical.TScrollbar",
              background=[("active", BORDER_HI), ("pressed", ACCENT)])

    style.layout("Dark.Vertical.TScrollbar", [
        ("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
            ("Vertical.Scrollbar.thumb", {"expand": 1, "sticky": "nswe"})]})])

    return style


def _chevron_element(root, style):
    """The image element 'Sotto.chevron', made once per Tk interpreter.

    The images belong to `root` itself, not to icons' cache, which serves
    whichever Tk was created first; and they are kept on it, because ttk only
    holds their names.
    """
    from . import icons                  # Pillow only: no theme import back

    if getattr(root, "_sotto_chevron", None):
        return
    images = [tk.PhotoImage(master=root, data=icons.png("chevron_down", px(16), colour))
              for colour in (TEXT_DIM, TEXT_MUTE)]
    try:
        style.element_create("Sotto.chevron", "image", images[0],
                             ("disabled", images[1]), padding=(px(2), 0, px(10), 0))
    except tk.TclError:                  # this interpreter has it already
        pass
    root._sotto_chevron = images


# ----------------------------------------------------------------------
def round_rect(canvas, x0, y0, x1, y1, radius, **kwargs):
    """A rounded rectangle drawn as a smoothed polygon.

    Tk has no rounded rectangles; a polygon with smooth=True and duplicated
    corner points produces the result without any library.

    The corners go to whole pixels. Tk rounds the points of the smoothed
    outline, and on an edge at x.5 the tiny errors of the spline rounded some
    stretches one way and some the other: short dashes a pixel off the edge,
    the light marks under every card.
    """
    x0, y0, x1, y1 = (math.floor(value) for value in (x0, y0, x1, y1))
    radius = max(0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    points = [
        x0 + radius, y0, x1 - radius, y0, x1, y0,
        x1, y0 + radius, x1, y1 - radius, x1, y1,
        x1 - radius, y1, x0 + radius, y1, x0, y1,
        x0, y1 - radius, x0, y0 + radius, x0, y0,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def ellipsize(font, text, max_width):
    """`text` cut short with an ellipsis so that it fits max_width pixels in `font`."""
    if font.measure(text) <= max_width:
        return text
    low, high = 0, len(text)
    while low < high:                      # the longest prefix that still fits
        middle = (low + high + 1) // 2
        if font.measure(text[:middle].rstrip() + "…") <= max_width:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + "…"


def mix(color_a, color_b, t):
    """Linear blend of two #rrggbb colours (t = 0 -> a, 1 -> b)."""
    a = tuple(int(color_a[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(
        int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))
