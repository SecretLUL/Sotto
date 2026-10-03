"""Design tokens and ttk theme.

Tkinter looks dated only because its defaults come from the nineties. With a
custom clam-based theme, consistent colour and spacing values and a handful of
hand-drawn widgets (see widgets.py) it is perfectly possible to build a modern
interface without any additional library.

Every colour and spacing value lives here - no colour literals are allowed
anywhere else in the UI code.
"""

import tkinter.font as tkfont
from tkinter import ttk

# ----------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------
BG = "#121418"            # window background
BG_DEEP = "#0a0c0e"       # deep background, input fields
CARD = "#1b1e26"          # card surface
CARD_HI = "#242834"       # card under the pointer
FIELD = "#14171f"         # input field background
BORDER = "#2a2f3d"        # card border
BORDER_HI = "#3e4659"     # border on focus/hover

TEXT = "#f1f5f9"          # body text
TEXT_DIM = "#94a3b8"      # labels
TEXT_MUTE = "#64748b"     # hints, footnotes

ACCENT = "#38bdf8"        # primary cyan/sky blue
ACCENT_HI = "#7dd3fc"
ACCENT_LO = "#0284c7"
REC = "#f43f5e"           # recording crimson
REC_HI = "#fb7185"
OK = "#10b981"            # success emerald
WARN = "#f59e0b"          # warning amber
DANGER = "#f43f5e"

# Level meter
METER_LOW = "#10b981"
METER_MID = "#f59e0b"
METER_HIGH = "#f43f5e"
METER_OFF_LOW = "#142922"
METER_OFF_MID = "#2e2110"
METER_OFF_HIGH = "#33161c"
METER_BG = "#0a0c0e"

# Speaker colours in the transcript
SPEAKER_SELF = "#38bdf8"
SPEAKER_OTHER = "#34d399"
TIMESTAMP = "#64748b"

# ----------------------------------------------------------------------
# Spacing (4 point grid)
# ----------------------------------------------------------------------
XS, SM, MD, LG, XL, XXL = 6, 10, 14, 18, 26, 34

RADIUS_CARD = 16
RADIUS_CTRL = 9
RADIUS_PILL = 999

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
    try:
        dpi = root.winfo_fpixels('1i')
        scaling = max(1.2, dpi / 72.0)
        root.tk.call('tk', 'scaling', scaling)
    except Exception:
        pass

    family = _pick_family(root, "Segoe UI Variable Text", _FAMILY, "Helvetica")
    display = _pick_family(root, "Segoe UI Variable Display", _FAMILY, "Helvetica")
    mono = _pick_family(root, "Cascadia Mono", _MONO, "Courier")

    fonts.update({
        "display": tkfont.Font(root=root, family=display, size=20, weight="bold"),
        "title": tkfont.Font(root=root, family=family, size=13, weight="bold"),
        "section": tkfont.Font(root=root, family=family, size=10, weight="bold"),
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
    for name in ("TCombobox", "Dark.TCombobox"):
        style.configure(name,
                        fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                        arrowcolor=TEXT_DIM, bordercolor=FIELD,
                        lightcolor=BORDER, darkcolor=BORDER,
                        selectbackground=FIELD, selectforeground=TEXT,
                        insertcolor=TEXT, padding=(12, 8), arrowsize=14)
        style.map(name,
                  background=[("readonly", FIELD), ("active", FIELD),
                              ("pressed", FIELD), ("disabled", BG_DEEP)],
                  fieldbackground=[("readonly", FIELD), ("disabled", BG_DEEP)],
                  bordercolor=[("disabled", BG_DEEP)],
                  foreground=[("disabled", TEXT_MUTE)],
                  lightcolor=[("focus", ACCENT), ("hover", BORDER_HI)],
                  darkcolor=[("focus", ACCENT), ("hover", BORDER_HI)],
                  arrowcolor=[("disabled", TEXT_MUTE), ("hover", TEXT)])

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
                    padding=(12, 8), selectbackground=ACCENT_LO,
                    selectforeground="#ffffff")
    style.map("Dark.TEntry",
              fieldbackground=[("disabled", BG_DEEP)],
              foreground=[("disabled", TEXT_MUTE)],
              lightcolor=[("focus", ACCENT)],
              darkcolor=[("focus", ACCENT)])

    # --- Scrollbar -----------------------------------------------------
    style.configure("Dark.Vertical.TScrollbar",
                    background=BORDER, troughcolor=CARD, bordercolor=CARD,
                    arrowcolor=CARD, darkcolor=BORDER, lightcolor=BORDER,
                    arrowsize=0, width=12)
    style.map("Dark.Vertical.TScrollbar",
              background=[("active", BORDER_HI), ("pressed", ACCENT)])

    style.layout("Dark.Vertical.TScrollbar", [
        ("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
            ("Vertical.Scrollbar.thumb", {"expand": 1, "sticky": "nswe"})]})])

    # --- Notebook (Tabs) -----------------------------------------------
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=[0, 0, 0, 8])
    style.configure("TNotebook.Tab", background=CARD, foreground=TEXT_DIM,
                    padding=(22, 11), font=fonts["button"], borderwidth=0,
                    focuscolor="", lightcolor=BORDER, darkcolor=BORDER)

    style.map("TNotebook.Tab",
              background=[("selected", CARD_HI), ("active", BORDER)],
              foreground=[("selected", ACCENT), ("active", TEXT)],
              focuscolor=[("selected", ""), ("active", ""), ("focus", "")],
              lightcolor=[("selected", ACCENT)],
              darkcolor=[("selected", ACCENT)])

    style.layout("TNotebook.Tab", [
        ("Notebook.tab", {
            "sticky": "nswe",
            "children": [
                ("Notebook.padding", {
                    "side": "top",
                    "sticky": "nswe",
                    "children": [
                        ("Notebook.label", {"side": "top", "sticky": ""})
                    ]
                })
            ]
        })
    ])

    return style




# ----------------------------------------------------------------------
def round_rect(canvas, x0, y0, x1, y1, radius, **kwargs):
    """A rounded rectangle drawn as a smoothed polygon.

    Tk has no rounded rectangles; a polygon with smooth=True and duplicated
    corner points produces the result without any library.
    """
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
