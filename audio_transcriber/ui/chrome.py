"""What the window looks like outside of Tk's reach: title bar, icon, taskbar.

Windows draws the title bar itself - light, whatever the window looks like,
with Tk's feather in the corner - and files a window under the program that
runs it: python.exe or pythonw.exe, so the taskbar showed Python's icon.
"""

import ctypes
import tkinter as tk

from . import icons
from . import theme as T

APP_ID = "SecretLUL.Sotto"

# Tk picks the size that suits the title bar, the taskbar and Alt+Tab.
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

# DwmSetWindowAttribute
_DARK_MODE = 20                  # DWMWA_USE_IMMERSIVE_DARK_MODE
_DARK_MODE_BEFORE_20H1 = 19      # its number in Windows 10 up to 1909
_BORDER_COLOUR = 34              # Windows 11 and later from here on
_CAPTION_COLOUR = 35
_TEXT_COLOUR = 36

# SetWindowPos: only make the window redraw its frame.
_REDRAW_FRAME = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020


def claim_taskbar_identity():
    """Give the process a taskbar identity of its own. Before the first window.

    Without one, Windows shows the icon of python.exe for the window, whatever
    icon the window itself has.
    """
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (AttributeError, OSError):
        pass


def set_icon(root):
    """Sotto's mark as the icon of `root` and of every window opened after it.

    iconphoto() copies the pictures, so they need not be kept.
    """
    root.iconphoto(True, *(tk.PhotoImage(master=root, data=icons.png("app_logo", size))
                           for size in ICON_SIZES))


def dark_title_bar(window):
    """A title bar in the window's own colours instead of the light default.

    Windows 10 knows a dark title bar; Windows 11 also takes exact colours, so
    the bar becomes part of the window. The frame only exists once the window
    is on screen. A window that is not there yet comes up fully transparent
    and is made visible once its frame is painted - otherwise the light title
    bar flashes up first.
    """
    if window.winfo_ismapped():
        _paint_frame(window)
        return

    _set_alpha(window, 0.0)

    def on_map(event):
        if event.widget is window:
            window.unbind("<Map>", binding)
            _paint_frame(window)
            _set_alpha(window, 1.0)

    binding = window.bind("<Map>", on_map, add="+")


def _set_alpha(window, alpha):
    try:
        window.attributes("-alpha", alpha)
    except tk.TclError:
        pass


def _paint_frame(window):
    try:
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        dwm = ctypes.windll.dwmapi
    except (AttributeError, OSError, tk.TclError):
        return
    if not hwnd:
        return

    def set_attribute(attribute, value):
        data = ctypes.c_int(value)
        return dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(data),
                                         ctypes.sizeof(data)) == 0

    if not set_attribute(_DARK_MODE, 1):
        set_attribute(_DARK_MODE_BEFORE_20H1, 1)
    # Ignored where the system does not know them (Windows 10).
    set_attribute(_CAPTION_COLOUR, colorref(T.BG))
    set_attribute(_TEXT_COLOUR, colorref(T.TEXT))
    set_attribute(_BORDER_COLOUR, colorref(T.BORDER))
    ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, _REDRAW_FRAME)


def colorref(colour):
    """'#rrggbb' as the 0x00bbggrr value Windows expects."""
    red, green, blue = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return red | (green << 8) | (blue << 16)
