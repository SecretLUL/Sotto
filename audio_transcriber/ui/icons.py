"""Vector icons for the Tkinter interface, drawn with Pillow.

One line style for all of them - round-capped strokes on a 24 x 24 grid, the
way line icons are usually drawn - in a single colour that the caller chooses
(fg), so a tab, a button or a source row can tint them to their state. The
only exception is Sotto's mark, which has its own colours.

Rendered at four times the size and scaled down with Lanczos for clean edges,
then converted to tk.PhotoImage.
"""

import io
import math
import tkinter as tk
from PIL import Image, ImageDraw

_ICON_CACHE = {}

DEFAULT_COLOUR = "#94a3b8"        # theme.TEXT_DIM


def get_icon(name: str, size: int = 24, fg: str = None) -> tk.PhotoImage:
    """Get a tk.PhotoImage for the given icon name and size (cached)."""
    cache_key = (name, size, fg)
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    photo = tk.PhotoImage(data=png(name, size, fg=fg))
    _ICON_CACHE[cache_key] = photo
    return photo


def png(name: str, size: int = 24, fg: str = None) -> bytes:
    """The icon as PNG data, for a PhotoImage of a particular Tk (not cached)."""
    buf = io.BytesIO()
    render_icon_image(name, size, fg=fg).save(buf, format="PNG")
    return buf.getvalue()


def render_icon_image(name: str, size: int = 24, fg: str = None) -> Image.Image:
    """Render a vector icon into a high-DPI RGBA PIL Image."""
    scale = 4
    canvas_size = size * scale
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    renderer = _RENDERERS.get(name, _draw_fallback)
    renderer(draw, canvas_size, fg=fg)

    # Downsample with Lanczos anti-aliasing for smooth crisp vector edges
    return img.resize((size, size), Image.Resampling.LANCZOS)


# ======================================================================
# Drawing on the 24 x 24 grid
# ======================================================================
class _Pen:
    """Round-capped strokes and plain fills in grid units (0 .. 24)."""

    def __init__(self, draw, S, colour=None, weight=2.0):
        self.draw = draw
        self.k = S / 24.0
        self.colour = colour or DEFAULT_COLOUR
        self.w = weight * self.k

    def _xy(self, x, y):
        return x * self.k, y * self.k

    def _width(self):
        return max(1, round(self.w))

    def _dot(self, x, y, radius):
        self.draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                          fill=self.colour)

    def line(self, *coords, closed=False):
        """A polyline through (x, y) pairs, round at its ends and joints."""
        points = [self._xy(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        if closed:
            points.append(points[0])
        self.draw.line(points, fill=self.colour, width=self._width(), joint="curve")
        for x, y in (points[0], points[-1]):
            self._dot(x, y, self.w / 2)

    def arc(self, cx, cy, r, start, end):
        """Degrees from three o'clock, clockwise - as in ImageDraw.arc()."""
        x, y = self._xy(cx, cy)
        radius, half = r * self.k, self.w / 2
        self.draw.arc([x - radius - half, y - radius - half,
                       x + radius + half, y + radius + half],
                      start, end, fill=self.colour, width=self._width())
        for angle in (start, end):
            a = math.radians(angle)
            self._dot(x + radius * math.cos(a), y + radius * math.sin(a), half)

    def rect(self, x0, y0, x1, y1, radius):
        """The outline of a rounded rectangle, the stroke centred on its edge."""
        half = self.w / 2
        (a, b), (c, d) = self._xy(x0, y0), self._xy(x1, y1)
        self.draw.rounded_rectangle([a - half, b - half, c + half, d + half],
                                    radius=radius * self.k + half,
                                    outline=self.colour, width=self._width())

    def ellipse(self, cx, cy, rx, ry):
        half = self.w / 2
        x, y = self._xy(cx, cy)
        self.draw.ellipse([x - rx * self.k - half, y - ry * self.k - half,
                           x + rx * self.k + half, y + ry * self.k + half],
                          outline=self.colour, width=self._width())

    def circle(self, cx, cy, r):
        self.ellipse(cx, cy, r, r)

    def disc(self, cx, cy, r):
        x, y = self._xy(cx, cy)
        self._dot(x, y, r * self.k)

    def block(self, x0, y0, x1, y1, radius):
        """A filled rounded rectangle."""
        (a, b), (c, d) = self._xy(x0, y0), self._xy(x1, y1)
        self.draw.rounded_rectangle([a, b, c, d], radius=radius * self.k,
                                    fill=self.colour)

    def erase(self, x0, y0, x1, y1, radius=0):
        """Clear an area again, to let one shape pass behind another."""
        (a, b), (c, d) = self._xy(x0, y0), self._xy(x1, y1)
        self.draw.rounded_rectangle([a, b, c, d], radius=radius * self.k,
                                    fill=(0, 0, 0, 0))

    def star(self, cx, cy, r_out, r_in):
        """A filled four-pointed star."""
        points = []
        for i in range(8):
            a = i * math.pi / 4 - math.pi / 2
            r = r_out if i % 2 == 0 else r_in
            points.append(self._xy(cx + r * math.cos(a), cy + r * math.sin(a)))
        self.draw.polygon(points, fill=self.colour)


# ======================================================================
# The icons
# ======================================================================
def _draw_microphone(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.rect(9, 2.5, 15, 14.5, 3)
    pen.arc(12, 11.5, 7, 0, 180)
    pen.line(12, 18.5, 12, 21.5)


def _draw_speaker(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.line(3.5, 9.5, 7.5, 9.5, 12.5, 5, 12.5, 19, 7.5, 14.5, 3.5, 14.5, closed=True)
    pen.arc(13, 12, 4, -45, 45)
    pen.arc(13, 12, 8, -48, 48)


def _draw_sparkle(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.star(10.5, 13, 8, 2.4)
    pen.star(18.5, 5.5, 3.6, 1.1)


def _draw_record(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    _Pen(draw, S, fg or "#f43f5e").disc(12, 12, 6)


def _draw_stop(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    _Pen(draw, S, fg or "#f43f5e").block(6.5, 6.5, 17.5, 17.5, 2.5)


def _draw_globe(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.circle(12, 12, 9)
    pen.ellipse(12, 12, 4, 9)
    pen.line(3, 12, 21, 12)


def _draw_brain(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    """The model: a chip."""
    pen = _Pen(draw, S, fg)
    pen.rect(6, 6, 18, 18, 2.5)
    pen.block(9.5, 9.5, 14.5, 14.5, 1)
    for at in (10, 14):
        pen.line(at, 2.5, at, 6)
        pen.line(at, 18, at, 21.5)
        pen.line(2.5, at, 6, at)
        pen.line(18, at, 21.5, at)


def _draw_lock(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.rect(5, 11, 19, 21, 2.5)
    pen.line(8, 11, 8, 7.5)
    pen.arc(12, 7.5, 4, 180, 360)
    pen.line(16, 7.5, 16, 11)


def _eye(pen):
    # The lid: two arcs through (2, 12) and (22, 12), 7 units apart at the middle.
    pen.arc(12, 15.64, 10.64, 200, 340)
    pen.arc(12, 8.36, 10.64, 20, 160)
    pen.circle(12, 12, 3)


def _draw_eye(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    _eye(_Pen(draw, S, fg))


def _draw_eye_off(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    _eye(pen)
    pen.line(4, 4, 20, 20)


def _draw_app_logo(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    """Sotto's mark, as in assets/logo.svg (drawn on a 512 grid there).

    An S from two elliptical arcs that meet in the middle: the upper one in the
    colour of your own voice, the lower one in that of everyone else.
    """
    u = S / 512
    draw.rounded_rectangle([24 * u, 24 * u, 488 * u, 488 * u], radius=116 * u,
                           fill="#171b25", outline="#2e3445",
                           width=max(1, round(6 * u)))

    rx, ry, half = 84 * u, 74 * u, 23 * u

    def band(cy, start, end, steps=64):
        """The stroke along an arc as one polygon: both edges offset along the
        ellipse's normal. draw.line() leaves notches between its segments."""
        outer, inner = [], []
        for i in range(steps + 1):
            a = math.radians(start + (end - start) * i / steps)
            x, y = 256 * u + rx * math.cos(a), cy + ry * math.sin(a)
            nx, ny = ry * math.cos(a), rx * math.sin(a)
            scale = half / math.hypot(nx, ny)
            outer.append((x + nx * scale, y + ny * scale))
            inner.append((x - nx * scale, y - ny * scale))
        return outer + inner[::-1], (x, y)

    # Upper arc from one degree past the joint (hidden under the lower one) to
    # its end at the top right; lower arc from the joint to the bottom left.
    # The angles are those of logo.svg.
    for cy, start, end, colour in ((182 * u, 89, 332, "#38bdf8"),
                                   (330 * u, -90, 152, "#34d399")):
        outline, (x, y) = band(cy, start, end)
        draw.polygon(outline, fill=colour)
        draw.ellipse([x - half, y - half, x + half, y + half], fill=colour)


def _draw_transcript(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.rect(5, 2.5, 19, 21.5, 2.5)
    pen.line(8.5, 8, 15.5, 8)
    pen.line(8.5, 12, 15.5, 12)
    pen.line(8.5, 16, 12.5, 16)


def _draw_copy(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.rect(3.5, 3.5, 14.5, 14.5, 2.5)
    pen.erase(7.5, 7.5, 22, 22, 3.5)
    pen.rect(9.5, 9.5, 20.5, 20.5, 2.5)


def _draw_trash(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.line(3.5, 6.5, 20.5, 6.5)
    pen.line(9, 6.5, 9, 4, 15, 4, 15, 6.5)
    pen.line(5.5, 6.5, 6.5, 20.5, 17.5, 20.5, 18.5, 6.5)
    pen.line(10, 10.5, 10, 16.5)
    pen.line(14, 10.5, 14, 16.5)


def _draw_warning(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg or "#f59e0b")
    pen.line(12, 3.5, 21.5, 20, 2.5, 20, closed=True)
    pen.line(12, 9.5, 12, 13.5)
    pen.disc(12, 16.8, 1.3)


def _draw_check(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    _Pen(draw, S, fg or "#10b981").line(4.5, 12.5, 9.5, 17.5, 19.5, 6.5)


def _tray(pen):
    pen.line(4, 15, 4, 20, 20, 20, 20, 15)


def _draw_upload(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.line(12, 15, 12, 3.5)
    pen.line(7, 8.5, 12, 3.5, 17, 8.5)
    _tray(pen)


def _draw_save(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.line(12, 3.5, 12, 15)
    pen.line(7, 10, 12, 15, 17, 10)
    _tray(pen)


def _draw_sources(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    """Two faders: the two sources."""
    pen = _Pen(draw, S, fg)
    pen.line(3.5, 8, 20.5, 8)
    pen.line(3.5, 16, 20.5, 16)
    pen.disc(9, 8, 2.8)
    pen.disc(15, 16, 2.8)


def _draw_folder(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    pen = _Pen(draw, S, fg)
    pen.line(3, 6, 9.5, 6, 11.5, 8.5, 21, 8.5, 21, 19.5, 3, 19.5, closed=True)


def _draw_settings(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    """A gear: eight teeth around a ring."""
    pen = _Pen(draw, S, fg)
    teeth = _Pen(draw, S, fg, weight=3.4)
    for i in range(8):
        a = i * math.pi / 4
        teeth.line(12 + 7.2 * math.cos(a), 12 + 7.2 * math.sin(a),
                   12 + 9.2 * math.cos(a), 12 + 9.2 * math.sin(a))
    pen.circle(12, 12, 6.4)
    pen.circle(12, 12, 2.4)


def _draw_chevron_down(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    _Pen(draw, S, fg).line(6.5, 9.5, 12, 15, 17.5, 9.5)


def _draw_refresh(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    """Two arrows chasing each other round - a new version of the app."""
    pen = _Pen(draw, S, fg)
    for start, end in ((200, 335), (20, 155)):
        pen.arc(12, 12, 7.5, start, end)
        angle = math.radians(end)
        tip_x, tip_y = 12 + 7.5 * math.cos(angle), 12 + 7.5 * math.sin(angle)
        # Back along the way the arc came, turned out to either side.
        back = angle - math.pi / 2
        wings = [(tip_x + 4 * math.cos(back + turn), tip_y + 4 * math.sin(back + turn))
                 for turn in (-0.55, 0.55)]
        pen.line(*wings[0], tip_x, tip_y, *wings[1])


def _draw_fallback(draw: ImageDraw.ImageDraw, S: float, fg: str = None):
    _Pen(draw, S, fg).disc(12, 12, 6)


_RENDERERS = {
    "microphone": _draw_microphone,
    "speaker": _draw_speaker,
    "sparkle": _draw_sparkle,
    "record": _draw_record,
    "stop": _draw_stop,
    "globe": _draw_globe,
    "brain": _draw_brain,
    "lock": _draw_lock,
    "eye": _draw_eye,
    "eye_off": _draw_eye_off,
    "app_logo": _draw_app_logo,
    "transcript": _draw_transcript,
    "copy": _draw_copy,
    "trash": _draw_trash,
    "warning": _draw_warning,
    "check": _draw_check,
    "upload": _draw_upload,
    "save": _draw_save,
    "sources": _draw_sources,
    "folder": _draw_folder,
    "settings": _draw_settings,
    "chevron_down": _draw_chevron_down,
    "refresh": _draw_refresh,
}
