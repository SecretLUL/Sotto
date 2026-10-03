"""The window lays out without cutting anything off.

The Processing Options card pushed its last switches (and the 'Save settings'
button) out of the window, because a long hint label stretched the grid columns
above it; hints in other cards were cut off at the card's edge. Nothing in the
suite looked at the real layout, so nothing noticed.

The window is shown for these tests - fully transparent, so that nothing flashes
on the screen of whoever runs them - because a window that is not mapped is
never laid out.
"""

import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

try:
    _probe = tk.Tk()
    _probe.destroy()
    HAVE_DISPLAY = True
except Exception:                                   # pragma: no cover
    HAVE_DISPLAY = False


def pump(root, rounds=8):
    for _ in range(rounds):
        root.update_idletasks()
        root.update()


def box(widget):
    left, top = widget.winfo_rootx(), widget.winfo_rooty()
    return left, top, left + widget.winfo_width(), top + widget.winfo_height()


def describe(widget):
    text = ""
    try:
        text = str(widget.cget("text"))[:30]
    except tk.TclError:
        pass
    return f"{type(widget).__name__}({widget.winfo_name()}){' ' + repr(text) if text else ''}"


def problems(widget, found=None):
    """Descendants that stick out of their parent or are squeezed below the size
    they asked for - the two ways something gets cut off."""
    from audio_transcriber.ui import widgets as W
    found = [] if found is None else found
    for child in widget.winfo_children():
        if isinstance(child, tk.Toplevel) or not child.winfo_viewable():
            continue
        pl, pt, pr, pb = box(widget)
        cl, ct, cr, cb = box(child)
        if cr > pr + 1 or cb > pb + 1 or cl < pl - 1 or ct < pt - 1:
            found.append(f"{describe(child)} sticks out of {describe(widget)}: "
                         f"{(cl, ct, cr, cb)} not within {(pl, pt, pr, pb)}")
        if isinstance(child, (tk.Label, W.Button, W.Switch)) \
                and child.winfo_width() + 1 < child.winfo_reqwidth():
            found.append(f"{describe(child)} is {child.winfo_width()} px wide but "
                         f"needs {child.winfo_reqwidth()}")
        problems(child, found)
    return found


class ShownApp:
    """The real window, laid out for real, with audio and settings stubbed."""

    def __init__(self, width=920, height=980):
        self.size = (width, height)

    def __enter__(self):
        from audio_transcriber import config, preflight
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons
        from audio_transcriber.ui.app import RecorderApp

        icons._ICON_CACHE.clear()
        self.out_dir = tempfile.mkdtemp()
        settings = config.Settings(output_dir=self.out_dir)
        self.patches = [
            patch.object(config, "load", return_value=(settings, [])),
            patch.object(AudioEngine, "configure", return_value=[]),
            patch.object(preflight, "check", return_value=[]),
        ]
        for patcher in self.patches:
            patcher.start()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.root.geometry("%dx%d+0+0" % self.size)
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:                          # pragma: no cover
            pass
        self.root.deiconify()
        pump(self.root)
        return self

    def __exit__(self, *_exc):
        import shutil
        from audio_transcriber.ui import icons
        from unittest.mock import patch as _patch
        with _patch("audio_transcriber.ui.app.messagebox"):
            self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        icons._ICON_CACHE.clear()
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def show(self, tab):
        self.app.notebook.select(getattr(self.app, tab))
        pump(self.root)
        return getattr(self.app, tab)


TABS = ("tab_recorder", "tab_settings", "tab_transcript")


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestNothingIsCutOff(unittest.TestCase):
    def test_no_tab_has_anything_sticking_out_or_squeezed(self):
        with ShownApp() as shown:
            for name in TABS:
                with self.subTest(tab=name):
                    tab = shown.show(name)
                    self.assertEqual(problems(tab), [])

    def test_all_the_switches_of_the_options_card_are_inside_the_window(self):
        """The regression itself: 'Separate tracks', 'VAD' and 'Save settings'
        were beyond the right edge."""
        with ShownApp() as shown:
            shown.show("tab_settings")
            window = box(shown.root)
            for switch in _walk(shown.root, "Switch"):
                with self.subTest(switch=describe(switch)):
                    left, top, right, bottom = box(switch)
                    self.assertLessEqual(right, window[2])
                    self.assertGreaterEqual(left, window[0])
            save = shown.app.save_settings_btn
            self.assertLessEqual(box(save)[2], window[2], "'Save settings' is out of view")
            self.assertTrue(save.winfo_viewable())

    def test_every_switch_of_the_card_is_on_screen_somewhere(self):
        with ShownApp() as shown:
            shown.show("tab_settings")
            captions = {switch._text for switch in _walk(shown.root, "Switch")}
            self.assertEqual(captions, {"Live transcription", "Live preview",
                                        "Separate tracks", "VAD", "Keep raw tracks",
                                        "GPU"})


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestFlow(unittest.TestCase):
    """Children side by side, wrapping where the width runs out."""

    def setUp(self):
        from audio_transcriber.ui import theme
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)
        self.addCleanup(self.root.destroy)

    def _flow(self, widths, height=20):
        from audio_transcriber.ui import widgets as W
        flow = W.Flow(self.root, gap=10, line_gap=6)
        items = [tk.Frame(flow, width=width, height=height) for width in widths]
        for item in items:
            flow.add(item)
        return flow, items

    @staticmethod
    def _at(item):
        info = item.place_info()
        return int(info["x"]), int(info["y"])

    def test_everything_stays_on_one_line_while_it_fits(self):
        flow, items = self._flow([100, 100, 100])
        flow.fit(400)
        self.assertEqual([self._at(item) for item in items],
                         [(0, 0), (110, 0), (220, 0)])
        self.assertEqual(int(flow.cget("height")), 20)

    def test_a_child_that_does_not_fit_moves_to_the_next_line(self):
        flow, items = self._flow([100, 100, 100])
        flow.fit(250)                   # two fit: 100 + 10 + 100 = 210; a third needs 320
        self.assertEqual([self._at(item) for item in items],
                         [(0, 0), (110, 0), (0, 26)])
        self.assertEqual(int(flow.cget("height")), 46)

    def test_the_order_is_kept_across_many_lines(self):
        flow, items = self._flow([90] * 7)
        flow.fit(200)                   # two per line
        positions = [self._at(item) for item in items]
        self.assertEqual(positions, [(0, 0), (100, 0), (0, 26), (100, 26),
                                     (0, 52), (100, 52), (0, 78)])

    def test_a_child_wider_than_the_flow_gets_a_line_to_itself(self):
        flow, items = self._flow([50, 300, 50])
        flow.fit(120)
        self.assertEqual([self._at(item) for item in items],
                         [(0, 0), (0, 26), (0, 52)])

    def test_widening_it_brings_the_children_back_up(self):
        flow, items = self._flow([100, 100, 100])
        flow.fit(250)
        flow.fit(400)
        self.assertEqual(self._at(items[2]), (220, 0))
        self.assertEqual(int(flow.cget("height")), 20)

    def test_before_it_has_a_width_everything_is_on_one_line(self):
        flow, items = self._flow([100, 100, 100])
        self.assertEqual([self._at(item)[1] for item in items], [0, 0, 0])

    def test_children_of_different_heights_share_a_line_by_the_tallest(self):
        from audio_transcriber.ui import widgets as W
        flow = W.Flow(self.root, gap=10, line_gap=6)
        small = tk.Frame(flow, width=50, height=10)
        tall = tk.Frame(flow, width=50, height=30)
        after = tk.Frame(flow, width=50, height=10)
        for item in (small, tall, after):
            flow.add(item)
        flow.fit(120)                   # small + tall on the first line, after below
        self.assertEqual(self._at(after), (0, 36))
        self.assertEqual(int(flow.cget("height")), 46)

    def test_it_asks_for_no_width_of_its_own(self):
        flow, _items = self._flow([400, 400])
        self.assertEqual(flow.winfo_reqwidth(), 1, "the room is given, not demanded")


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestWrapLabel(unittest.TestCase):
    LONG = ("A hint that is much too long for one line: it goes on and on about "
            "what this setting does, and what happens if it is switched on. " * 2)

    def setUp(self):
        from audio_transcriber.ui import theme
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)
        self.addCleanup(self.root.destroy)

    def _shown(self, width):
        from audio_transcriber.ui import widgets as W
        self.root.geometry(f"{width}x300+0+0")
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:                          # pragma: no cover
            pass
        self.root.deiconify()
        label = W.WrapLabel(self.root, text=self.LONG)
        label.pack(fill=tk.X)
        pump(self.root)
        return label

    def test_it_asks_for_next_to_no_width_however_long_the_text(self):
        from audio_transcriber.ui import widgets as W
        label = W.WrapLabel(self.root, text=self.LONG)
        self.assertLess(label.winfo_reqwidth(), 60)

    def test_it_wraps_to_the_width_it_is_given(self):
        label = self._shown(400)
        self.assertAlmostEqual(int(label.cget("wraplength")), label.winfo_width() - 4,
                               delta=2)
        self.assertGreater(label.winfo_height(), 2 * 12, "several lines tall")
        self.assertLessEqual(label.winfo_width(), 400)

    def test_a_wider_room_needs_fewer_lines(self):
        label = self._shown(400)
        narrow = label.winfo_height()
        self.root.geometry("1200x300+0+0")
        pump(self.root)
        self.assertLess(label.winfo_height(), narrow)

    def test_the_text_is_not_cut_off(self):
        label = self._shown(400)
        self.assertGreaterEqual(label.winfo_height() + 1, label.winfo_reqheight())


def _walk(widget, class_name):
    for child in widget.winfo_children():
        if type(child).__name__ == class_name:
            yield child
        yield from _walk(child, class_name)


if __name__ == "__main__":
    unittest.main()
