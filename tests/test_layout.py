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


def pump(root, rounds=20):
    for _ in range(rounds):
        root.update_idletasks()
        root.update()


def box(widget):
    left, top = widget.winfo_rootx(), widget.winfo_rooty()
    return left, top, left + widget.winfo_width(), top + widget.winfo_height()


def wheel_down(widget):
    """One notch of the mouse wheel over `widget`, as the platform sends it."""
    if widget.tk.call("tk", "windowingsystem") == "x11":
        widget.event_generate("<Button-5>")
    else:
        widget.event_generate("<MouseWheel>", delta=-120)


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
        if isinstance(widget.master, W.Scroller) and child is widget.master.body:
            pt, pb = ct, cb             # a page taller than its window scrolls
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
    """The real window, laid out for real, with audio and settings stubbed.

    dpi simulates a display scaling and screen a screen of that size; with a
    screen the window keeps the size it picks for it, without one it is made
    width x height.
    """

    def __init__(self, width=920, height=980, dpi=None, screen=None):
        self.size = (width, height)
        self.dpi = dpi
        self.screen = screen

    def __enter__(self):
        from audio_transcriber import config, preflight
        from audio_transcriber.audio.capture import AudioEngine
        from audio_transcriber.ui import icons, theme
        from audio_transcriber.ui.app import RecorderApp

        self._scale_before = theme.SCALE
        icons._ICON_CACHE.clear()
        self.out_dir = tempfile.mkdtemp()
        settings = config.Settings(output_dir=self.out_dir)
        self.patches = [
            patch.object(config, "load", return_value=(settings, [])),
            patch.object(AudioEngine, "configure", return_value=[]),
            patch.object(preflight, "check", return_value=[]),
        ]
        if self.dpi:
            self.patches.append(patch.object(
                theme, "_display_dpi", lambda root, _dpi=self.dpi: float(_dpi)))
        if self.screen:
            width, height = self.screen
            self.patches += [
                patch.object(tk.Misc, "winfo_screenwidth", lambda _w: width),
                patch.object(tk.Misc, "winfo_screenheight", lambda _w: height)]
        for patcher in self.patches:
            patcher.start()
        self.root = tk.Tk()
        # 'tk scaling' belongs to the display, not to this window: the next
        # Tk() in the process would take on a simulated dpi. Put back in __exit__.
        self._tk_scaling = self.root.tk.call("tk", "scaling")
        self.root.withdraw()
        self.app = RecorderApp(self.root)
        self.root.geometry("+0+0" if self.screen else "%dx%d+0+0" % self.size)
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:                          # pragma: no cover
            pass
        self.root.deiconify()
        pump(self.root)
        return self

    def __exit__(self, *_exc):
        import shutil
        from audio_transcriber.ui import icons, theme
        from unittest.mock import patch as _patch
        self.root.tk.call("tk", "scaling", self._tk_scaling)
        with _patch("audio_transcriber.ui.app.messagebox"):
            self.app.on_close()
        for patcher in self.patches:
            patcher.stop()
        theme._set_scale(self._scale_before * 96)     # no scaling left behind
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
                                        "GPU", "Check for updates at start"})


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


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestCardHint(unittest.TestCase):
    """The line at the right of a card's heading.

    set_hint() did nothing on a card that was made without a hint - the Sources
    card among them, whose hint says which output the system audio is taken
    from and, since the stricter matching, why it could not find one. It was
    never visible.
    """

    def setUp(self):
        from audio_transcriber.ui import theme
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)
        self.addCleanup(self.root.destroy)

    def _card(self, width=None, **kw):
        from audio_transcriber.ui import widgets as W
        card = W.Card(self.root, title="Sources", **kw)
        card.pack(fill=tk.X)
        if width:
            self.root.geometry(f"{width}x200+0+0")
            try:
                self.root.attributes("-alpha", 0.0)
            except tk.TclError:                      # pragma: no cover
                pass
            self.root.deiconify()
            pump(self.root)
        return card

    @staticmethod
    def _text(card):
        return card.itemcget(card._hint_item, "text")

    def test_a_card_made_without_a_hint_can_still_be_given_one(self):
        card = self._card()
        card.set_hint("System audio captured from: Speakers")
        self.assertEqual(self._text(card), "System audio captured from: Speakers")

    def test_a_hint_given_at_the_start_is_shown(self):
        self.assertEqual(self._text(self._card(hint="start hint")), "start hint")

    def test_the_hint_can_be_cleared(self):
        card = self._card(hint="x")
        card.set_hint("")
        self.assertEqual(self._text(card), "")

    def test_a_long_hint_is_cut_short_rather_than_running_over_the_heading(self):
        card = self._card(width=500)
        card.set_hint("System audio captured from: " + "a very long device name " * 8)
        pump(self.root)
        shown = self._text(card)
        self.assertTrue(shown.endswith("…"), shown)
        left, _top, right, _bottom = card.bbox(card._hint_item)
        title_right = card.bbox(card._title_item)[2]
        self.assertGreater(left, title_right, "it must not reach into the heading")
        self.assertLessEqual(right, card.winfo_width() - card.padx + 1)

    def test_a_hint_that_fits_is_shown_whole(self):
        card = self._card(width=700)
        card.set_hint("short")
        pump(self.root)
        self.assertEqual(self._text(card), "short")

    def test_warnings_are_amber_and_the_rest_is_muted(self):
        from audio_transcriber.ui import theme as T
        card = self._card()
        card.set_hint("⚠ no loopback device")
        self.assertEqual(card.itemcget(card._hint_item, "fill"), T.WARN)
        card.set_hint("System audio captured from: Speakers")
        self.assertEqual(card.itemcget(card._hint_item, "fill"), T.TEXT_MUTE)


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestSourcesCardHint(unittest.TestCase):
    def test_the_sources_card_of_the_window_says_where_the_system_audio_comes_from(self):
        with ShownApp() as shown:
            card = shown.app.sources_card
            text = card.itemcget(card._hint_item, "text")
            self.assertTrue(text.startswith(("System audio captured from:", "⚠")), text)


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestDisplayScaling(unittest.TestCase):
    """Pixel sizes follow the display: at 150 % the caption of 'Start recording'
    (186 px) was wider than its 170 px button, and 'Upload file' ran under it."""

    # Display scalings, each on a screen it is commonly found on. Where the
    # screen cannot give the window the full height, the tabs scroll.
    SETUPS = ((96, (1366, 768)), (120, (1920, 1080)), (144, (1920, 1080)),
              (192, (2560, 1440)))
    LARGE_SCREEN = (3840, 2160)

    def test_nothing_is_cut_off_at_any_display_scaling(self):
        for dpi, screen in self.SETUPS:
            with self.subTest(dpi=dpi, screen=screen), \
                    ShownApp(dpi=dpi, screen=screen) as shown:
                for name in TABS:
                    tab = shown.show(name)
                    self.assertEqual(problems(tab), [], f"{name} at {dpi} dpi")

    def test_the_window_fits_on_the_screen(self):
        for dpi, screen in self.SETUPS:
            with self.subTest(dpi=dpi, screen=screen), \
                    ShownApp(dpi=dpi, screen=screen) as shown:
                self.assertLessEqual(shown.root.winfo_width(), screen[0])
                self.assertLessEqual(shown.root.winfo_height(), screen[1])

    def test_every_button_is_wide_enough_for_its_caption(self):
        from audio_transcriber.ui import theme as T
        for dpi, screen in self.SETUPS:
            with self.subTest(dpi=dpi), ShownApp(dpi=dpi, screen=screen) as shown:
                for name in TABS:
                    shown.show(name)
                buttons = list(_walk(shown.root, "Button"))
                self.assertTrue(buttons)
                for button in buttons:
                    needed = button.content_width() + 2 * T.px(button.PAD_X)
                    self.assertGreaterEqual(button.winfo_reqwidth(), needed,
                                            f"{describe(button)} at {dpi} dpi")

    def test_sizes_grow_with_the_display(self):
        heights = {}
        for dpi in (96, 192):
            with ShownApp(dpi=dpi, screen=self.LARGE_SCREEN) as shown:
                shown.show("tab_recorder")
                heights[dpi] = shown.app.start_btn.winfo_height()
        self.assertAlmostEqual(heights[192] / heights[96], 2.0, delta=0.1)

    def test_the_status_pill_grows_to_fit_a_long_status(self):
        from audio_transcriber.ui import theme as T
        text = "processing recovered recording…"
        for dpi in (96, 192):
            with self.subTest(dpi=dpi), \
                    ShownApp(dpi=dpi, screen=self.LARGE_SCREEN) as shown:
                shown.app.status.set(text)
                pump(shown.root)
                needed = T.fonts["small"].measure(text) + T.px(32)
                self.assertGreaterEqual(shown.app.status.winfo_reqwidth(), needed)

    def test_a_simulated_scaling_does_not_outlive_its_window(self):
        """'tk scaling' belongs to the display: a 200 % run made every later
        window of the test process think it was on a 200 % screen."""
        def dpi_of_a_new_window():
            root = tk.Tk()
            try:
                return root.winfo_fpixels("1i")
            finally:
                root.destroy()

        before = dpi_of_a_new_window()
        with ShownApp(dpi=192, screen=self.LARGE_SCREEN):
            pass
        self.assertAlmostEqual(dpi_of_a_new_window(), before, delta=0.5)


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestTooLittleRoom(unittest.TestCase):
    """Where the screen cannot give the window the height of a tab, the tab
    scrolls. At 200 % on a 1440p screen the Settings tab needs 1082 px and gets
    922: pack gave the Processing Options card 142 of the 302 px it asked for
    and cut it off - at 150 % on a 1080p screen, too."""

    def test_the_settings_scroll_instead_of_being_cut_off(self):
        with ShownApp(dpi=192, screen=(2560, 1440)) as shown:
            tab = shown.show("tab_settings")
            self.assertEqual(problems(tab), [])
            page = next(_walk(tab, "Scroller"))
            self.assertTrue(page.scrolling)

            page.canvas.yview_moveto(1.0)
            pump(shown.root)
            _left, top, _right, bottom = box(shown.app.save_settings_btn)
            view = box(page.canvas)
            self.assertGreaterEqual(top, view[1], "'Save settings' can be reached")
            self.assertLessEqual(bottom, view[3], "'Save settings' can be reached")

    def test_a_window_with_room_enough_does_not_scroll(self):
        with ShownApp() as shown:
            # Windows keeps a window within the screen: on the 1024x768 screen
            # of a CI runner the window gets ~780 of its 980 px, and there the
            # Settings tab rightly scrolls.
            if shown.root.winfo_height() < shown.size[1]:   # pragma: no cover
                self.skipTest("the screen cannot give the window its height")
            for name in ("tab_recorder", "tab_settings"):
                page = next(_walk(shown.show(name), "Scroller"))
                self.assertFalse(page.scrolling, name)
                self.assertFalse(page.scrollbar.winfo_ismapped(), name)


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestScroller(unittest.TestCase):
    """A page that scrolls where it is taller than the room it is given."""

    def setUp(self):
        from audio_transcriber.ui import theme
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)
        self.addCleanup(self.root.destroy)

    def _show(self, room):
        self.root.geometry(f"300x{room}+0+0")
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:                          # pragma: no cover
            pass
        self.root.deiconify()
        pump(self.root)

    def _page(self, room, content=400):
        from audio_transcriber.ui import widgets as W
        page = W.Scroller(self.root)
        page.pack(fill=tk.BOTH, expand=True)
        block = tk.Frame(page.body, width=200, height=content)
        block.pack(fill=tk.X)
        self._show(room)
        return page, block

    def _combo(self, page):
        from tkinter import ttk
        combo = ttk.Combobox(page.body, values=["a", "b", "c"], state="readonly")
        combo.current(0)
        combo.pack(fill=tk.X)
        pump(self.root)
        return combo

    def test_a_page_that_fits_has_no_scrollbar(self):
        page, _block = self._page(room=500)
        self.assertFalse(page.scrolling)
        self.assertFalse(page.scrollbar.winfo_ismapped())

    def test_a_page_that_does_not_fit_keeps_its_height_and_scrolls(self):
        page, block = self._page(room=200)
        self.assertTrue(page.scrolling)
        self.assertTrue(page.scrollbar.winfo_ismapped())
        self.assertEqual(block.winfo_height(), 400, "it must not be squeezed")
        page.canvas.yview_moveto(1.0)
        pump(self.root)
        self.assertAlmostEqual(box(block)[3], box(page.canvas)[3], delta=1,
                               msg="scrolled to the end, the end is in view")

    def test_the_page_is_as_wide_as_its_window(self):
        page, _block = self._page(room=200)
        self.assertEqual(page.body.winfo_width(), page.canvas.winfo_width())

    def test_the_scrollbar_can_be_seen_and_grabbed(self):
        """clam makes the thumb as thick as -arrowsize, which the style set to
        0: the scrollbar (the transcript's too) was 1 px wide."""
        from audio_transcriber.ui import theme as T
        page, _block = self._page(room=200)
        self.assertGreaterEqual(page.scrollbar.winfo_width(), T.px(12))

    def test_with_room_again_the_scrollbar_goes_and_the_page_is_back_at_the_top(self):
        page, _block = self._page(room=200)
        page.canvas.yview_moveto(1.0)
        self._show(600)
        self.assertFalse(page.scrolling)
        self.assertFalse(page.scrollbar.winfo_ismapped())
        self.assertEqual(page.canvas.yview()[0], 0.0)

    def test_the_wheel_scrolls_the_page_from_anywhere_on_it(self):
        page, block = self._page(room=200)
        wheel_down(block)
        self.assertGreater(page.canvas.yview()[0], 0.0)

    def test_a_combo_box_the_pointer_passes_does_not_catch_the_wheel(self):
        page, _block = self._page(room=200)
        combo = self._combo(page)
        wheel_down(combo)
        self.assertEqual(combo.current(), 0, "its value must not change")
        self.assertGreater(page.canvas.yview()[0], 0.0)

    def test_a_combo_box_with_the_focus_keeps_the_wheel(self):
        page, _block = self._page(room=200)
        combo = self._combo(page)
        combo.focus_force()
        pump(self.root)
        if str(self.root.tk.call("focus")) != str(combo):    # pragma: no cover
            self.skipTest("the window cannot take the keyboard focus here")
        wheel_down(combo)
        self.assertEqual(combo.current(), 1)
        self.assertEqual(page.canvas.yview()[0], 0.0)

    def test_while_everything_fits_the_widgets_keep_the_wheel(self):
        page, _block = self._page(room=600)
        combo = self._combo(page)
        wheel_down(combo)
        self.assertEqual(combo.current(), 1)


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestTabs(unittest.TestCase):
    """The tab bar that replaced ttk.Notebook, whose selected tab was smaller
    than the others and whose pages had a light frame."""

    def setUp(self):
        from audio_transcriber.ui import icons, theme
        icons._ICON_CACHE.clear()
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)
        self.addCleanup(self.root.destroy)
        self.addCleanup(icons._ICON_CACHE.clear)

    def _tabs(self):
        from audio_transcriber.ui import widgets as W
        tabs = W.Tabs(self.root)
        tabs.pack(fill=tk.BOTH, expand=True)
        pages = [tk.Frame(tabs, width=200, height=100) for _ in range(3)]
        for page, caption in zip(pages, ("One", "A longer one", "Three")):
            tabs.add(page, caption, "settings")
        self.root.geometry("640x300+0+0")
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:                          # pragma: no cover
            pass
        self.root.deiconify()
        pump(self.root)
        return tabs, pages

    def test_the_first_page_is_shown_and_only_it(self):
        tabs, pages = self._tabs()
        self.assertIs(tabs.select(), pages[0])
        self.assertEqual([page.winfo_ismapped() for page in pages],
                         [True, False, False])

    def test_selecting_shows_that_page_instead(self):
        tabs, pages = self._tabs()
        tabs.select(pages[2])
        pump(self.root)
        self.assertEqual([page.winfo_ismapped() for page in pages],
                         [False, False, True])

    def test_a_click_on_a_tab_selects_its_page(self):
        tabs, pages = self._tabs()
        width = tabs._tab_width()
        tabs.bar.event_generate("<Button-1>", x=int(tabs.INSET + 1.5 * width), y=10)
        pump(self.root)
        self.assertIs(tabs.select(), pages[1])

    def test_the_tabs_keep_their_size_when_selected(self):
        tabs, pages = self._tabs()
        before = (tabs.bar.winfo_height(), tabs._tab_width(), tabs._bar_width())
        for page in pages:
            tabs.select(page)
            pump(self.root)
            self.assertEqual(
                (tabs.bar.winfo_height(), tabs._tab_width(), tabs._bar_width()),
                before)

    def test_every_caption_fits_its_tab(self):
        from audio_transcriber.ui import theme as T
        tabs, _pages = self._tabs()
        needed = max(T.fonts["button"].measure(text) for _p, text, _i in tabs._tabs)
        self.assertGreaterEqual(tabs._tab_width(), needed + T.px(tabs.ICON))


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestTabOrder(unittest.TestCase):
    # A class of its own: ShownApp makes its own Tk, and icons' cache serves
    # whichever Tk exists first.
    def test_the_window_shows_recorder_transcript_settings_in_that_order(self):
        with ShownApp() as shown:
            app = shown.app
            self.assertEqual(app.notebook.pages(),
                             [app.tab_recorder, app.tab_transcript, app.tab_settings])
            self.assertIs(app.notebook.select(), app.tab_recorder)


@unittest.skipUnless(HAVE_DISPLAY, "no graphical display available")
class TestTranscriptPane(unittest.TestCase):
    """An empty pane says what will appear in it, and its scrollbar shows only
    while there is something to scroll."""

    def setUp(self):
        from audio_transcriber.ui import theme
        from audio_transcriber.ui import widgets as W
        self.root = tk.Tk()
        self.root.withdraw()
        theme.apply(self.root)
        self.addCleanup(self.root.destroy)
        self.pane = W.Transcript(self.root, height=4)
        self.pane.pack(fill=tk.BOTH, expand=True)
        self.root.geometry("400x200+0+0")
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:                          # pragma: no cover
            pass
        self.root.deiconify()
        pump(self.root)

    def test_the_placeholder_shows_only_while_the_pane_is_empty(self):
        pane = self.pane
        self.assertEqual(pane.placeholder.winfo_manager(), "place")
        pane.append("Recording started.\n")
        self.assertEqual(pane.placeholder.winfo_manager(), "")
        pane.clear()
        self.assertEqual(pane.placeholder.winfo_manager(), "place")

    def test_the_scrollbar_shows_only_while_there_is_more_than_fits(self):
        pane = self.pane
        self.assertEqual(pane.scroll.winfo_manager(), "")
        pane.append("".join(f"[00:{i:02d}] [You]: line {i}\n" for i in range(60)))
        pump(self.root)
        self.assertEqual(pane.scroll.winfo_manager(), "pack")
        pane.clear()
        pump(self.root)
        self.assertEqual(pane.scroll.winfo_manager(), "")


class TestThemeScale(unittest.TestCase):
    def tearDown(self):
        from audio_transcriber.ui import theme
        theme._set_scale(96)

    def test_lengths_scale_and_never_vanish(self):
        from audio_transcriber.ui import theme
        theme._set_scale(144)
        self.assertEqual(theme.px(10), 15)
        self.assertGreaterEqual(theme.px(1), 1)
        self.assertEqual(theme.px(0), 0)

    def test_a_display_below_96_dpi_is_not_shrunk(self):
        from audio_transcriber.ui import theme
        theme._set_scale(72)
        self.assertEqual(theme.SCALE, 1.0)
        self.assertEqual(theme.px(10), 10)

    def test_the_spacing_constants_follow(self):
        from audio_transcriber.ui import theme
        theme._set_scale(192)
        self.assertEqual(theme.XL, 52)
        theme._set_scale(96)
        self.assertEqual(theme.XL, 26)


def _walk(widget, class_name):
    for child in widget.winfo_children():
        if type(child).__name__ == class_name:
            yield child
        yield from _walk(child, class_name)


if __name__ == "__main__":
    unittest.main()
