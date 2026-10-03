"""Entry point of the Audio AI Recorder.

Start with:
    pythonw main.py        (no console window)
    python  main.py        (with a console, for troubleshooting)
"""

import sys
import traceback


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            # Enable High-DPI awareness so Windows does not blur or shrink Tkinter on WQHD / 4K monitors
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    try:
        import tkinter as tk
        from audio_transcriber import paths
        from audio_transcriber.ui.app import RecorderApp
    except ImportError as exc:

        _fatal(f"A required library is missing: {exc}\n\n"
               f"Install with:\n"
               f"    pip install -r requirements.txt")
        return 1

    try:
        # Before anything reads settings.json or creates bin/ and output/: a
        # packaged build older than this one kept them inside _internal.
        paths.migrate_legacy_data()
        root = tk.Tk()
        RecorderApp(root)
        root.mainloop()
    except Exception:
        _fatal("The application could not be started:\n\n"
               + traceback.format_exc())
        return 1
    return 0


def _fatal(message):
    """Show an error even when the app runs without a console."""
    print(message, file=sys.stderr)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Audio AI Recorder", message)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
