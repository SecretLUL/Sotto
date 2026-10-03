"""Sotto: private meeting transcripts, made on your machine.

Package layout:
    paths       - central path resolution (script directory, never cwd)
    config      - settings, including the DPAPI-encrypted API key
    events      - worker -> GUI bridge (one queue, one pump)
    audio/      - devices, capture, DSP
    transcribe/ - whisper.cpp and ElevenLabs backends
    diarize     - merges both tracks into a single transcript
    ui/         - Tkinter interface
"""

__version__ = "2.0.0"
