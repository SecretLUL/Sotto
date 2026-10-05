"""Sotto: private meeting transcripts, made on your machine.

Package layout:
    paths       - central path resolution (script directory, never cwd)
    config      - settings, including the DPAPI-encrypted API key
    events      - worker -> GUI bridge (one queue, one pump)
    version     - which release this build is, and its archive names
    update      - checking for, downloading and installing new releases
    audio/      - devices, capture, DSP
    transcribe/ - whisper.cpp and ElevenLabs backends
    diarize     - merges both tracks into a single transcript
    ui/         - Tkinter interface
"""
