# 🎙️ Audio AI Recorder & Transcriber

[![Release: v1.0.0](https://img.shields.io/github/v/release/SecretLUL/Audio-Transcriber?color=7289da&label=Release)](https://github.com/SecretLUL/Audio-Transcriber/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue.svg)](#-platform-compatibility--downloads)
[![FOSS: 100%](https://img.shields.io/badge/FOSS-100%25-brightgreen.svg)](#-license--foss-status)

An advanced, privacy-focused audio recording and AI transcription suite. It captures **microphone** and **system audio** independently and generates high-accuracy meeting transcripts with precise **speaker attribution** — locally via `whisper.cpp` or in the cloud via the ElevenLabs Scribe API.

---

## 📌 Table of Contents

- [🚀 Quick Start](#-quick-start)
  - [Option A: Pre-built Standalone Release](#option-a-pre-built-standalone-release-recommended-)
  - [Option B: Running from Source](#option-b-running-from-source-for-developers-)
- [🐧 Platform Compatibility & Downloads](#-platform-compatibility--downloads)
- [✨ Features](#-features)
- [🏗️ Architecture & Project Structure](#-architecture--project-structure)
- [🗣️ How Speaker Diarization Works](#-how-speaker-diarization-works)
- [🎛️ User Interface & Tabs](#-user-interface--tabs)
- [⚙️ Technical Highlights](#-technical-highlights)
- [🧪 Testing & Quality Assurance](#-testing--quality-assurance)
- [📦 Automated Release Build](#-automated-release-build)
- [📄 License & FOSS Status](#-license--foss-status)

---

## 🚀 Quick Start


### Option A: Pre-built Standalone Release (Recommended 💻)

No Python installation or terminal setup required!

1. Download the latest release package for your operating system from the **[Releases Page](https://github.com/SecretLUL/Audio-Transcriber/releases)**:
   - **Windows (x64)**: `AudioTranscriber-<version>-windows-x64.zip`
   - **Linux (x64)**: `AudioTranscriber-<version>-linux-x64.tar.gz`
   - **macOS (Apple Silicon)**: `AudioTranscriber-<version>-macos-arm64.zip`
2. Extract the archive to any folder.
3. Launch `AudioTranscriber.exe` (Windows), `./AudioTranscriber` (Linux), or `AudioTranscriber.app` (macOS).

> 🍎 **Intel Macs**: the released macOS build is Apple Silicon only. On an Intel
> Mac, run from source (Option B) or build locally with `python build_release.py`,
> which produces a `macos-x64` archive.

---

### Option B: Running from Source (For Developers 🛠️)

1. Clone the repository and install the Python dependencies:
   ```shell
   git clone https://github.com/SecretLUL/Audio-Transcriber.git
   cd Audio-Transcriber
   pip install -r requirements.txt
   ```
2. Launch the application:
   - **Windows (no console)**: Double-click `Start-Recorder.vbs`
   - **Terminal / Cross-platform**: Run `python main.py`

> 💡 **Note**: `whisper.cpp` binaries and GGML models are automatically fetched on demand to the `bin/` directory on first launch.

---

## 🐧 Platform Compatibility & Downloads

| Operating System | Pre-built Executable | Running from Source | Audio Capture Backend |
| :--- | :---: | :---: | :--- |
| **Windows 10 / 11 (x64)** | ✅ Supported (`.exe` in `.zip`) | ✅ Supported | Native WASAPI Loopback (`pyaudiowpatch`) |
| **Linux (Ubuntu, Debian, Arch, etc.)** | ✅ Supported (`.tar.gz`) | ✅ Supported | PulseAudio / PipeWire Monitor (`pyaudio`) |
| **macOS (Intel & Apple Silicon M1-M4)** | ✅ Supported (`.zip`) | ✅ Supported | CoreAudio + [BlackHole](https://github.com/ExistentialAudio/BlackHole) Loopback |

> 🤖 **Automated CI/CD Pipeline**: GitHub Actions automatically compiles and packages standalone releases for **Windows**, **Linux**, and **macOS** on every release tag (`v*`).

---

## ✨ Features

- 🎤 **Dual-Track Capture**: Separate recording of your voice (microphone) and other participants (system audio).
- 📁 **Audio File Upload & Transcription**: Upload and transcribe any local audio file (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.wma`, `.mp4`, `.webm`, `.opus`, etc.).
- 📂 **Custom Output Directory**: Choose any target folder on your system for saving output `.wav` audio files and `.txt` transcripts.
- 🗂️ **Tabbed Interface**: Modern tabbed layout dividing controls cleanly into **Recorder**, **Settings**, and **Transcript**.
- 🗣️ **Smart Speaker Diarization**: Exact speaker tagging (`[You]` vs `[Participant]`) via track-relative level comparison.
- 🤖 **Offline Local AI**: Integrated `whisper.cpp` engine with automatic GGML model downloads (Tiny to Large-v3).
- ☁️ **Cloud API Acceleration**: Optional ElevenLabs Scribe v2 integration with token-level timestamp alignment.
- ⚡ **Live Transcription**: Recognises the recording in silence-aligned chunks while it runs, so pressing stop only has to catch up with the last seconds. Includes an optional live preview to read along.
- 🎛️ **DAW-Grade VUMeters**: Custom hand-drawn peak/RMS meters with dynamic dB readouts and gain sliders (-20 dB to +20 dB).
- 🔒 **Encrypted Secret Storage**: ElevenLabs API keys are encrypted via Windows DPAPI or secure user-scoped storage.
- 🎨 **Modern Dark Interface**: Custom Tkinter `Canvas` design system with zero external UI framework dependencies.
- 🆓 **100% Free & Open Source**: Released under the permissive MIT License.

---

## 🏗️ Architecture & Project Structure

```text
Audio-Transcriber/
 ├── main.py                     Entry point for python / pythonw launch
 ├── Start-Recorder.vbs          Windows double-click launcher
 ├── build_release.py            Automated PyInstaller standalone build & zip packaging
 ├── requirements.txt            Core dependencies (pyaudiowpatch, soundfile, scipy, numpy, Pillow)
 ├── LICENSE                     MIT License (100% FOSS)
 ├── README.md                   Project documentation
 ├── audio_transcriber/          Main application package
 │    ├── paths.py               Central path resolution & filename sanitization
 │    ├── config.py              Typed & versioned settings (Schema v2)
 │    ├── secretstore.py         DPAPI Windows key encryption
 │    ├── events.py              Thread-safe UI event bridge & pump
 │    ├── diarize.py             Speaker merging, bleed & hallucination filter
 │    ├── pipeline.py            Live transcription, closing pass & preview engine
 │    ├── audio/                 Audio processing module
 │    │    ├── devices.py        WASAPI / PulseAudio enumeration & loopback matching
 │    │    ├── capture.py        Multi-device audio capture & disk streaming
 │    │    ├── dsp.py            Active channel downmix & 16 kHz polyphase resampling
 │    │    └── loader.py         Universal audio file loader with soundfile & FFmpeg fallback
 │    ├── transcribe/            AI Transcription engines
 │    │    ├── base.py           Shared backend interface & timestamp parsers
 │    │    ├── binaries.py       Atomic model & binary downloader
 │    │    ├── whispercpp.py     Local whisper.cpp CLI subprocess runner
 │    │    └── elevenlabs.py     Cloud ElevenLabs Scribe API runner
 │    └── ui/                    Interface module
 │         ├── theme.py          Color tokens & dark ttk clam styles
 │         ├── widgets.py        Custom Canvas Cards, Sliders, Switches & Transcript
 │         └── app.py            Main Tkinter window controller
 └── legacy/                     Archived single-file implementation
```

---

## 🗣️ How Speaker Diarization Works

Instead of downmixing audio upfront and estimating speakers probabilistically, both audio channels are captured and transcribed **independently**. Speaker attribution is then determined deterministically in `diarize.py`:

1. **Crosstalk / Bleed Filtering**: Speaker audio bleeding into the microphone is filtered using relative track energy levels (independent of gain sliders).
2. **Hallucination Prevention**: Audio segments with no signal energy on their source track are automatically dropped.
3. **Duplicate Resolution**: Identical sentences detected on both tracks are merged, giving priority to the clearer audio source.

---

## 🎛️ User Interface & Tabs

- 🎙️ **Recorder Tab**: Audio device selection, gain sliders (-20.0 dB to +20.0 dB), real-time gained VUMeters, file naming, **Upload file** button, and **Start/Stop recording** (`F5`).
- ⚙️ **Settings Tab**:
  - **Transcription Engine**: Model selection (Tiny to Large-v3), Language choice (Auto-detect + 100+ languages), and ElevenLabs API key.
  - **Output Directory**: Target folder selector for `.wav` audio files and `.txt` transcripts with **Browse...** and **Reset** buttons.
  - **Processing Switches**:
    - ⚡ **Live transcription**: Recognises the recording in chunks while it runs, so stopping only has to catch up with the tail instead of working through the whole session. Local models only — the cloud backend still transcribes after you stop. On by default.
    - 👀 **Live preview**: Shows the running text in the Transcript tab while recording.
    - 🎛️ **Separate tracks**: Dual-channel capture mode (Microphone + System Audio).
    - 🎙️ **VAD**: Voice Activity Detection pre-filtering (Silero VAD).
    - 💾 **Keep raw tracks**: Preserves unmixed separate microphone and system audio `.wav` files for post-processing in DAWs/NLEs (e.g. Audacity, Premiere Pro).
- 📄 **Transcript Tab**: Full transcript viewer with **Copy**, **Save** (export as `.txt`), and **Clear** toolbar buttons.

> ⌨️ **Global Hotkey**: Press **`F5`** at any time to start or stop recording instantly.

---

## ⚙️ Technical Highlights

- **WASAPI / PulseAudio Loopback**: Native system audio capture guarantees zero-loss recording.
- **Universal Audio Loader**: Reads native audio formats via `soundfile` with an automatic `FFmpeg` fallback for M4A, AAC, WMA, MP4, WebM, and Opus files.
- **High-DPI & WQHD Display Scaling**: Native Windows High-DPI awareness (`SetProcessDpiAwareness(2)`) and dynamic scaling for WQHD (1440p) and 4K displays.
- **Voice Activity Detection (VAD)**: Optional Silero VAD (`ggml-silero-vad.bin`) pre-filtering. When enabled, it strips silence before feeding audio to Whisper. It is disabled by default to preserve precise sentence timestamps essential for speaker diarization; `diarize.py` filters pause hallucinations using track-relative RMS energy instead.
- **Polyphase Resampling**: High-quality 16 kHz Mono resampling via `scipy.signal.resample_poly`.
- **Atomic File Operations**: Safe binary downloads and atomic settings updates prevent file corruption.
- **Vulkan / GPU Notice**: By default, GPU flags (`-ng`) fall back to CPU execution to prevent driver crashes on unsupported hardware. Compatible builds can enable GPU acceleration via `allow_gpu`.

---

## 🧪 Testing & Quality Assurance

The project includes an extensive automated test suite covering all modules:

```shell
python run_tests.py
```

- **128+ Unit Tests**:
  - 🎵 **Audio & DSP**: Resampling, downmixing, active channel detection, RMS metering, and universal audio file loading (FFmpeg fallback).
  - 🗣️ **Diarization**: Speaker attribution, crosstalk filtering, and overlap deduplication.
  - 🔒 **Security & Config**: Windows DPAPI encryption roundtrips, secret storage, and schema migrations.
  - 🖥️ **GUI & Lifecycle**: Widget geometries, high-DPI scaling, tabbed layout, gain slider meter reactivity, and window teardown handlers.

---

## 📦 Automated Release Build

To build a standalone executable and release archive locally using PyInstaller:

```shell
python build_release.py
```

The build runs on Windows, Linux and macOS and names the archive after the
current platform and version, e.g. `AudioTranscriber-v1.2.0-windows-x64.zip`.

The version is resolved in this order:

1. An explicit argument — `python build_release.py v1.2.0`
2. The `RELEASE_VERSION` environment variable (the CI sets this from the tag)
3. The git tag pointing at `HEAD`
4. `v0.0.0-dev` as a fallback

Before archiving, the script verifies that every module the app imports at
startup actually made it into the bundle and fails the build otherwise.

The output executable and archive are saved in the `dist/` directory.

---

## 📄 License & FOSS Status

This project is **100% Free and Open Source Software (FOSS)** released under the permissive **[MIT License](LICENSE)**.

```text
MIT License - Copyright (c) 2026 SecretLUL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.
```

You are free to use, modify, distribute, and integrate this software in personal or commercial projects.

