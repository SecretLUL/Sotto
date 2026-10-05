"""Device enumeration and loopback matching.

Fixes M4 and M6:
  * The previous version listed every device three times (MME, DirectSound,
    WASAPI). As a result the selection often landed on an MME device at
    44.1 kHz although the same microphone is available through WASAPI at
    48 kHz - the same rate as the loopback, so less resampling and less drift.
  * Loopback matching compared the first 15 characters of the name and
    otherwise fell back to "any loopback device" without saying so.
"""

from dataclasses import dataclass

WASAPI_HOST_NAMES = ("WASAPI",)

# A loopback name that is not "<playback name>..." still counts as the same
# device if it shares at least this much of the playback name from the start:
# an absolute minimum and a share of the name, whichever is larger.
MIN_FUZZY_PREFIX = 12
FUZZY_SHARE = 0.6


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_rate: int
    is_loopback: bool

    @property
    def label(self):
        return f"{self.index}: {self.name}"

    @property
    def is_input(self):
        return self.max_input_channels > 0

    @property
    def is_output(self):
        return self.max_output_channels > 0


class DeviceError(RuntimeError):
    pass


def enumerate_devices(pa, wasapi_only=True):
    """Read all devices from a PyAudio instance.

    wasapi_only=True hides the MME/DirectSound duplicates.
    """
    host_names = {}
    for index in range(pa.get_host_api_count()):
        try:
            host_names[index] = pa.get_host_api_info_by_index(index)["name"]
        except Exception:
            host_names[index] = f"HostApi{index}"

    wasapi_ids = {index for index, name in host_names.items()
                  if any(tag in name for tag in WASAPI_HOST_NAMES)}
    if wasapi_only and not wasapi_ids:
        wasapi_only = False

    devices = []
    for index in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(index)
        except Exception:
            continue
        if wasapi_only and info.get("hostApi") not in wasapi_ids:
            continue

        devices.append(Device(
            index=index,
            name=str(info.get("name", f"Device {index}")),
            host_api=host_names.get(info.get("hostApi"), "?"),
            max_input_channels=int(info.get("maxInputChannels", 0)),
            max_output_channels=int(info.get("maxOutputChannels", 0)),
            default_rate=int(info.get("defaultSampleRate", 48000)),
            is_loopback=bool(info.get("isLoopbackDevice", False)),
        ))
    return devices


def microphone_candidates(devices):
    """Real capture devices - without loopback and stereo-mix pseudo devices."""
    blocked = ("loopback", "stereomix", "stereo mix", "what u hear")
    result = []
    for device in devices:
        if not device.is_input or device.is_loopback:
            continue
        if any(keyword in device.name.lower() for keyword in blocked):
            continue
        result.append(device)
    return result


def playback_candidates(devices):
    """Playback devices whose output can be captured."""
    return [d for d in devices if d.is_output and not d.is_loopback]


def loopback_devices(devices):
    return [d for d in devices if d.is_loopback and d.is_input]


def find_loopback_for(devices, playback_device):
    """Find the loopback counterpart of a playback device.

    Returns: (Device or None, human-readable reason).

    Instead of the 15-character heuristic of the previous version, the longest
    common prefix is compared across ALL loopback devices and the unambiguous
    best match wins. If none is unambiguous the result is None plus a reason -
    a clear message beats silently capturing the wrong source.
    """
    candidates = loopback_devices(devices)
    if not candidates:
        return None, ("No WASAPI loopback device was found. "
                      "System audio cannot be captured.")

    if playback_device is None:
        return None, "No playback device selected."

    if playback_device.is_loopback:
        return playback_device, "Loopback device selected directly."

    target = _normalize(playback_device.name)

    scored = []
    for device in candidates:
        name = _normalize(device.name)
        # Loopback names are typically "<device name> [Loopback]"
        prefix = _common_prefix_len(target, name)
        if name == target:
            score = 2000 + prefix              # literally "<name> [Loopback]"
        elif name.startswith(target):
            score = 1000 + prefix              # starts with the playback name
        elif prefix >= max(MIN_FUZZY_PREFIX, FUZZY_SHARE * len(target)):
            score = prefix                     # same name, cut or reworded
        else:
            # A few shared letters are no match: "speakers (" is the start of
            # every Windows speaker name, and accepting it picked the wrong
            # device without a word whenever it was the only one left.
            continue
        scored.append((score, device))

    if not scored:
        return None, (f"No matching loopback device was found for "
                      f"'{playback_device.name}'. Please select the loopback "
                      f"device directly.")

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]

    if len(scored) > 1 and scored[1][0] == best_score:
        return None, (f"The match for '{playback_device.name}' is ambiguous "
                      f"('{best.name}' and '{scored[1][1].name}' fit equally "
                      f"well). Please select the loopback device directly.")

    return best, f"'{playback_device.name}' -> '{best.name}'"


def by_label(devices, label):
    """Find a device by its display label ('4: Headphones (...)').

    If the label is not found (device unplugged, indices shifted) the plain
    name is tried as well - Windows device indices change regularly, the name
    stays.
    """
    for device in devices:
        if device.label == label:
            return device
    if ":" in label:
        wanted = label.split(":", 1)[1].strip()
        for device in devices:
            if device.name == wanted:
                return device
    return None


def _normalize(name):
    return " ".join(name.lower().replace("[loopback]", "").split())


def _common_prefix_len(a, b):
    count = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        count += 1
    return count
