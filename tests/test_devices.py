"""Device matching: which loopback belongs to the chosen playback device."""

import unittest

from audio_transcriber.audio import devices


def playback(index, name):
    return devices.Device(index, name, "WASAPI", 0, 2, 48000, False)


def loopback(index, name):
    return devices.Device(index, f"{name} [Loopback]", "WASAPI", 2, 0, 48000, True)


def microphone(index, name):
    return devices.Device(index, name, "WASAPI", 1, 0, 48000, False)


class TestFindLoopback(unittest.TestCase):
    def test_the_counterpart_of_a_playback_device_is_found(self):
        speakers = playback(4, "Speakers (Realtek Audio)")
        found, reason = devices.find_loopback_for(
            [speakers, loopback(9, "Speakers (Realtek Audio)"),
             loopback(10, "Headphones (USB Audio)")], speakers)
        self.assertEqual(found.index, 9)
        self.assertIn("->", reason)

    def test_a_loopback_chosen_directly_is_used_as_it_is(self):
        chosen = loopback(9, "Speakers (Realtek Audio)")
        found, _reason = devices.find_loopback_for([chosen], chosen)
        self.assertIs(found, chosen)

    def test_a_few_shared_letters_are_no_match(self):
        """Regression: 'speakers (' is how every Windows speaker name starts.
        The 6-character minimum took the only loopback left - another device -
        without saying so."""
        usb = playback(5, "Speakers (USB Audio)")
        found, reason = devices.find_loopback_for(
            [usb, loopback(9, "Speakers (Realtek Audio)")], usb)
        self.assertIsNone(found)
        self.assertIn("No matching loopback device", reason)
        self.assertIn("select the loopback device directly", reason)

    def test_a_cut_or_reworded_name_still_matches(self):
        """The same device under a name that was truncated on the way."""
        headset = playback(6, "Headphones (PRO X 2 LIGHTSPEED Gaming Headset)")
        found, _reason = devices.find_loopback_for(
            [headset, loopback(11, "Headphones (PRO X 2 LIGHTSPEED)")], headset)
        self.assertEqual(found.index, 11)

    def test_two_equally_good_matches_are_ambiguous(self):
        speakers = playback(4, "Speakers")
        found, reason = devices.find_loopback_for(
            [speakers, loopback(9, "Speakers (Realtek Audio)"),
             loopback(10, "Speakers (NVIDIA High Definition Audio)")], speakers)
        self.assertIsNone(found)
        self.assertIn("ambiguous", reason)

    def test_an_exact_counterpart_beats_a_name_that_merely_starts_alike(self):
        """'Speakers [Loopback]' is this device; 'Speakers (Realtek Audio)
        [Loopback]' is another one whose name happens to begin the same way.
        Both used to score alike, so the answer was 'ambiguous'."""
        speakers = playback(4, "Speakers")
        found, _reason = devices.find_loopback_for(
            [speakers, loopback(10, "Speakers (Realtek Audio)"),
             loopback(9, "Speakers")], speakers)
        self.assertEqual(found.index, 9)

    def test_no_loopback_devices_at_all(self):
        speakers = playback(4, "Speakers (Realtek Audio)")
        found, reason = devices.find_loopback_for([speakers], speakers)
        self.assertIsNone(found)
        self.assertIn("No WASAPI loopback device", reason)

    def test_no_playback_device_selected(self):
        found, reason = devices.find_loopback_for(
            [loopback(9, "Speakers (Realtek Audio)")], None)
        self.assertIsNone(found)
        self.assertIn("No playback device selected", reason)


class TestDeviceLookup(unittest.TestCase):
    def test_a_label_is_found_even_when_the_indices_shifted(self):
        """Windows renumbers devices; the name stays."""
        current = [microphone(2, "Microphone (Yeti X)"),
                   microphone(7, "Line In (Realtek Audio)")]
        found = devices.by_label(current, "1: Microphone (Yeti X)")
        self.assertEqual(found.index, 2)

    def test_an_unknown_label_is_not_guessed(self):
        self.assertIsNone(devices.by_label(
            [microphone(2, "Microphone (Yeti X)")], "1: Something else"))

    def test_microphone_candidates_hide_loopbacks_and_stereo_mix(self):
        everything = [microphone(1, "Microphone (Yeti X)"),
                      microphone(2, "Stereo Mix (Realtek Audio)"),
                      loopback(3, "Speakers (Realtek Audio)"),
                      playback(4, "Speakers (Realtek Audio)")]
        names = [d.name for d in devices.microphone_candidates(everything)]
        self.assertEqual(names, ["Microphone (Yeti X)"])


if __name__ == "__main__":
    unittest.main()
