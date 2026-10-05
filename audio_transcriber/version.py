"""Which release this is, and what its download is called.

A packaged build carries its version in a file next to this module (RELEASE),
written by build_release.py from the tag the build was made for. A run from
source has none: it is whatever the checkout holds, and updates come with git.

The archive name lives here rather than in build_release.py because the app
needs it too: the updater picks its download by the same name the build gave
it.
"""

import os
import re

APP_NAME = "Sotto"

# What the download is built for - the only system Sotto is released for.
ARCHIVE_SLUG = "windows-x64"

RELEASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RELEASE")

# v1.2.3 or v1.2.3-rc1; the "v" is optional.
_VERSION_RE = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-(?P<pre>[0-9A-Za-z.-]+))?$")


def current(path=None):
    """The tag this build was made for ('v1.2.3'), or None when run from source.

    A build without a tag (v0.0.0-dev, see build_release.py) counts as none
    either: it was made by hand from some checkout, and every release would
    look newer than it.
    """
    path = RELEASE_FILE if path is None else path
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tag = handle.read().strip()
    except OSError:
        return None
    parsed = parse(tag)
    if parsed is None or parsed[:3] == (0, 0, 0):
        return None
    return tag if tag.startswith("v") else f"v{tag}"


def parse(tag):
    """'v1.2.3-rc1' -> (1, 2, 3, 'rc1'); 'v1.2.3' -> (1, 2, 3, None); else None."""
    match = _VERSION_RE.match((tag or "").strip())
    if not match:
        return None
    return (int(match["major"]), int(match["minor"]), int(match["patch"]),
            match["pre"])


def is_newer(candidate, installed):
    """Whether the tag `candidate` is a later release than `installed`.

    A pre-release comes before the release of the same number (1.2.0-rc1 <
    1.2.0); two pre-releases of one number are compared by their text. Anything
    that is not a version is never newer.
    """
    new, old = parse(candidate), parse(installed)
    if new is None or old is None:
        return False
    if new[:3] != old[:3]:
        return new[:3] > old[:3]
    if new[3] == old[3]:
        return False
    if new[3] is None:                  # the release after its pre-release
        return True
    if old[3] is None:
        return False
    return new[3] > old[3]


def archive_name(tag):
    """'Sotto-v1.2.3-windows-x64.zip': the download of `tag`."""
    return f"{APP_NAME}-{tag}-{ARCHIVE_SLUG}.zip"
