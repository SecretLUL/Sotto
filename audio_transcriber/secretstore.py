"""API key storage via the Windows Data Protection API (DPAPI).

Fixes H6: the key used to sit in settings.json as plain text. DPAPI encrypts
it against the signed-in Windows account - another user, a copied folder or a
backup restored on a different machine can no longer read the value.

Deliberately ctypes plus the standard library only, no extra dependency.
"""

import base64
import ctypes
import os
from ctypes import wintypes

# Extra entropy: besides the Windows account, an attacker also needs this
# application-specific value to decrypt the blob. It predates the name Sotto and
# must stay as it is: stored keys can only be decrypted with the same value.
_ENTROPY = b"AudioTranscriber/v2/elevenlabs"


class _Blob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _to_blob(data):
    """Return (blob, backing buffer). The buffer must stay alive as long as
    the blob is in use, otherwise the API reads freed memory."""
    buffer = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _from_blob(blob):
    """Copy the blob contents out and release the memory Windows allocated
    (otherwise every call leaks a few bytes)."""
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        if blob.pbData:
            ctypes.windll.kernel32.LocalFree(blob.pbData)


def encrypt(plaintext):
    """Encrypt a string and return it base64 encoded.

    Raises:
        OSError: if the call fails.
    """
    if not plaintext:
        return ""

    data_in, _keep1 = _to_blob(plaintext.encode("utf-8"))
    entropy, _keep2 = _to_blob(_ENTROPY)
    data_out = _Blob()

    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(data_in), None, ctypes.byref(entropy),
        None, None, 0, ctypes.byref(data_out),
    )
    if not ok:
        raise OSError(f"CryptProtectData failed "
                      f"(GetLastError={ctypes.GetLastError()})")
    return base64.b64encode(_from_blob(data_out)).decode("ascii")


def decrypt(token):
    """Decrypt a base64 token. Returns "" when that is not possible
    (different user account, corrupted value)."""
    if not token:
        return ""

    try:
        raw = base64.b64decode(token.encode("ascii"), validate=True)
    except Exception:
        return ""

    data_in, _keep1 = _to_blob(raw)
    entropy, _keep2 = _to_blob(_ENTROPY)
    data_out = _Blob()

    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(data_in), None, ctypes.byref(entropy),
        None, None, 0, ctypes.byref(data_out),
    )
    if not ok:
        return ""
    try:
        return _from_blob(data_out).decode("utf-8")
    except Exception:
        return ""


def from_environment():
    """Fallback when DPAPI fails or no key has been stored yet."""
    return os.environ.get("ELEVENLABS_API_KEY", "").strip()
