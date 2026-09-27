"""API key storage, via the OS credential store.

Keys go to the operating system's credential store through ``keyring``
(Windows Credential Manager, the macOS Keychain, or a Secret Service such as
GNOME Keyring / KWallet on Linux) -- never into the settings JSON, the
session database, or the .npz files, all of which a user might reasonably
share along with a scan.
"""
from __future__ import annotations

SERVICE = "Aether"

# Keys saved before the project was renamed live under the old service
# name. They are still read, so nobody has to re-enter a key after an
# update; anything saved from now on goes under the current name.
LEGACY_SERVICE = "RTLBaseline"

# Shown wherever the UI tells the user where keys live.
STORE_DESCRIPTION = "your system's credential store"

# keyring backends that do not actually protect a secret. The fail and null
# backends store nothing; everything in keyrings.alt writes to a file under
# the user's home directory (plaintext, or encrypted with a password prompt
# on the console, which a GUI app cannot answer). keyring will pick one of
# these silently when no real store is available -- typical on a headless
# or minimal Linux install with no Secret Service running.
_INSECURE_BACKENDS = (
    "keyring.backends.fail.",
    "keyring.backends.null.",
    "keyrings.alt.",
)

_NO_STORE_MESSAGE = (
    "No secure credential store is available, so API keys cannot be saved. "
    "On Linux, install and unlock a Secret Service provider such as GNOME "
    "Keyring or KWallet, then try again."
)


class KeyStoreError(RuntimeError):
    pass


def _insecure(backend) -> bool:
    cls = type(backend)
    name = "%s.%s" % (cls.__module__, cls.__qualname__)
    return name.startswith(_INSECURE_BACKENDS)


def _backend(*, secure: bool = True):
    """The keyring module, checked against the backend it resolved to.

    With ``secure`` set, refuse to hand it back if the key would end up in a
    store that does not protect it. Deleting passes ``secure=False`` so a key
    that already landed in such a store can still be removed.
    """
    try:
        import keyring
    except ImportError as exc:
        raise KeyStoreError(
            "The 'keyring' package is not installed, so API keys cannot be "
            "stored securely. Install it with: pip install keyring"
        ) from exc
    if secure:
        active = keyring.get_keyring()
        # A ChainerBackend wraps several backends and writes to the first
        # that accepts; treat the chain as insecure if any member is.
        members = getattr(active, "backends", None) or [active]
        if _insecure(active) or any(_insecure(b) for b in members):
            raise KeyStoreError(_NO_STORE_MESSAGE)
    return keyring


def storage_problem() -> str | None:
    """Why keys cannot be stored securely here, or None if they can."""
    try:
        _backend()
    except KeyStoreError as exc:
        return str(exc)
    return None


def set_key(provider: str, key: str) -> None:
    kr = _backend()
    try:
        kr.set_password(SERVICE, provider, key)
    except Exception as exc:
        raise KeyStoreError("Could not save the key: %s" % exc) from exc


def get_key(provider: str) -> str | None:
    try:
        kr = _backend()
    except KeyStoreError:
        return None
    for service in (SERVICE, LEGACY_SERVICE):
        try:
            key = kr.get_password(service, provider)
        except Exception:
            continue
        if key:
            return key
    return None


def delete_key(provider: str) -> None:
    """Remove the key from both the current and the pre-rename store, so
    'Remove' in the UI really does remove it."""
    try:
        kr = _backend(secure=False)
    except KeyStoreError:
        return
    for service in (SERVICE, LEGACY_SERVICE):
        try:
            kr.delete_password(service, provider)
        except Exception:
            pass


def has_key(provider: str) -> bool:
    return bool(get_key(provider))


def masked(provider: str) -> str:
    """A key preview safe to show in the UI."""
    key = get_key(provider)
    if not key:
        return "not set"
    return "%s...%s (%d chars)" % (key[:7], key[-4:], len(key)) if len(key) > 14 else "set"
