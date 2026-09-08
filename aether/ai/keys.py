"""API key storage, via the OS credential store.

Keys go to Windows Credential Manager through ``keyring`` -- never into the
settings JSON, the session database, or the .npz files, all of which a user
might reasonably share along with a scan.
"""
from __future__ import annotations

SERVICE = "Aether"

# Keys saved before the project was renamed live under the old service
# name. They are still read, so nobody has to re-enter a key after an
# update; anything saved from now on goes under the current name.
LEGACY_SERVICE = "RTLBaseline"


class KeyStoreError(RuntimeError):
    pass


def _backend():
    try:
        import keyring
    except ImportError as exc:
        raise KeyStoreError(
            "The 'keyring' package is not installed, so API keys cannot be "
            "stored securely. Install it with: pip install keyring"
        ) from exc
    return keyring


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
        kr = _backend()
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
