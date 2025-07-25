from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

SECRET_KEY_PATH = Path(os.getenv("DGM_SECRET_KEY_PATH", "/app/keys/dgm_ed25519.key"))


def _load_private_key(path: Path | None = None) -> Ed25519PrivateKey:
    if path is None:
        path = SECRET_KEY_PATH
    data = path.read_bytes()
    # The loaded key is a general type; we assert it's the expected Ed25519 type
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError(f"Expected Ed25519PrivateKey, but got {type(key).__name__}")
    return key


def sign_patch(diff_str: str, *, key_path: Path | None = None) -> str:
    """Return base64 signature for *diff_str* using Ed25519."""
    key = _load_private_key(key_path)
    sig = key.sign(diff_str.encode())
    return base64.b64encode(sig).decode()


def verify_patch(diff_str: str, sig: str, *, key_path: Path | None = None) -> bool:
    """Return True if *sig* is a valid signature of *diff_str*."""
    try:
        priv = _load_private_key(key_path)
        pub: Ed25519PublicKey = priv.public_key()
        pub.verify(base64.b64decode(sig), diff_str.encode())
        return True
    except Exception:
        return False
