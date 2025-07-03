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


def _load_private_key(path: Path = SECRET_KEY_PATH) -> Ed25519PrivateKey:
    data = path.read_bytes()
    return serialization.load_pem_private_key(data, password=None)


def sign_patch(diff_str: str, *, key_path: Path = SECRET_KEY_PATH) -> str:
    """Return base64 signature for *diff_str* using Ed25519."""
    key = _load_private_key(key_path)
    sig = key.sign(diff_str.encode())
    return base64.b64encode(sig).decode()


def verify_patch(diff_str: str, sig: str, *, key_path: Path = SECRET_KEY_PATH) -> bool:
    """Return True if *sig* is a valid signature of *diff_str*."""
    try:
        priv = _load_private_key(key_path)
        pub: Ed25519PublicKey = priv.public_key()
        pub.verify(base64.b64decode(sig), diff_str.encode())
        return True
    except Exception:
        return False
