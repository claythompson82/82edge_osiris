from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from dgm_kernel import crypto_sign


def _gen_key(path):
    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return key


def test_round_trip(tmp_path, monkeypatch):
    key_path = tmp_path / "key.pem"
    _gen_key(key_path)
    monkeypatch.setattr(crypto_sign, "SECRET_KEY_PATH", key_path)

    diff = "a\nb"
    sig = crypto_sign.sign_patch(diff)
    assert crypto_sign.verify_patch(diff, sig) is True


def test_verify_fails_on_tamper(tmp_path, monkeypatch):
    key_path = tmp_path / "key.pem"
    _gen_key(key_path)
    monkeypatch.setattr(crypto_sign, "SECRET_KEY_PATH", key_path)

    diff = "orig"
    sig = crypto_sign.sign_patch(diff)
    assert crypto_sign.verify_patch("other", sig) is False

