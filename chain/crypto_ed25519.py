from typing import Tuple

try:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.exceptions import BadSignatureError
    _HAS_ED = True
except Exception:  # pragma: no cover
    _HAS_ED = False


def available() -> bool:
    return _HAS_ED


def generate_keypair() -> Tuple[bytes, bytes]:
    if not _HAS_ED:
        raise ImportError("PyNaCl not available")
    sk = SigningKey.generate()
    vk = sk.verify_key
    return bytes(vk), bytes(sk)


def sign(secret: bytes, message: bytes) -> bytes:
    if not _HAS_ED:
        raise ImportError("PyNaCl not available")
    sk = SigningKey(secret)
    return bytes(sk.sign(message).signature)


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    if not _HAS_ED:
        return False
    try:
        vk = VerifyKey(public)
        vk.verify(message, signature)
        return True
    except BadSignatureError:
        return False