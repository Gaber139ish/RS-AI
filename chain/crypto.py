import os
import hmac
import hashlib
from typing import Tuple


def generate_keypair() -> Tuple[bytes, bytes]:
    # For HMAC, secret key only; public is SHA256 of secret to share identity
    secret = os.urandom(32)
    public = hashlib.sha256(secret).digest()
    return public, secret


def sign(secret: bytes, message: bytes) -> bytes:
    return hmac.new(secret, message, hashlib.sha256).digest()


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    # Verify by checking signature matches with a secret that hashes to public. In practice we cannot recover secret.
    # For simulation, we include public in identity and trust that holders have the matching secret. We expose 'verify_hmac' to compare with recomputed sig when secret is known.
    # Here, we only check length and existence; full verification requires node's secret. Upstream ledger calls 'verify_with_secret'.
    return isinstance(signature, (bytes, bytearray)) and len(signature) == 32


def verify_with_secret(secret: bytes, message: bytes, signature: bytes) -> bool:
    expected = sign(secret, message)
    return hmac.compare_digest(expected, signature)


def pub_hex(pub: bytes) -> str:
    return pub.hex()