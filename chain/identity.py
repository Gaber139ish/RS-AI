import hashlib
from typing import Tuple

from . import crypto as hmac_crypto
from . import crypto_ed25519 as ed


class Identity:
    def __init__(self):
        self._use_ed = False
        if hasattr(ed, 'available') and ed.available():
            self._use_ed = True
            pub, sec = ed.generate_keypair()
            self.public = pub
            self.secret = sec
        else:
            pub, sec = hmac_crypto.generate_keypair()
            self.public = pub
            self.secret = sec
        self.peer_id = hashlib.sha256(self.public).hexdigest()

    def sign(self, message: bytes) -> bytes:
        if self._use_ed:
            return ed.sign(self.secret, message)
        return hmac_crypto.sign(self.secret, message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        if self._use_ed:
            return ed.verify(self.public, message, signature)
        return hmac_crypto.verify_with_secret(self.secret, message, signature)

    def pub_hex(self) -> str:
        return self.public.hex()