from __future__ import annotations

import hashlib
import os

try:
    from pqcrypto.sign import dilithium3
except Exception:  # pragma: no cover
    dilithium3 = None


class PQCKeypair:

    def __init__(self):
        self.algorithm = "dilithium3" if dilithium3 is not None else "shim-sha256"
        if dilithium3 is not None:
            public_key, secret_key = dilithium3.generate_keypair()
            self.private_key = secret_key
            self.public_key = public_key
        else:
            self.private_key = os.urandom(32)
            self.public_key = hashlib.sha256(self.private_key).hexdigest().encode("utf-8")

    def get_public_key(self):
        return self.public_key.decode("utf-8", errors="ignore") if isinstance(self.public_key, bytes) else self.public_key