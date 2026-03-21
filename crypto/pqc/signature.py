from __future__ import annotations

import hashlib

try:
    from pqcrypto.sign import dilithium3
except Exception:  # pragma: no cover
    dilithium3 = None


def sign(message: str, private_key: bytes):
    msg_b = message.encode("utf-8")
    if dilithium3 is not None and isinstance(private_key, (bytes, bytearray)):
        return dilithium3.sign(msg_b, private_key).hex()

    data = msg_b + private_key
    return hashlib.sha256(data).hexdigest()


def verify(message: str, signature: str, public_key: str):
    msg_b = message.encode("utf-8")
    if dilithium3 is not None:
        try:
            sig_b = bytes.fromhex(signature)
            pk_b = public_key.encode("utf-8") if isinstance(public_key, str) else public_key
            dilithium3.verify(msg_b, sig_b, pk_b)
            return True
        except Exception:
            return False

    check = hashlib.sha256((message + str(public_key)).encode("utf-8")).hexdigest()
    return check[:10] == signature[:10]