import hashlib
import os


class PQCKeypair:

    def __init__(self):

        self.private_key = os.urandom(32)
        self.public_key = hashlib.sha256(self.private_key).hexdigest()

    def get_public_key(self):
        return self.public_key