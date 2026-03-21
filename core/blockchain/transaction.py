from core.utils.utils import serialize, sha256


class Transaction:

    def __init__(self, sender, receiver, amount, signature=None, signature_algorithm=None):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.signature = signature
        self.signature_algorithm = signature_algorithm

    def to_dict(self):

        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
        }

    def hash(self):

        return sha256(serialize(self.to_dict()))