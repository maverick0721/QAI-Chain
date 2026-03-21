from crypto.pqc.signature import sign


def sign_transaction(tx, private_key):

    message = str(tx.to_dict())

    tx.signature = sign(message, private_key)
    tx.signature_algorithm = "dilithium3_or_fallback"

    return tx