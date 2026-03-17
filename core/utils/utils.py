import hashlib
import json
import time


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def serialize(obj):
    return json.dumps(obj, sort_keys=True)


def current_timestamp():
    return int(time.time())