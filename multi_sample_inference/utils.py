import base64
import hashlib
import json


def get_folder_name(config: dict, length=6) -> str:
    encoded = json.dumps(config, sort_keys=True).encode()

    # use .digest() instead of .hexdigest() to get raw binary data
    digest = hashlib.md5(encoded).digest()
    b64_bytes = base64.urlsafe_b64encode(digest)
    folder_name = b64_bytes.decode().rstrip("=")

    return folder_name[:length]
