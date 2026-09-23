import base64
import hashlib
import mimetypes
from pathlib import Path


def load_image_parts(image_paths, image_detail="low", cache=None):
    if not isinstance(image_paths, list) or not image_paths:
        raise ValueError("image_paths harus berupa list yang tidak kosong")
    if cache is None:
        cache = {}

    image_parts = []
    image_keys = []
    for image_path in image_paths:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image tidak ditemukan: {path}")

        cache_key = str(path)
        if cache_key not in cache:
            payload = path.read_bytes()
            mime_type = mimetypes.guess_type(path.name)[0] or ""
            if not mime_type.startswith("image/"):
                raise ValueError(f"File bukan image yang didukung: {path}")
            digest = hashlib.sha256(payload).hexdigest()
            cache[cache_key] = (
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{mime_type};base64,"
                            f"{base64.b64encode(payload).decode('ascii')}"
                        ),
                        "detail": image_detail,
                    },
                    "uuid": f"sha256:{digest}",
                },
                f"sha256:{digest}",
            )
        image_part, image_key = cache[cache_key]
        image_parts.append(image_part)
        image_keys.append(image_key)

    return image_parts, tuple(image_keys)
