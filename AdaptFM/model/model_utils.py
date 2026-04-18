import inspect
import numpy as np

def extract_tunable_params(func):
    sig = inspect.signature(func)
    params = {}

    for name, p in sig.parameters.items():
        if p.default is inspect.Parameter.empty:
            continue

        params[name] = {
            "type": type(p.default),
            "default": p.default,
        }

    return params


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    minv = img.min()
    maxv = img.max()

    if maxv == minv:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - minv) / (maxv - minv)
    img = (img * 255.0).clip(0, 255)

    return img.astype(np.uint8)



