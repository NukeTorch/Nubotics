# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""HTTP client for NukeTorch's policy route, POST /v1/nuketorch/policy/act.

A request is multipart/form-data with two parts:
  request  JSON: token_ids [n], state [width], height, width, mean [3], std [3], optional grids [[t, h, w]] per frame,
           seed, noise [horizon * action width], response_format ("json" or "binary")
  frames   raw 8-bit RGB, the cameras one after another, each height x width x 3 bytes
The answer is the model's normalized action chunk, float32 [horizon][width]. Standard library only.
"""
import json
import urllib.error
import urllib.request
import uuid

import numpy as np

ROUTE = "/v1/nuketorch/policy/act"
LOCAL = "127.0.0.1"                       # this Mac (loopback); the NukeTorch server listens here by default
DEFAULT_URL = f"http://{LOCAL}:8080"


class PolicyError(RuntimeError):
    pass


def encode(token_ids, frames, state, mean, std, grids=None, seed=None, noise=None):
    """-> (body bytes, content type). frames: uint8 [cameras][h][w][3]."""
    frames = np.ascontiguousarray(frames, np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"frames must be uint8 [cameras][h][w][3], got {frames.shape}")
    request = {
        "token_ids": [int(x) for x in np.asarray(token_ids).ravel()],
        "state": [float(x) for x in np.asarray(state, np.float32).ravel()],
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "response_format": "binary",
    }
    if grids is not None:
        request["grids"] = [[int(v) for v in g] for g in np.asarray(grids)]
    if seed is not None:
        request["seed"] = int(seed)
    if noise is not None:
        request["noise"] = [float(x) for x in np.asarray(noise, np.float32).ravel()]
    boundary = "nubotics-" + uuid.uuid4().hex
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\nContent-Type: application/json\r\n\r\n'.encode(),
        json.dumps(request).encode(), b"\r\n",
        f'--{boundary}\r\nContent-Disposition: form-data; name="frames"; filename="frames"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(),
        frames.tobytes(), b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


class Client:
    def __init__(self, url=DEFAULT_URL, timeout=120):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def act(self, token_ids, frames, state, mean, std, grids=None, seed=None, noise=None):
        """One observation -> float32 [horizon][width], the model's normalized actions."""
        body, ctype = encode(token_ids, frames, state, mean, std, grids, seed, noise)
        req = urllib.request.Request(self.url + ROUTE, data=body, method="POST", headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = r.read()
                horizon = int(r.headers.get("X-NukeTorch-Horizon", "0"))
                width = int(r.headers.get("X-NukeTorch-Width", "0"))
        except urllib.error.HTTPError as e:
            raise PolicyError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None
        except urllib.error.URLError as e:
            raise PolicyError(f"no NukeTorch server at {self.url} ({e.reason}); start it with `nubotics serve`") from None
        a = np.frombuffer(data, dtype="<f4")
        if not horizon or not width or a.size != horizon * width:
            raise PolicyError(f"unexpected answer: {a.size} values for horizon {horizon} x width {width}")
        return a.reshape(horizon, width)

    def models(self):
        with urllib.request.urlopen(self.url + "/v1/models", timeout=self.timeout) as r:
            return json.loads(r.read())
