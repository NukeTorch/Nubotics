# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""Client-side tests: no server, no GPU. Model-folder tests run when NUBOTICS_MODEL (or ~/NukeTorch/models/Nubotics) is present.
    python -m unittest discover tests"""
import json
import os
import pathlib
import unittest

import numpy as np

from nubotics import Embodiment, Observation
from nubotics.client import encode
from nubotics.formats import from_gr00t, from_lerobot, to_gr00t

MODEL = pathlib.Path(os.environ.get("NUBOTICS_MODEL", os.path.expanduser("~/NukeTorch/models/Nubotics")))


def parts(body, ctype):
    """Split a multipart body into {name: bytes}."""
    boundary = ctype.split("boundary=")[1].encode()
    out = {}
    for chunk in body.split(b"--" + boundary)[1:-1]:
        head, _, data = chunk.strip(b"\r\n").partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        out[name] = data
    return out


class Encode(unittest.TestCase):
    def test_request_follows_the_route(self):
        frames = np.arange(2 * 256 * 256 * 3, dtype=np.uint64).astype(np.uint8).reshape(2, 256, 256, 3)
        body, ctype = encode([1, 2, 3], frames, np.zeros(132, np.float32), [0.5] * 3, [0.5] * 3, grids=[[1, 16, 16]] * 2, seed=7)
        p = parts(body, ctype)
        self.assertEqual(set(p), {"request", "frames"})
        r = json.loads(p["request"])
        self.assertEqual((r["height"], r["width"], r["seed"], r["response_format"]), (256, 256, 7, "binary"))
        self.assertEqual(r["grids"], [[1, 16, 16], [1, 16, 16]])
        self.assertEqual(len(r["state"]), 132)
        self.assertEqual(p["frames"], frames.tobytes())   # raw RGB, cameras one after another
        self.assertEqual(r["height"] % 32, 0)

    def test_refuses_frames_without_channels(self):
        with self.assertRaises(ValueError):
            encode([1], np.zeros((2, 256, 256), np.uint8), [0.0], [0.5] * 3, [0.5] * 3)


class Formats(unittest.TestCase):
    emb = Embodiment.load("libero_sim")

    def test_gr00t_nested_round_trip(self):
        obs = {"video": {k: np.zeros((2, 1, 64, 64, 3), np.uint8) for k in self.emb.video},
               "state": {k: np.full((2, 1, b - a), i, np.float32) for i, (k, (a, b)) in enumerate(self.emb.state.items())},
               "language": {self.emb.language: [["a"], ["b"]]}}
        rows, flat = from_gr00t(obs, self.emb)
        self.assertFalse(flat)
        self.assertEqual([r.task for r in rows], ["a", "b"])
        self.assertEqual(rows[0].state.tolist(), [0, 1, 2, 3, 4, 5, 6, 6])
        out = to_gr00t(np.zeros((2, 16, 7), np.float32), self.emb, flat)
        self.assertEqual(out["gripper"].shape, (2, 16, 1))

    def test_gr00t_refuses_wrong_dtype(self):
        obs = {"video": {k: np.zeros((1, 1, 8, 8, 3), np.float32) for k in self.emb.video}, "state": {}, "language": {}}
        with self.assertRaisesRegex(ValueError, "np.uint8"):
            from_gr00t(obs, self.emb)

    def test_lerobot_float_images(self):
        batch = {"observation.images.image": np.ones((1, 3, 8, 8), np.float32), "observation.state": np.zeros((1, 8)), "task": "x"}
        rows = from_lerobot(batch, ["image"])
        self.assertEqual(rows[0].images["image"].dtype, np.uint8)
        self.assertEqual(int(rows[0].images["image"].max()), 255)


@unittest.skipUnless((MODEL / "policy_preprocessor.json").exists(), "the Nubotics model folder is not present")
class ModelFolder(unittest.TestCase):
    def setUp(self):
        from nubotics import Processor
        self.p = Processor(MODEL)

    def test_observation_shapes(self):
        rng = np.random.default_rng(0)
        images = {c: rng.integers(0, 256, (360, 360, 3), dtype=np.uint8) for c in self.p.cameras}
        ids, frames, grids, state = self.p.observation(images, np.zeros(self.p.state_dim, np.float32), "put the bowl on the plate")
        self.assertEqual(frames.shape, (2, 256, 256, 3))
        self.assertEqual(grids.tolist(), [[1, 16, 16]] * 2)
        self.assertEqual(state.shape, (132,))
        self.assertEqual(int((ids == self.p.tok.token_to_id("<|image_pad|>")).sum()), 2 * 64)

    def test_actions_in_robot_units(self):
        a = self.p.actions(np.full((40, 132), 0.3, np.float32))
        self.assertEqual(a.shape, (16, 7))
        self.assertTrue(set(np.unique(a[:, -1])) <= {-1.0, 1.0})


if __name__ == "__main__":
    unittest.main()
