# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""Nubotics policy: observation in, actions out, through a NukeTorch server on this Mac.

    from nubotics import Policy
    policy = Policy("~/NukeTorch/models/Nubotics")                   # the server `nubotics serve` starts
    actions = policy.act({"image": rgb, "wrist_image": rgb2}, state, "put the bowl on the plate")   # float32 [16][7]

images: {camera: uint8 [H][W][3]} for policy.cameras; state: float32 [policy.state_dim] in robot units; task: the instruction.
Each call's seed is drawn from `seed` and the call's number, so a run repeats exactly.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .client import DEFAULT_URL, Client
from .formats import Observation
from .processor import Processor

DEFAULT_MODEL = os.path.expanduser("~/NukeTorch/models/Nubotics")


class Policy:
    def __init__(self, model_dir=None, url=None, seed=0, max_batch=16, timeout=120):
        self.processor = Processor(model_dir or os.environ.get("NUBOTICS_MODEL", DEFAULT_MODEL))
        self.client = Client(url or os.environ.get("NUBOTICS_URL", DEFAULT_URL), timeout)
        p = self.processor
        self.cameras, self.state_dim, self.action_dim = p.cameras, p.state_dim, p.action_dim
        self.n_action_steps, self.embodiment_tag = p.n_action_steps, p.embodiment_tag
        self.seed, self.calls, self.max_batch = int(seed), 0, int(max_batch)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=self.max_batch)

    def _next_seed(self):
        with self._lock:
            s = (self.seed * 1000003 + self.calls) & 0xFFFFFFFFFFFFFFFF
            self.calls += 1
        return s

    def _call(self, o, seed, steps, env_gripper):
        ids, frames, grids, state = self.processor.observation(o.images, o.state, o.task)
        chunk = self.client.act(ids, frames, state, self.processor.mean, self.processor.std, grids=grids, seed=seed)
        return self.processor.actions(chunk, steps, env_gripper=env_gripper)

    def act(self, images, state, task, steps=None, env_gripper=True):
        """One robot, one call -> float32 [steps][action_dim] in robot units (steps: the checkpoint's n_action_steps by default).
        env_gripper True applies the checkpoint's gripper step (LIBERO: the simulator's -1 / +1); False returns the raw value."""
        return self._call(Observation(images, state, task), self._next_seed(), steps, env_gripper)

    def act_many(self, observations, steps=None, env_gripper=True):
        """Several robots at once: [Observation] -> float32 [B][steps][action_dim]. The requests go out together; the server
        answers the ones waiting together in one call, each robot with its own instruction."""
        seeds = [self._next_seed() for _ in observations]
        futures = [self._pool.submit(self._call, o, s, steps, env_gripper) for o, s in zip(observations, seeds)]
        out = [f.result() for f in futures]
        n = steps or self.n_action_steps
        return np.stack(out) if out else np.zeros((0, n, self.action_dim), np.float32)

    def close(self):
        self._pool.shutdown(wait=False)
