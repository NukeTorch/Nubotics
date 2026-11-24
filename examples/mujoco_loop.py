# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""A closed loop in the LIBERO-10 simulator (robosuite on MuJoCo): the policy is asked for 16 actions whenever its queue is empty,
each action stepped at 20 Hz. This is the shape of any loop of your own: build the observation (camera frames, state, instruction),
call policy.act, step the actions.

Needs a running server (`nubotics serve`) and LIBERO installed in this Python environment.
usage: python examples/mujoco_loop.py [--task 0] [--episode 0]
"""
import argparse
import json
import os

import numpy as np

from nubotics import Policy

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--model", default=None)
ap.add_argument("--url", default=None)
a = ap.parse_args()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def axis_angle(q):
    """robosuite's quaternion (x, y, z, w) -> the axis-angle the checkpoint was trained on."""
    q = np.asarray(q, np.float32)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den <= 1e-10:
        return np.zeros(3, np.float32)
    return (q[:3] / np.float32(den) * np.float32(2.0 * np.arccos(w))).astype(np.float32)


def observe(raw):
    """Cameras turned 180 degrees as in training; 8 state values."""
    images = {"image": np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
              "wrist_image": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])}
    state = np.concatenate([raw["robot0_eef_pos"], axis_angle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"]]).astype(np.float32)
    return images, state


suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(a.task)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                         camera_heights=360, camera_widths=360, control_freq=20, hard_reset=True)
env.seed(1000 + a.episode)
env.reset()
raw = env.set_init_state(suite.get_task_init_states(a.task)[a.episode])
for _ in range(10):   # let the scene settle
    raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
for robot in env.robots:
    robot.controller.use_delta = True

policy = Policy(a.model, a.url, seed=a.episode)
queue, steps, success = [], 0, False
while steps < 520 and not success:
    if not queue:
        images, state = observe(raw)
        queue = list(policy.act(images, state, task.language))
    raw, _, _, _ = env.step(queue.pop(0))
    steps += 1
    success = bool(env.check_success())
print(json.dumps({"task": task.language, "success": success, "steps": steps, "policy_calls": policy.calls}))
policy.close()
env.close()
