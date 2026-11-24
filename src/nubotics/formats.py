# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""Observation and action formats.

Turns the observations that existing robot stacks send into one plain observation, and the policy's actions back into the
formats those stacks expect. Pure data: numpy only, no network.

Plain observation: images {camera: uint8 [H][W][3]}, state float32 [D] in robot units (the checkpoint's order), task (the
instruction). Actions: float32 [steps][D] in robot units.

Formats:
  NVIDIA Isaac-GR00T policy interface (gr00t/policy/gr00t_policy.py, commit 3b49067943eb1ab2eb2203b93d461c4390c78da3)
    nested  {"video": {camera: uint8 [B][T][H][W][3]}, "state": {key: float32 [B][T][D]}, "language": {key: [[text]]}}
            -> {key: float32 [B][horizon][D]}
    flat    {"video.<camera>": ..., "state.<key>": ..., "<language key>": B texts}   (its simulation wrapper)
            -> {"action.<key>": float32 [B][horizon][D]}
  LeRobot   {"observation.images.<camera>": float [B][3][H][W] in 0..1 or uint8 [B][H][W][3], "observation.state": [B][D],
             "task": B texts} -> [B][steps][D]
"""
import json
import pathlib
from collections import namedtuple

import numpy as np

Observation = namedtuple("Observation", "images state task")


class Embodiment:
    """Names the cameras, cuts the state and action vectors into the robot's keys, and gives the action horizon."""

    def __init__(self, spec):
        self.tag = spec["embodiment_tag"]
        self.video = list(spec["video"])
        self.language = spec["language"]
        self.action_horizon = int(spec["action_horizon"])
        self.state = {k: (int(a), int(b)) for k, (a, b) in spec["state"].items()}
        self.action = {k: (int(a), int(b)) for k, (a, b) in spec["action"].items()}
        self.state_dim = max(b for _, b in self.state.values())
        self.action_dim = max(b for _, b in self.action.values())

    @classmethod
    def load(cls, tag_or_file):
        """An embodiment shipped in nubotics/embodiments (by tag) or a JSON file of your own."""
        p = pathlib.Path(tag_or_file)
        if not p.suffix:
            p = pathlib.Path(__file__).parent / "embodiments" / f"{tag_or_file}.json"
        return cls(json.loads(p.read_text()))


def _need(ok, message):
    if not ok:
        raise ValueError(message)


# ---- NVIDIA Isaac-GR00T ----
def from_gr00t(observation, emb):
    """One request's observation (nested or flat) -> ([Observation] * B, flat). Error texts follow Gr00tPolicy.check_observation."""
    _need(isinstance(observation, dict), f"Observation must be a dictionary. Got {type(observation)}")
    flat = not any(m in observation for m in ("video", "state", "language"))
    if not flat:
        for m in ("video", "state", "language"):
            _need(m in observation, f"Observation must contain a '{m}' key")
            _need(isinstance(observation[m], dict), f"Observation '{m}' must be a dictionary. Got {type(observation[m])}: {observation[m]}")
    if flat:
        name = lambda m, k: k if m == "language" else f"{m}.{k}"
        get = lambda m, k: observation.get(name(m, k))
    else:
        name = lambda m, k: k
        get = lambda m, k: observation[m].get(k)
    size = []

    def batch(what, key, n):
        size.append(n)
        _need(n == size[0], f"{what} key '{key}' must have batch size {size[0]}. Got {n}")

    videos, states = {}, {}
    for k in emb.video:
        v, key = get("video", k), name("video", k)
        _need(v is not None, f"Video key '{key}' must be in observation")
        _need(isinstance(v, np.ndarray), f"Video key '{key}' must be a numpy array. Got {type(v)}")
        _need(v.dtype == np.uint8, f"Video key '{key}' must be a numpy array of type np.uint8. Got {v.dtype}")
        _need(v.ndim == 5, f"Video key '{key}' must be a numpy array of shape (B, T, H, W, C), got {v.shape}")
        batch("Video", key, v.shape[0])
        _need(v.shape[1] == 1, f"Video key '{key}'s horizon must be 1. Got {v.shape[1]}")
        _need(v.shape[-1] == 3, f"Video key '{key}'s channel 'C' must be 3. Got {v.shape[-1]}")
        videos[k] = v
    for k, (a, b) in emb.state.items():
        v, key = get("state", k), name("state", k)
        _need(v is not None, f"State key '{key}' must be in observation")
        _need(isinstance(v, np.ndarray), f"State key '{key}' must be a numpy array. Got {type(v)}")
        _need(v.dtype == np.float32, f"State key '{key}' must be a numpy array of type np.float32. Got {v.dtype}")
        _need(v.ndim == 3, f"State key '{key}' must be a numpy array of shape (B, T, D), got {v.shape}")
        batch("State", key, v.shape[0])
        _need(v.shape[1] == 1, f"State key '{key}'s horizon must be 1. Got {v.shape[1]}")
        _need(v.shape[2] == b - a, f"State key '{key}' must have {b - a} values. Got {v.shape[2]}")
        states[k] = v
    t, key = get("language", emb.language), emb.language
    _need(t is not None, f"Language key '{key}' must be in observation")
    if flat:   # the simulation wrapper: a text, or B texts as tuple / list / array
        if isinstance(t, np.ndarray):
            t = t.reshape(-1).tolist()
        if isinstance(t, str):
            t = [t]
        _need(isinstance(t, (tuple, list)), f"Language key '{key}' must be a tuple, list, or numpy array. Got {type(t)}")
        _need(len(t) > 0, f"Language key '{key}' must not be empty")
        texts = [str(x) for x in t]
    else:
        _need(isinstance(t, list), f"Language key '{key}' must be a list. Got {type(t)}")
        texts = []
        for item in t:
            _need(isinstance(item, list), f"Language batch item must be a list. Got {type(item)}")
            _need(len(item) == 1, f"Language batch item must have exactly one item. Got {len(item)}")
            _need(isinstance(item[0], str), f"Language batch item must be a string. Got {type(item[0])}")
            texts.append(item[0])
    batch("Language", key, len(texts))
    order = sorted(emb.state, key=lambda k: emb.state[k][0])
    rows = [Observation({k: videos[k][i, 0] for k in emb.video},
                        np.concatenate([states[k][i, 0] for k in order]).astype(np.float32), texts[i]) for i in range(size[0])]
    return rows, flat


def to_gr00t(actions, emb, flat):
    """float32 [B][horizon][D] in robot units -> the Isaac-GR00T action dict (keys prefixed with action. in the flat form)."""
    a = np.asarray(actions, np.float32)
    _need(a.ndim == 3 and a.shape[2] >= emb.action_dim, f"actions must be [B][horizon][{emb.action_dim}], got {a.shape}")
    return {(f"action.{k}" if flat else k): np.ascontiguousarray(a[:, :, s:e]) for k, (s, e) in emb.action.items()}


def gr00t_modality_config(emb):
    """What Isaac-GR00T's get_modality_config returns for this embodiment, in its wire form."""
    def cfg(deltas, keys):
        return {"__ModalityConfig__": True, "as_json": {"delta_indices": list(deltas), "modality_keys": list(keys), "sin_cos_embedding_keys": None,
                                                        "mean_std_embedding_keys": None, "action_configs": None}}
    return {"video": cfg([0], emb.video), "state": cfg([0], sorted(emb.state, key=lambda k: emb.state[k][0])),
            "action": cfg(range(emb.action_horizon), sorted(emb.action, key=lambda k: emb.action[k][0])), "language": cfg([0], [emb.language])}


# ---- LeRobot ----
def from_lerobot(batch, cameras, state_key="observation.state", task_key="task"):
    """A LeRobot observation batch -> [Observation] * B. Images: float [B][3][H][W] in 0..1 or uint8 [B][H][W][3]; torch or numpy."""
    def arr(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    imgs = {}
    for cam in cameras:
        key = f"observation.images.{cam}"
        _need(key in batch, f"'{key}' must be in the batch")
        v = arr(batch[key])
        if v.ndim == 3:
            v = v[None]
        _need(v.ndim == 4, f"'{key}' must be [B][3][H][W] or [B][H][W][3], got {v.shape}")
        if v.dtype != np.uint8:
            _need(v.shape[1] == 3, f"'{key}' as floats must be [B][3][H][W], got {v.shape}")
            v = np.clip(np.rint(v.astype(np.float32).transpose(0, 2, 3, 1) * 255.0), 0, 255).astype(np.uint8)
        else:
            _need(v.shape[-1] == 3, f"'{key}' as bytes must be [B][H][W][3], got {v.shape}")
        imgs[cam] = v
    _need(state_key in batch, f"'{state_key}' must be in the batch")
    st = arr(batch[state_key]).astype(np.float32)
    st = st[None] if st.ndim == 1 else st.reshape(st.shape[0], -1)
    n = st.shape[0]
    task = batch.get(task_key, "")
    task = [task] * n if isinstance(task, str) else [str(t) for t in task]
    _need(len(task) == n and all(v.shape[0] == n for v in imgs.values()), "images, state and task must have one batch size")
    return [Observation({c: imgs[c][i] for c in cameras}, st[i], task[i]) for i in range(n)]
