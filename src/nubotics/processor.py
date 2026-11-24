# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""Host-side processing for GR00T N1.7 LeRobot-format checkpoints, following the checkpoint's own processor files
(policy_preprocessor.json, policy_postprocessor.json and their stats).

Observation: camera frames -> resized and cropped 8-bit frames, task -> prompt token ids, state -> normalized model state.
Actions: the model's normalized actions -> robot actions.
numpy, OpenCV and tokenizers only.
"""
import json
import pathlib

import cv2
import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

IMAGE_PAD, VISION_START, VISION_END = "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"


class Processor:
    def __init__(self, model_dir, cameras=None, tokenizer_dir=None):
        ck = pathlib.Path(model_dir).expanduser()
        cfg = json.loads((ck / "config.json").read_text())
        # Camera order: the checkpoint's visual inputs by name (image, wrist_image for LIBERO-10).
        self.cameras = list(cameras) if cameras else sorted(k.split(".")[-1] for k, v in cfg["input_features"].items() if v["type"] == "VISUAL")
        self.state_dim = int(cfg["input_features"]["observation.state"]["shape"][0])
        self.action_dim = int(cfg["output_features"]["action"]["shape"][0])
        self.n_action_steps = int(cfg["n_action_steps"])
        self.embodiment_tag = cfg.get("embodiment_tag")

        pre = json.loads((ck / "policy_preprocessor.json").read_text())
        post = json.loads((ck / "policy_postprocessor.json").read_text())
        steps = {s["registry_name"]: s for s in pre["steps"]}
        pack, vlm = steps["groot_n1_7_pack_inputs_v1"], steps["groot_n1_7_vlm_encode_v1"]
        self.pack, self.vlm = pack["config"], vlm["config"]
        ps = load_file(str(ck / pack["state_file"]))
        self.state_min = ps["observation.state.min"].astype(np.float32)
        self.state_max = ps["observation.state.max"].astype(np.float32)
        self.state_width = int(self.pack["max_state_dim"])
        dec = post["steps"][0]
        self.dec = dec["config"]
        ds = load_file(str(ck / dec["state_file"]))
        self.action_min = ds["action.min"].astype(np.float32)
        self.action_max = ds["action.max"].astype(np.float32)
        self.embodiment = int(self.pack["embodiment_mapping"][self.pack["embodiment_tag"]])

        tk = pathlib.Path(tokenizer_dir).expanduser() if tokenizer_dir else ck / "tokenizer"
        self.tok = Tokenizer.from_file(str(tk / "tokenizer.json"))
        ip = json.loads((tk / "preprocessor_config.json").read_text())
        self.patch, self.merge = int(ip["patch_size"]), int(ip["merge_size"])
        self.mean = [float(x) for x in ip["image_mean"]]
        self.std = [float(x) for x in ip["image_std"]]
        self._prompts = {}

    def image(self, frame):
        """The checkpoint's eval transform: shortest edge, center crop, shortest edge (OpenCV, INTER_AREA)."""
        c = self.vlm
        edge = c["shortest_image_edge"] or c["image_target_size"][0]

        def shortest(f):
            h, w = f.shape[:2]
            s = min(h, w)
            if s == edge:
                return f
            k = edge / float(s)
            return cv2.resize(f, (max(1, int(round(w * k))), max(1, int(round(h * k)))), interpolation=cv2.INTER_AREA)

        f = shortest(np.ascontiguousarray(frame))
        frac = c["crop_fraction"]
        if frac is not None and 0 < frac < 1:
            h, w = f.shape[:2]
            ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
            top, left = max(0, (h - ch) // 2), max(0, (w - cw) // 2)
            f = f[top:top + ch, left:left + cw]
        return shortest(f)

    def prompt(self, task, grids):
        text = task if task else "Perform the task."
        if self.pack["formalize_language"]:
            text = "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace() or ch == "_")
        key = (text, tuple(grids))
        if key not in self._prompts:
            vision = "".join(VISION_START + IMAGE_PAD * (g[1] * g[2] // (self.merge * self.merge)) + VISION_END for g in grids)
            s = "<|im_start|>user\n" + vision + text + "<|im_end|>\n"
            self._prompts[key] = np.array(self.tok.encode(s, add_special_tokens=False).ids, np.uint32)
        return self._prompts[key]

    def state(self, raw):
        raw = np.asarray(raw, np.float32).ravel()
        if raw.size != self.state_dim:
            raise ValueError(f"state must have {self.state_dim} values, got {raw.size}")
        d = self.state_max - self.state_min
        safe = np.where(d != 0, d, 1)
        x = np.where(d != 0, 2 * (raw - self.state_min) / safe - 1, 0).astype(np.float32)
        if self.pack["clip_outliers"]:
            x = np.clip(x, -1, 1)
        out = np.zeros(self.state_width, np.float32)
        out[:len(x)] = x
        return out

    def observation(self, images, state, task):
        """-> (token_ids uint32 [n], frames uint8 [cameras][h][w][3], grids uint32 [cameras][3], state float32 [state_width])."""
        for c in self.cameras:
            f = images.get(c)
            if not (isinstance(f, np.ndarray) and f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 3):
                raise ValueError(f"image '{c}' must be uint8 [H][W][3]")
        frames = [self.image(images[c]) for c in self.cameras]
        unit = self.patch * self.merge
        for f in frames:
            if f.shape != frames[0].shape or f.shape[0] % unit or f.shape[1] % unit:
                raise ValueError(f"prepared frames must share one size, a multiple of {unit}: got {[x.shape for x in frames]}")
        grids = [(1, f.shape[0] // self.patch, f.shape[1] // self.patch) for f in frames]
        return self.prompt(task, grids), np.ascontiguousarray(np.stack(frames)), np.array(grids, np.uint32), self.state(state)

    def actions(self, normalized, steps=None, env_gripper=True):
        """Model actions [horizon][width] -> robot actions [steps][action_dim]: clip, inverse min-max, LIBERO gripper.
        env_gripper False keeps the gripper value un-normalized, as NVIDIA's policy server returns it."""
        n = self.dec["env_action_dim"]
        a = np.asarray(normalized, np.float32)[:steps or self.n_action_steps, :n]
        if self.dec["clip_normalized_action"]:
            a = np.clip(a, -1, 1)
        d = self.action_max - self.action_min
        safe = np.where(d != 0, d, 1)
        a = np.where(d != 0, (a + 1) * 0.5 * safe + self.action_min, self.action_min).astype(np.float32)
        if env_gripper and self.dec.get("libero_gripper_action") and n >= 7:
            g = a[..., -1]
            a[..., -1] = -np.sign(2 * g - 1) if self.dec.get("libero_gripper_binarize") else -(2 * g - 1)
        return a
