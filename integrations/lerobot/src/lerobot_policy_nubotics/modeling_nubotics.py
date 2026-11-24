# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""NuboticsPolicy: LeRobot's policy interface over nubotics.Policy. LeRobot's tensors in, LeRobot's tensors out; no model weights in
this process. One call fills the action queue with n_action_steps actions for every environment in the batch; select_action hands
them out one step at a time."""
from collections import deque

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from nubotics import Policy
from nubotics.formats import from_lerobot

from .configuration_nubotics import NuboticsConfig


class NuboticsPolicy(PreTrainedPolicy):
    config_class = NuboticsConfig
    name = "nubotics"

    def __init__(self, config: NuboticsConfig, **kwargs):
        super().__init__(config)
        self.engine = Policy(config.model_dir, config.server_url, seed=config.engine_seed)
        missing = sorted(set(self.engine.cameras) - set(config.cameras.values()))
        if missing:
            raise ValueError(f"cameras must map LeRobot image keys onto the checkpoint's cameras {self.engine.cameras}; missing {missing}")
        self.steps = int(config.n_action_steps or self.engine.n_action_steps)
        self.reset()

    def get_optim_params(self) -> dict:
        return {}

    def reset(self):
        self._queue = deque([], maxlen=self.steps)

    def forward(self, batch: dict[str, Tensor]):
        raise NotImplementedError("The nubotics policy is for inference only; use select_action or predict_action_chunk.")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """[B][n_action_steps][action_dim] in robot units (the checkpoint's gripper step applied, as LeRobot's postprocessor would)."""
        rows = from_lerobot(batch, list(self.config.cameras))
        rows = [r._replace(images={self.config.cameras[k]: v for k, v in r.images.items()}) for r in rows]
        return torch.from_numpy(np.ascontiguousarray(self.engine.act_many(rows, steps=self.steps, env_gripper=True)))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if len(self._queue) == 0:
            self._queue.extend(self.predict_action_chunk(batch).transpose(0, 1))
        return self._queue.popleft()
