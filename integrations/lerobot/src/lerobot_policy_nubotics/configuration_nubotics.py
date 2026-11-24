# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""The "nubotics" policy type: the Nubotics model folder, answered by a NukeTorch server on this Mac."""
from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig


@PreTrainedConfig.register_subclass("nubotics")
@dataclass
class NuboticsConfig(PreTrainedConfig):
    model_dir: str | None = None          # the Nubotics model folder; empty: ~/NukeTorch/models/Nubotics
    server_url: str | None = None         # the NukeTorch server; empty: the one `nubotics serve` starts
    engine_seed: int = 0                  # each call's seed is drawn from this and the call's number
    n_action_steps: int | None = None     # actions executed per policy call; empty: the checkpoint's own (16 for LIBERO-10)
    # LeRobot image key (after observation.images.) -> the checkpoint's camera. LeRobot's LIBERO environment calls the wrist camera image2.
    cameras: dict[str, str] = field(default_factory=lambda: {"image": "image", "image2": "wrist_image"})
    device: str | None = "cpu"            # LeRobot's tensors stay on the CPU; the policy runs in the NukeTorch server

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("The nubotics policy is for inference only.")

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        return None
