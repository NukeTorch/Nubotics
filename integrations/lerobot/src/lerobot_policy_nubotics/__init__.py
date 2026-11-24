# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""LeRobot policy type "nubotics". LeRobot imports policy packages by their name prefix (lerobot_policy_), so
`lerobot-eval --policy.type=nubotics ...` and any LeRobot loop can use it unchanged."""
try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError("lerobot is not installed. Install lerobot to use this policy package.") from e
from .configuration_nubotics import NuboticsConfig
from .modeling_nubotics import NuboticsPolicy
from .processor_nubotics import make_nubotics_pre_post_processors

__all__ = ["NuboticsConfig", "NuboticsPolicy", "make_nubotics_pre_post_processors"]
