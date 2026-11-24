# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""The nubotics policy applies the checkpoint's own input and output processing (policy_preprocessor.json /
policy_postprocessor.json) in nubotics.Processor, so LeRobot's pipelines around it pass everything through."""
from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_nubotics_pre_post_processors(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]:
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](steps=[], name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](steps=[], name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
                                                            to_transition=policy_action_to_transition, to_output=transition_to_policy_action),
    )
