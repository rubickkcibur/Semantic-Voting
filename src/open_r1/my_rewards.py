# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Literal, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.code_providers import get_provider
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask
from data_processor.processor_registers import get_metrics


def accuracy_reward(completions: list[str], acc_func: Callable, **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    batch_size = len(completions)
    kwargs_list = [
        {key: kwargs[key][i] for key in kwargs}
        for i in range(batch_size)
    ]
    rewards = [
        acc_func(completion, **kwargs) if completion is not None else 0.0
        for completion, kwargs in zip(completions, kwargs_list)
    ]
    return rewards

def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": update_wrapper(
            partial(
                accuracy_reward,
                acc_func=get_metrics(script_args.dataset_name)
            ),
            accuracy_reward
        )
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
