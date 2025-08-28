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
import torch
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import numpy as np
from trl.models.utils import unwrap_model_for_generation
import transformers

def accuracy_reward(prompts, completions: list[str], emb_tokenizer, emb_model, **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    print(emb_model.device)
    # kwargs_list = [
    #     {key: kwargs[key][i] for key in kwargs}
    #     for i in range(batch_size)
    # ]
    # rewards = [
    #     acc_func(completion, **kwargs) if completion is not None else 0.0
    #     for completion, kwargs in zip(completions, kwargs_list)
    # ]
    return [None, None, None, None]

def get_cluster_reward_func(emb_tokenizer, emb_model):
    def cluster_score(prompts, completions: list[str], accelerator, **kwargs) -> list[Optional[float]]:
        # emb_tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        #     padding_side="left",
        # )
        # local_emb_model = transformers.AutoModel.from_pretrained(
        #     "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        #     torch_dtype=torch.float32
        # )
        device = accelerator.device
        local_emb_model = emb_model.to(device)
        local_emb_model.eval()
        def _extract_pred(txt):
            pattern = r'boxed\{([^}]*)\}'
            results = re.findall(pattern, txt)
            if results:
                ret = results[0]
                # ret = ret.replace("\"", "")
                # ret = ret.replace("\'", "")
                ret = ret.strip()
                return ret
            else:
                return None
        def _is_valid_answer(txt):
            if txt is None or txt == "None":
                return False
            if not txt.isascii():
                return False
            return True
        extracted_candidates = [_extract_pred(comp) for comp in completions]
        valid_index = [i for i in range(len(extracted_candidates)) if _is_valid_answer(extracted_candidates[i])]
        inputs = [cand for cand in extracted_candidates if _is_valid_answer(cand)]
        if len(valid_index) <= 4:
            return [None] * len(extracted_candidates)
        encodings = emb_tokenizer(
            inputs,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=512,).to(device)
        with torch.no_grad():
            output = local_emb_model(**encodings)
            emb = torch.nn.functional.normalize(output.last_hidden_state[:, 0, :], p=2, dim=1)
            emb = emb.detach().cpu().double().numpy()
        sim_matrix = cosine_similarity(emb)
        clusteror = hdbscan.HDBSCAN(
            metric='precomputed', 
            min_cluster_size = 5, 
            min_samples = 2, 
            provide_probabilities = True, 
            allow_single_cluster = True,
            # prediction_data = True
        )
        labels = clusteror.fit_predict(sim_matrix)
        clusters = dict()
        for label_idx, c in enumerate(labels):
            if c < 0:
                continue
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(label_idx)
        max_size = max([len(v) for (k, v) in clusters.items()])
        max_c_name = -1
        for k in clusters.keys():
            if len(clusters[k]) == max_size: 
                max_c_name = k
        main_cluster = clusters[max_c_name]
        if len(main_cluster) < 5:
            return [None] * len(extracted_candidates)
        featured_cluster = [emb[idx] for idx in main_cluster]
        center_feature = np.stack(featured_cluster).mean(axis=0) / np.linalg.norm(np.stack(featured_cluster).mean(axis=0))
        cos_sim_cluster = [np.dot(center_feature, f)/ (np.linalg.norm(center_feature) * np.linalg.norm(f)) for f in featured_cluster]
        record_cosine_score = [None] * len(extracted_candidates)
        for idx_v, cosine in zip(main_cluster, cos_sim_cluster):
            original_idx = valid_index[idx_v]
            record_cosine_score[original_idx] = cosine
        for i, v in enumerate(record_cosine_score):
            record_cosine_score[i] = 0.0 if v is None else float(0.5 + 0.5 * v)
        return record_cosine_score
    return cluster_score

def entropy_reward(prompts, completions: list[str], **kwargs) -> list[Optional[float]]:
    def _extract_pred(txt):
        pattern = r'boxed\{([^}]*)\}'
        results = re.findall(pattern, txt)
        if results:
            ret = results[0]
            # ret = ret.replace("\"", "")
            # ret = ret.replace("\'", "")
            ret = ret.strip()
            return ret
        else:
            return None
    def _is_valid_answer(txt):
        if txt is None or txt == "None":
            return False
        if not txt.isascii():
            return False
        return True
    extracted_candidates = [_extract_pred(comp) for comp in completions]
    reward = [1 if _is_valid_answer(cand) else None for cand in extracted_candidates]
    return reward


def get_reward_funcs(script_args) -> list[Callable]:
    emb_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        padding_side="left",
    )
    emb_model = transformers.AutoModel.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        torch_dtype=torch.float32
    )
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "cluster_score": get_cluster_reward_func(emb_tokenizer=emb_tokenizer, emb_model=emb_model),
        "entropy": entropy_reward
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
