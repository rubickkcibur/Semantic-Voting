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

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.my_rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from data_processor.processor_registers import load_custom_dataset
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from functools import partial, update_wrapper
import torch

logger = logging.getLogger("MainLogger")

class MyGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, emb_tokenizer=None, emb_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.emb_tokenizer = emb_tokenizer
        # self.emb_model = emb_model.to(self.accelerator.device)
        # self.emb_model.eval()
        for i, reward_func in enumerate(self.reward_funcs):
            self.reward_funcs[i] = update_wrapper(
                    partial(
                        reward_func,
                        # emb_tokenizer=self.emb_tokenizer,
                        # emb_model=self.emb_model,
                        accelerator=self.accelerator,
                    ),
                    reward_func
                )
    def _compute_loss(self, model, inputs):
        # processing nan
        def nanmin(tensor: torch.Tensor) -> torch.Tensor:
            if torch.isnan(tensor).all():
                return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
            return torch.min(tensor[~torch.isnan(tensor)])
        def nanmax(tensor: torch.Tensor) -> torch.Tensor:
            if torch.isnan(tensor).all():
                return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
            return torch.max(tensor[~torch.isnan(tensor)])
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        min_advs = advantages.min()
        advantage_mask = torch.isclose(advantages, min_advs, atol=1e-6)
        valid_element = advantages[~advantage_mask]
        if valid_element.numel() > 0:
            valid_mean = valid_element.mean()
            valid_std = valid_element.std()
        else:
            valid_mean = torch.tensor(0.0, device=valid_element.device)
            valid_std = torch.tensor(1.0, device=valid_element.device)
            advantage_mask = torch.zeros_like(advantages, dtype=torch.bool)
        advantages = (advantages - valid_mean) / (valid_std + 1e-4)
        advantages = advantages.detach()

        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            valid_per_token_loss = per_token_loss[~advantage_mask, :]
            valid_completion_mask = completion_mask[~advantage_mask, :]
            # print(f"valid_per_token_loss: {valid_per_token_loss.shape}, valid_completion_mask: {valid_completion_mask.shape}")
            valid_loss = (valid_per_token_loss * valid_completion_mask).sum() / valid_completion_mask.sum().clamp(min=1.0)
            loss = valid_loss
            # loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    # dataset = get_dataset(script_args)
    

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # # Format into conversation
    # def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
    #     prompt = []

    #     if training_args.system_prompt is not None:
    #         prompt.append({"role": "system", "content": training_args.system_prompt})

    #     if prompt_column not in example:
    #         raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

    #     prompt.append({"role": "user", "content": example[prompt_column]})
    #     return {"prompt": prompt}

    # dataset = dataset.map(make_conversation)

    # for split in dataset:
    #     if "messages" in dataset[split].column_names:
    #         dataset[split] = dataset[split].remove_columns("messages")
    dataset = load_custom_dataset(
        script_args.dataset_name,
        tokenizer=tokenizer,
        cot=False,
        apply_chat_template=True,
    )
    #############################
    # Initialize the GRPO trainer
    #############################
    # emb_tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
    #     padding_side="left",
    # )
    # emb_model = transformers.AutoModel.from_pretrained(
    #     "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
    #     torch_dtype=torch.float32
    # )
    trainer = MyGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        # emb_tokenizer=emb_tokenizer,
        # emb_model=emb_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=(dataset["test"] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
