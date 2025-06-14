from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, gather_object
import logging
import tqdm
from data_processor.processor_registers import load_custom_dataset, get_metrics
# from peft import PeftModel

'''
This script is to evaluate the LLM's performance on test dataset
'''

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
device = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the tokenizer. If not provided, will use the model's tokenizer."
        },
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None
    )
    few_shot_cot: bool = field(
        default=False,
        metadata={
            "help": "Whether to use few-shot chain-of-thought (CoT) examples in the dataset."
        },
    )
    mode: str = field(default="chat")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=800,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_new_tokens: int = field(
        default=512,
        metadata={
            "help": "Maximum number of new tokens to generate in the output."
        },
    )
    


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def evaluation_main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # output_dir = "modelL-filter_strategy_{}-time_{}".format(data_args.data_filter_mode, int(time.time()))
    # training_args.output_dir = os.path.join(training_args.output_dir, output_dir)
    # os.makedirs(training_args.output_dir, exist_ok=True)
    # ROLE_CONTENT = "You are a calculation assistant. You will be given an arithmetic question. Please think step by step and give the answer. After giving your thoughts, use 'The answer is:' followed by the answer."
    accelerator = Accelerator()
    device = accelerator.device

    logger.info('Loading causal model...')
    modelL = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    # if len(model_args.peft_model_path) > 0:
    #     logger.info("loading peft weights from{}".format(model_args.peft_model_path))
    #     modelL = PeftModel.from_pretrained(modelL, model_args.peft_model_path)
    #     modelL.merge_and_unload()
    tokenizerL = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path if (model_args.tokenizer_path is not None and len(model_args.tokenizer_path)>0) else model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        padding_side="left")
    tokenizerL.pad_token_id = tokenizerL.eos_token_id

    # terminators = [
    #     tokenizerL.eos_token_id,
    #     tokenizerL.convert_tokens_to_ids("<|eot_id|>")
    # ] if model_args.mode == "chat" else tokenizerL.eos_token_id
    terminators = tokenizerL.eos_token_id

    test_dataset = load_custom_dataset(
        data_args.dataset_name,
        tokenizer = tokenizerL,
        cot = data_args.few_shot_cot,
        apply_chat_template = (data_args.mode == "chat"),
    )
    prompts = [test_dataset["test"][i]["prompt"] for i in range(len(test_dataset["test"]))]
    metric_func = get_metrics(data_args.dataset_name)


    def prepare_prompts(prompts, tokenizer, batch_size=16):
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
        batches_tok = []
        tokenizer.padding_side = "left"
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=training_args.model_max_length,
                    add_special_tokens=True).to(device)
            )
        return batches_tok

    modelL.eval()
    modelL.to(device)
    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(prompts) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = prepare_prompts(prompts, tokenizerL, batch_size=training_args.per_device_eval_batch_size)
        pbar = tqdm.tqdm(total=len(prompt_batches), disable=(not accelerator.is_local_main_process))

        for prompts_tokenized in prompt_batches:
            with torch.no_grad():
                outputs_tokenized = modelL.generate(
                    **prompts_tokenized,
                    max_new_tokens=training_args.max_new_tokens,
                    eos_token_id=terminators,
                    # num_return_sequences=1,
                    # temperature=0,
                    pad_token_id=tokenizerL.eos_token_id,
                    do_sample=False
                )

            # remove prompt from gen. tokens
            outputs_tokenized = [tok_out[len(tok_in):]
                                 for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]

            # count and decode gen. tokens
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = tokenizerL.batch_decode(outputs_tokenized)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
            if accelerator.is_local_main_process:
                pbar.update(1)
            torch.cuda.empty_cache()
        results = [results]  # transform to list, otherwise gather_object() will not collect correctly
    results_gathered = gather_object(results)
    if accelerator.is_main_process:
        total_results = []
        for r in results_gathered:
            total_results += r["outputs"]
        total_results = [txt.split(tokenizerL.eos_token)[0] for txt in total_results]

        rewards = []
        length = len(total_results)
        for i in range(length):
            data_params = test_dataset["test"][i].copy()
            data_params.pop("prompt")
            r = metric_func(total_results[i], **data_params)
            rewards.append(r)
        avg_reward = sum(rewards) / length if length > 0 else 0.0
        # acc = METRIC[data_args.dataset_name](total_results, answers, tokenizerL) if "wmt" in data_args.dataset_name else METRIC[data_args.dataset_name](total_results, answers)
        logger.info(f"acc is {avg_reward}")
        # dump results
        dump_path = training_args.output_dir if (training_args.output_dir is not None and len(training_args.output_dir) > 0) else model_args.model_name_or_path
        with open(os.path.join(dump_path, "debug_{}.txt".format(data_args.dataset_name)), "w",
                  encoding="utf8") as f:
            for i in range(length):
                f.write("Prompt: " + str(test_dataset["test"][i]["prompt"]))
                f.write("\n")
                f.write("Pred: " + str(total_results[i]))
                f.write("\n")
                f.write("-----------------------------")
                f.write("\n")
        with open(os.path.join(dump_path, "acc_{}.txt".format(data_args.dataset_name)), "w",
                  encoding="utf8") as f:
            f.write(str(avg_reward) + "\n")


if __name__ == "__main__":
    # 注意seed，原设置是没有do_sample的
    seed = os.environ.get("SEED", 114514)
    seed = int(seed)
    print("================set global random seed to {}================".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    evaluation_main()


