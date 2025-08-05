from dataclasses import dataclass, field
import transformers
import torch
import random
import numpy as np
import jsonlines
from tqdm import tqdm
import os
from accelerate import Accelerator
from accelerate.utils import set_seed, gather_object
import tqdm
import torch.nn.functional as F

'''
This script is to compute losses of LLM on generated data or valid data
'''

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="",
        metadata={"help": "The model name or path to use for generation"}
    )
    tokenizer_path: str = field(
        default="",
        metadata={"help": "The tokenizer path to use for generation"}
    )
    mode: str = field(
        default="chat",
        metadata={"help": "The mode of the model, either 'chat' or 'base'"}
    )
    candidates_path: str = field(
        default="",
        metadata={"help": "The name of the dataset to use for generation"}
    )
    scored_path: str = field(
        default="",
        metadata={"help": "The path to store entropy-scored candidates"}
    )
    dpo_path: str = field(
        default="",
        metadata={"help": "The path to store entropy-scored dpo candidates"}
    )
    few_shot_cot: bool = field(
        default=False,
        metadata={"help": "Whether to use few-shot chain of thought (CoT) examples in the dataset"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "The batch size for processing inputs"}
    )
    max_model_len: int = field(
        default=2048,
        metadata={"help": "The maximum length of the model input (default: 2048)"}
    )
    # seed: int = field(
    #     default=42,
    #     metadata={"help": "The random seed for reproducibility (default: 42)"}
    # )

@dataclass
class DataArguments:
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_lora: bool = False

@dataclass
class LoraArguments:
    q_lora: bool = False

def make_contrast_pair(obj):
    prompt = obj["prompt"]
    outputs = obj["outputs"]
    scores = obj["scores"]
    max_score = max(scores)
    chosen_id = scores.index(max_score)
    chosen = outputs[chosen_id]
    rej_score = min(scores)
    rej_id = scores.index(rej_score)
    rejected = outputs[rej_id]
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

def record_loss():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    seed = int(training_args.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    #Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    accelerator = Accelerator()
    device = accelerator.device

    model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path if (model_args.tokenizer_path is not None and len(model_args.tokenizer_path) > 0) else model_args.model_name_or_path,
        trust_remote_code = True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    qa_pairs = []
    with jsonlines.open(model_args.candidates_path, "r") as f:
        for obj in f:
            prompt = obj["prompt"]
            outputs = obj["outputs"]
            for output in outputs:
                qa_pairs.append({
                    "prompt": prompt,
                    "output": output
                })

    inputs = [qa["prompt"] + qa["output"] for qa in qa_pairs]

    def prepare_prompts(prompts, tokenizer, batch_size=16):
        batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok=[]
        tokenizer.padding_side="left"     
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=True, 
                    max_length=model_args.max_model_len,
                    add_special_tokens=True).to(device) 
                )
        return batches_tok
    
    propmt_length = len(inputs)
    model.eval()
    model.to(device)
    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(inputs) as prompts:
        losses=[]
        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=model_args.batch_size)
        # prompt_batches = prepare_encodings(encodings, batch_size=training_args.per_device_eval_batch_size)
        pbar = tqdm.tqdm(total=len(prompt_batches), disable=(not accelerator.is_local_main_process))

        for prompts_tokenized in prompt_batches:
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = model(**prompts_tokenized)
                logits = outputs.get("logits")
                masks = prompts_tokenized["attention_mask"]
                labels = prompts_tokenized["input_ids"]
                batch_size = logits.shape[0]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
                loss = loss.view(batch_size, -1)
                masked_loss = loss * masks[:, 1:].float()
                masked_loss = torch.sum(masked_loss, dim = -1)

            losses.extend(masked_loss.float().detach().cpu().numpy().tolist())
            if accelerator.is_local_main_process:
                pbar.update(1)

        losses=[ losses ]
    losses_gathered=gather_object(losses)

    if accelerator.is_main_process:
        total_results = []
        for r in losses_gathered:
            total_results += r
        scores = [- float(d) for d in total_results]
        all_objs = []
        with jsonlines.open(model_args.candidates_path, "r") as fr, jsonlines.open(model_args.scored_path, "w") as fw:
            score_indx = 0
            for obj in fr:
                prompt = obj["prompt"]
                outputs = obj["outputs"]
                batch_scores = scores[score_indx:score_indx + len(outputs)]
                score_indx += len(outputs)
                obj = {
                    "prompt": prompt,
                    "outputs": outputs,
                    "scores": batch_scores,
                }
                all_objs.append(obj)
                fw.write(obj)
        with jsonlines.open(model_args.dpo_path, "w") as fw:
            for obj in all_objs:
                pair = make_contrast_pair(obj)
                if pair is not None:
                    fw.write(pair)
        

if __name__ == "__main__":
    record_loss()
    

