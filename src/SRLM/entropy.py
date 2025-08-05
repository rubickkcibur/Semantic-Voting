import torch
import jsonlines
import os
import random
import numpy as np
import transformers
import re
from tqdm import tqdm
import torch.nn.functional as F
    
def compute_entropy(model, tokenizer, batch_inputs):
    inputs = tokenizer(batch_inputs, return_tensors='pt', padding="longest", truncation=True)
    inputs = inputs.to("cuda:0")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.get("logits")
        masks = inputs["attention_mask"]
        labels = inputs["input_ids"]
        batch_size = logits.shape[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        loss = loss.view(batch_size, -1)
        masked_loss = loss * masks[:, :-1].float()
        masked_loss = torch.sum(masked_loss, dim = -1)
    return masked_loss.float().detach().cpu().numpy().tolist()
    
def eval(args):
    # data preparation
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path if (args.tokenizer_path is not None and len(args.tokenizer_path) > 0) else args.model_name_or_path,
        trust_remote_code = True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code = True,
    )
    model.eval()
    model.to("cuda:0")

    qa_pairs = []
    with jsonlines.open(args.candidates_path, "r") as f:
        for obj in f:
            prompt = obj["prompt"]
            outputs = obj["outputs"]
            for output in outputs:
                qa_pairs.append({
                    "prompt": prompt,
                    "output": output
                })

    inputs = [qa["prompt"] + qa["output"] for qa in qa_pairs]
    traj_entropies = []
    with tqdm(total=len(inputs), desc="computing qa entropy") as pbar:
        for i in range(0, len(inputs), args.batch_size):
            batch_inputs = inputs[i:i + args.batch_size]
            losses = compute_entropy(model, tokenizer, batch_inputs)
            traj_entropies.extend(losses)
            pbar.update(len(batch_inputs))
    
   
    scores = [- p for p in traj_entropies]  # Convert losses to scores
    all_objs = []
    with jsonlines.open(args.candidates_path, "r") as fr, jsonlines.open(args.scored_path, "w") as fw:
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
    with jsonlines.open(args.dpo_path, "w") as fw:
        for obj in all_objs:
            pair = make_contrast_pair(obj)
            if pair is not None:
                fw.write(pair)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The arguments for entropy-based score computation")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The model name or path to use for generation",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="The tokenizer path to use for generation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        help="The mode of the model, either 'chat' or 'base'",
    )
    parser.add_argument(
        "--candidates_path",
        type=str,
        default=None,
        help="The name of the dataset to use for generation",
    )
    parser.add_argument(
        "--scored_path",
        type=str,
        default=None,
        help="The path to store self-scored candidates",
    )
    parser.add_argument(
        "--dpo_path",
        type=str,
        default=None,
        help="The path to store self-scored dpo candidates",
    )
    parser.add_argument(
        "--few_shot_cot",
        type=bool,
        default=False,
        help="Whether to use few-shot chain of thought (CoT) examples in the dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The batch size for processing inputs",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="The maximum length of the model input (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    seed = int(args.seed)
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

    eval(args)
        