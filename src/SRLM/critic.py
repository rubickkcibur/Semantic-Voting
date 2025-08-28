from vllm import LLM, SamplingParams
import torch
import ray
import jsonlines
import os
import random
import numpy as np
import transformers
from data_processor.processor_registers import load_custom_dataset
import re
import time

ray.init()

@ray.remote(num_gpus = 1)
class VLLMActor:
    def __init__(self, args):
        self.llm = LLM(
            model = args.model_name_or_path,
            tokenizer = args.tokenizer_path if (args.tokenizer_path is not None and len(args.tokenizer_path) > 0) else args.model_name_or_path,
            max_model_len = args.max_model_len,
            trust_remote_code = True,
            dtype = "bfloat16",
            seed = args.seed,
            gpu_memory_utilization = 0.9
        )
    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params)
    
def split_data(data, num_workers):
    chunk_size = len(data) // num_workers
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    
def start_generation(args, n_workers = 8):
    # data preparation
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path if (args.tokenizer_path is not None and len(args.tokenizer_path) > 0) else args.model_name_or_path,
        trust_remote_code = True,
    )
    tokenizer.padding_side = "left"

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

    pattern = "Review the user's prompts and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:\n- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.\n- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.\n- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.\n- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.\n- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.\n<User_prompt>\n{}\n</User_prompt>\n<response>\n{}\n</response>\nAfter examining the user's prompts and the response:\n- Briefly justify your total score, up to 100 words.\n- Conclude with the score using the format: \"Score: <total points>\" (e.g. \"Score: 4\") \nRemember to assess from the AI Assistant perspective, utilizing your self knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria."
    # pattern = "Review a user's prompts and the corresponding response using the 5-point scoring system. Scoring 1 represents the worst and scoring 5 means the best.\nHere is the prompt-response pair:\n<User_prompt>\n{}\n</User_prompt>\n<response>\n{}\n</response>\nConclude with the score using the format: \"Score: <total points>\" (e.g. \"Score: 4\") \nRemember to assess from the AI Assistant perspective, utilizing your self knowledge as necessary."
    inputs = [pattern.format(qa["prompt"], qa["output"]) for qa in qa_pairs]
    inputs = [[dict(role="user", content=inp)] for inp in inputs]
    inputs = [tokenizer.apply_chat_template(inp, tokenize=False) for inp in inputs]
    
    data_chunks = split_data(inputs, n_workers)

    #generate
    sampling_params = SamplingParams(
        n = 1,
        temperature = 0,
        # top_p = 0.9,
        max_tokens = args.max_new_tokens, 
        seed = args.seed,
    )
    actors = [VLLMActor.remote(args) for _ in range(n_workers)]
    futures = [actor.generate.remote(chunk, sampling_params) 
            for actor, chunk in zip(actors, data_chunks)]
    results = ray.get(futures)
    final_results = []
    for result in results:
        final_results.extend(result)

    def extract_score(output_text):
        pattern = r"Score: ([0-5])"
        matches = re.findall(pattern, output_text)
        if matches:
            score = matches[0].strip()
            return int(score) if score.isdigit() else None
        
    scores = [extract_score(result.outputs[0].text) for result in final_results]
    debug_results = [result.outputs[0].text for result in final_results]
    scores = [-1 if score is None else score for score in scores]  # Replace None with 0
    all_objs = []
    with jsonlines.open(args.candidates_path, "r") as fr, jsonlines.open(args.scored_path, "w") as fw:
        score_indx = 0
        for obj in fr:
            prompt = obj["prompt"]
            outputs = obj["outputs"]
            batch_scores = scores[score_indx:score_indx + len(outputs)]
            batch_debug_results = debug_results[score_indx:score_indx + len(outputs)]
            score_indx += len(outputs)
            obj = {
                "prompt": prompt,
                "outputs": outputs,
                "scores": batch_scores,
                "debug_results": batch_debug_results
            }
            all_objs.append(obj)
            fw.write(obj)

    def make_contrast_pair(obj):
        def extract_pred(txt):
            pattern = r'boxed\{([^}]*)\}'
            results = re.findall(pattern, txt)
            if results:
                ret = results[0]
                ret = ret.strip()
                return ret
            else:
                return None
        
        prompt = obj["prompt"]
        outputs = obj["outputs"]
        scores = obj["scores"]
        invalid_index = [i for i, output in enumerate(outputs) if extract_pred(output) is None]
        invalid_index = set(invalid_index)
        if args.use_format_filter:
            scores_for_max = [-1e6 if i in invalid_index else s for i, s in enumerate(scores)]
            max_score = max(scores_for_max)
            if max_score < 0:
                return None
            chosen_ids = [i for i, s in enumerate(scores_for_max) if s == max_score]
            chosen_id = random.choice(chosen_ids)
            # chosen_id = chosen_ids[-1]
            chosen = outputs[chosen_id]

            scores_for_min = [1e6 if i in invalid_index else s for i, s in enumerate(scores)]
            min_score = min(scores_for_min)
            rej_ids = [i for i, s in enumerate(scores_for_min) if s == min_score]
            rej_id = random.choice(rej_ids)
            # rej_id = rej_ids[-1]
            rejected = outputs[rej_id]
        else:
            max_score = max(scores)
            if max_score < 0:
                return None
            chosen_ids = [i for i, score in enumerate(scores) if score == max_score]
            chosen_id = random.choice(chosen_ids)
            chosen = outputs[chosen_id]

            rej_score = min([1e6 if s < 0 else s for s in scores])
            rej_ids = [i for i, score in enumerate(scores) if score == rej_score]
            rej_id = random.choice(rej_ids)
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
    parser = argparse.ArgumentParser(description="The arguments for generation with vLLM on a HuggingFace dataset")
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
        "--return_sequences",
        type=int,
        default=16,
        help="The number of sequences to return for each input (default: 16)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="The maximum length of the model input (default: 2048)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="The maximum number of new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed for reproducibility (default: 114514)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="The number of workers to use for data loading (default: 8)",
    )
    parser.add_argument(
        "--use_format_filter",
        type=bool,
        default=False,
        help="Whether to use format filter (default: False)",
    )

    args = parser.parse_args()

    # seed = int(args.seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.cuda.manual_seed(seed)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.use_deterministic_algorithms(True)
    # #Enable CUDNN deterministic mode
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    random.seed(time.time())

    start_generation(args, n_workers = args.n_workers)
