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
            gpu_memory_utilization = 0.7
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

    pattern = "Review the user's prompts and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:\n- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.\n- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.\n- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.\n- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.\n- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.\n<User_prompt>{}</User_prompt>\n<response>{}</response>\nAfter examining the user's prompts and the response:\n- Briefly justify your total score, up to 100 words.\n- Conclude with the score using the format: \"Score: <total points>\" (e.g. \"Score: 4\") \nRemember to assess from the AI Assistant perspective, utilizing your self knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria."
    inputs = [pattern.format(qa["prompt"], qa["output"]) for qa in qa_pairs]
    
    data_chunks = split_data(inputs, n_workers)

    #generate
    sampling_params = SamplingParams(
        n = 1,
        temperature = 0,
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
    scores = [0 if score is None else score for score in scores]  # Replace None with 0
    with jsonlines.open(args.candidates_path, "r") as fr, jsonlines.open(args.candidates_path + "_scored", "w") as fw:
        score_indx = 0
        for obj in fr:
            prompt = obj["prompt"]
            outputs = obj["outputs"]
            batch_scores = scores[score_indx:score_indx + len(outputs)]
            score_indx += len(outputs)
            fw.write({
                "prompt": prompt,
                "outputs": outputs,
                "scores": batch_scores
            })
    os.rename(args.candidates_path + "_scored", args.candidates_path)



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
        "--output_dir",
        type=str,
        default=None,
        help="The output directory to save the results",
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
        default=114514,
        help="The random seed for reproducibility (default: 114514)",
    )
    args = parser.parse_args()

    seed = args.seed
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.output_dir is not None and len(args.output_dir) > 0:
        os.makedirs(args.output_dir, exist_ok=True)
    start_generation(args, n_workers = 8)
        