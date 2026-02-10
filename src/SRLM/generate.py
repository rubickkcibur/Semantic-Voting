from vllm import LLM, SamplingParams
import torch
import ray
import jsonlines
import os
import random
import numpy as np
import transformers
from data_processor.processor_registers import load_custom_dataset

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
    dataset = load_custom_dataset(
        args.dataset_name,
        tokenizer = tokenizer,
        cot = args.few_shot_cot,
        apply_chat_template = (args.mode == "chat"),
    )
    inputs = [ item["prompt"] for item in dataset["train"] ]
    data_chunks = split_data(inputs, n_workers)

    #generate
    sampling_params = SamplingParams(
        n = args.return_sequences,
        temperature = args.temperature,
        top_p = args.top_p,
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

    with jsonlines.open(os.path.join(args.output_dir, "{}_output_{}.jsonl".format(args.dataset_name, str(args.return_sequences))), "w") as writer:
        for instance in final_results:
            prompt = instance.prompt
            outputs = instance.outputs
            outputs = [output.text for output in outputs]
            writer.write({
                "prompt": prompt,
                "outputs": outputs
            })



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
        "--dataset_name",
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
        "--temperature",
        type=float,
        default=0.7,
        help="The temperature to use for sampling (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The top-p sampling probability (default: 0.9)",
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
    if args.output_dir is not None and len(args.output_dir) > 0:
        os.makedirs(args.output_dir, exist_ok=True)
    start_generation(args, n_workers = 8)
        