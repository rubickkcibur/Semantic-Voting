import os
import subprocess

def evaluate(model_name_or_path, dataset_name, tokenizer_path=None, max_new_tokens=512):
    # Define the command to run the evaluation script
    command = [
        "accelerate", "launch", "--config_file", "/mnt/{}/rubickjiang/codes/accelerate_config/config_acc.yaml".format(os.environ["MACLAB_NAS_NAME"]),
        "src/open_r1/evaluation.py",
        "--model_name_or_path", model_name_or_path,
        "--tokenizer_path", tokenizer_path if tokenizer_path else model_name_or_path,
        "--output_dir", "",
        "--mode", "chat",
        "--dataset_name", dataset_name,
        "--bf16", "True",
        "--few_shot_cot", "False",
        "--per_device_eval_batch_size", "8",
        "--max_new_tokens", "{}".format(max_new_tokens),
        "--model_max_length", "2048",
    ]

    # Run the command
    subprocess.run(command, check=True)

def define_system_vars():
    os.environ["MACLAB_NAS_NAME"] = "maclabcv2"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["SEED"] = "42"
    os.environ["ACCELERATE_LOG_LEVEL"]="info"


if __name__ == "__main__":
    define_system_vars()
    for model in [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Meta-Llama-3-8B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]:
        evaluate(
            model_name_or_path="/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/{}".format(model),
            tokenizer_path="/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/{}".format(model),
            dataset_name="alpaca_eval",
            max_new_tokens=512
        )
    
    