import os
import subprocess

accelerate_config_path = "" # customize your accelerate config path here

def evaluate(model_name_or_path, dataset_name, tokenizer_path=None, max_new_tokens=512):
    # Define the command to run the evaluation script
    command = [
        "accelerate", "launch", "--config_file", accelerate_config_path,
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
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["SEED"] = "42"
    os.environ["ACCELERATE_LOG_LEVEL"]="info"


if __name__ == "__main__":
    model_name = "" # customize your testing model name here
    dataset = "wmt24pp_de" # customize your testing dataset name here
    define_system_vars()
    evaluate(
        model_name_or_path=model_name,
        tokenizer_path=model_name,
        dataset_name=dataset,
        max_new_tokens=800
    )
    
    