import os
import subprocess

def evaluate(model_name_or_path, dataset_name):
    # Define the command to run the evaluation script
    command = [
        "accelerate", "launch", "--config_file", "/mnt/{}/rubickjiang/codes/accelerate_config/config_acc.yaml".format(os.environ["MACLAB_NAS_NAME"]),
        "src/open_r1/evaluation.py",
        "--model_name_or_path", model_name_or_path,
        "--tokenizer_path", "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Meta-Llama-3-8B-Instruct",
        "--output_dir", "",
        "--mode", "chat",
        "--dataset_name", dataset_name,
        "--bf16", "True",
        "--few_shot_cot", "False",
        "--per_device_eval_batch_size", "8",
        "--max_new_tokens", "512",
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
    # Example usage
    for base_model_name in [
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Qwen2.5-7B-Instruct",
    ]:
    # dataset_name = "wmt24pp_es"
        for dataset_name in [
            "wmt24pp_zh",
            "wmt24pp_de", 
            "wmt24pp_fr", 
            "wmt24pp_ru", 
            "wmt24pp_es"
        ]:
            define_system_vars()
            evaluate("/mnt/maclabcv2/rubickjiang/codes/open-r1/data/models/Meta-Llama-3-8B-Instruct-DPO-{}-4-2".format(dataset_name), dataset_name)
    