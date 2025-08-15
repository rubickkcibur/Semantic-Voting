import os
import subprocess

def evaluate(model_name_or_path, dataset_name, tokenizer_path=None):
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
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Llama-3.2-1B-Instruct",
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Qwen2.5-1.5B-Instruct",
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Llama-3.2-3B-Instruct",
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Qwen2.5-3B-Instruct",
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Meta-Llama-3-8B-Instruct",
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Qwen2.5-7B-Instruct",
    ]:
        for dataset_name in [
            "wmt14_es",
            "wmt14_fr",
            "wmt19_de",
            "wmt19_ru",
        ]:
            define_system_vars()
            evaluate(base_model_name, dataset_name)
    trained_model_paths = [
        ("wmt14_es", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Llama-3.2-1B-Instruct-DPO-wmt24pp_es-6-5", "Llama-3.2-1B-Instruct"),
        ("wmt14_es", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/models/Llama-3.2-3B-Instruct-DPO-wmt24pp_es-5-2", "Llama-3.2-3B-Instruct"),
        ("wmt14_es", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Meta-Llama-3-8B-Instruct-DPO-wmt24pp_es-4-2", "Meta-Llama-3-8B-Instruct"),
        ("wmt14_es", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-1.5B-Instruct-DPO-wmt24pp_es-5-1", "Qwen2.5-1.5B-Instruct"),
        ("wmt14_es", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-3B-Instruct-DPO-wmt24pp_es-6-1", "Qwen2.5-3B-Instruct"),
        ("wmt14_es", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-7B-Instruct-DPO-wmt24pp_es-4-2", "Qwen2.5-7B-Instruct"),
        ("wmt14_fr", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Llama-3.2-1B-Instruct-DPO-wmt24pp_fr-5-2", "Llama-3.2-1B-Instruct"),
        ("wmt14_fr", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/models/Llama-3.2-3B-Instruct-DPO-wmt24pp_fr-5-2", "Llama-3.2-3B-Instruct"),
        ("wmt14_fr", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Meta-Llama-3-8B-Instruct-DPO-wmt24pp_fr-4-2", "Meta-Llama-3-8B-Instruct"),
        ("wmt14_fr", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-1.5B-Instruct-DPO-wmt24pp_fr-5-2", "Qwen2.5-1.5B-Instruct"),
        ("wmt14_fr", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-3B-Instruct-DPO-wmt24pp_fr-6-2", "Qwen2.5-3B-Instruct"),
        ("wmt14_fr", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-7B-Instruct-DPO-wmt24pp_fr-4-2", "Qwen2.5-7B-Instruct"),
        ("wmt19_de", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Llama-3.2-1B-Instruct-DPO-wmt24pp_de-5-2", "Llama-3.2-1B-Instruct"),
        ("wmt19_de", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/models/Llama-3.2-3B-Instruct-DPO-wmt24pp_de-5-2", "Llama-3.2-3B-Instruct"),
        ("wmt19_de", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Meta-Llama-3-8B-Instruct-DPO-wmt24pp_de-4-2", "Meta-Llama-3-8B-Instruct"),
        ("wmt19_de", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-1.5B-Instruct-DPO-wmt24pp_de-5-2", "Qwen2.5-1.5B-Instruct"),
        ("wmt19_de", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-3B-Instruct-DPO-wmt24pp_de-5-2", "Qwen2.5-3B-Instruct"),
        ("wmt19_de", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-7B-Instruct-DPO-wmt24pp_de-4-2", "Qwen2.5-7B-Instruct"),
        ("wmt19_ru", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Llama-3.2-1B-Instruct-DPO-wmt24pp_ru-5-2", "Llama-3.2-1B-Instruct"),
        ("wmt19_ru", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/models/Llama-3.2-3B-Instruct-DPO-wmt24pp_ru-5-2", "Llama-3.2-3B-Instruct"),
        ("wmt19_ru", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Meta-Llama-3-8B-Instruct-DPO-wmt24pp_ru-4-2", "Meta-Llama-3-8B-Instruct"),
        ("wmt19_ru", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-1.5B-Instruct-DPO-wmt24pp_ru-6-1", "Qwen2.5-1.5B-Instruct"),
        ("wmt19_ru", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-3B-Instruct-DPO-wmt24pp_ru-5-2", "Qwen2.5-3B-Instruct"),
        ("wmt19_ru", "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/main_models/Qwen2.5-7B-Instruct-DPO-wmt24pp_ru-4-1", "Qwen2.5-7B-Instruct"),
    ]
    for dataset_name, model_path, tokenizer_path in trained_model_paths:
        define_system_vars()
        evaluate(model_path, dataset_name, tokenizer_path=tokenizer_path)
    