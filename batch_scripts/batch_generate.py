import os
import subprocess

accelerate_config_path = "" # customize your accelerate config path here
model_dir = "" # customize your model directory

def generate_sr_candidates(base_model_name, dataset_name, temperature="0.7"):
    # Define the command to run the SRLM generation script
    command = [
        "nohup", "python", "src/SRLM/generate.py",
        "--model_name_or_path", "data/main_results/main_models/{}-DPO-{}-entropy".format( base_model_name, dataset_name),
        "--tokenizer_path", "{}/{}".format(model_dir, base_model_name),
        "--output_dir", "data/main_results/candidates/{}".format(base_model_name),
        "--mode", "chat",
        "--dataset_name", dataset_name,
        "--few_shot_cot", "False",
        "--batch_size", "4",
        "--return_sequences", "64",
        "--max_new_tokens", "512",
        "--max_model_len", "2048",
        "--seed", "42",
        "--temperature", temperature,
    ]

    # Run the command
    subprocess.run(command, check=True)

def define_system_vars():
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["SEED"] = "42"
    os.environ["ACCELERATE_LOG_LEVEL"]="info"


if __name__ == "__main__":
    define_system_vars()
    for model in [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Meta-Llama-3-8B-Instruct",
    ]:
        for dataset in [
            "wmt24pp_de",
            "wmt24pp_fr",
            "wmt24pp_ru",
            "wmt24pp_es",
            "cnn_dailymail",
            "pubmed_summary"
        ]:
            generate_sr_candidates(
                base_model_name=model,
                dataset_name=dataset,
                temperature="0.7"
            )
    
    