import os
import subprocess
import tempfile
import yaml

accelerate_config_path = "" # customize your accelerate config path here
model_dir = "" # customize your model directory

def compute_self_scores(base_model_name, dataset_name):
    # Define the command to run the SRLM cluster scoring script
    command = [
        "nohup", "python", "src/SRLM/critic.py",
        "--model_name_or_path", "{}/{}".format(model_dir, base_model_name),
        "--tokenizer_path", "{}/{}".format(model_dir, base_model_name),
        "--mode", "chat",
        "--candidates_path", "data/main_results/candidates/{}/{}_output_64.jsonl".format(base_model_name, dataset_name),
        "--scored_path", "data/main_results/candidates/{}/{}_self_scored.jsonl".format(base_model_name, dataset_name),
        "--dpo_path", "data/main_results/candidates/{}/{}_self_dpo.jsonl".format(base_model_name, dataset_name),
        "--few_shot_cot", "False",
        "--batch_size", "4",
        "--return_sequences", "1",
        "--max_model_len", "3072",
        "--max_new_tokens", "512",
        "--use_format_filter", "True",
        "--seed", "42",
        "--n_workers", "8"
    ]

    # Run the command
    subprocess.run(command, check=True)

def train(base_model_name, dataset_name):
    # Define the command to run the DPO training script
    yaml_file = "src/configs/dpo/{}.yaml".format(base_model_name)
    assert os.path.exists(yaml_file)
    with open(yaml_file, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    params["model_name_or_path"] = "{}/{}".format(model_dir, base_model_name)
    params["dataset_name"] = "data/main_results/candidates/{}/{}_self_dpo.jsonl".format(base_model_name, dataset_name)
    params["output_dir"] = "data/main_results/models/{}-DPO-{}-self".format(base_model_name, dataset_name)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmpfile:
        yaml.dump(params, tmpfile, allow_unicode=True)
        temp_file_name = tmpfile.name

    command = [
        "nohup", "accelerate", "launch", "--config_file", accelerate_config_path,
        "src/SRLM/dpo.py",
        "--config", temp_file_name,
    ]

    # Run the command
    subprocess.run(command, check=True)
    os.unlink(temp_file_name)  # Clean up the temporary file

def evaluate(base_model_name, dataset_name, max_new_tokens=512):
    # Define the command to run the evaluation script
    command = [
        "nohup", "accelerate", "launch", "--config_file", accelerate_config_path,
        "src/open_r1/evaluation.py",
        "--model_name_or_path", "data/main_results/models/{}-DPO-{}-self".format(base_model_name, dataset_name),
        "--tokenizer_path", "{}/{}".format(model_dir, base_model_name),
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
    # Example usage
    pairs = [
        ("Llama-3.2-1B-Instruct", "wmt24pp_de"),
        ("Llama-3.2-3B-Instruct", "wmt24pp_de"),
        ("Meta-Llama-3-8B-Instruct", "wmt24pp_de"),
        ("Qwen2.5-1.5B-Instruct", "wmt24pp_de"),
        ("Qwen2.5-3B-Instruct", "wmt24pp_de"),
        ("Qwen2.5-7B-Instruct", "wmt24pp_de"),
    ]
    for base_model_name, dataset_name in pairs:
        try:
            define_system_vars()
            compute_self_scores(base_model_name, dataset_name)
            train(base_model_name, dataset_name)
            evaluate(
                base_model_name, 
                dataset_name, 
                max_new_tokens=512
            )
        except Exception as e:
            print(f"An error occurred while processing {base_model_name} on {dataset_name}: {e}")
            quit()