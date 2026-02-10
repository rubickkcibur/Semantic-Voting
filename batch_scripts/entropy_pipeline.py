import os
import subprocess
import tempfile
import yaml

accelerate_config_path = "" # customize your accelerate config path here
model_dir = "" # customize your model directory

def compute_entropy_scores(base_model_name, dataset_name):
    # Define the command to run the SRLM cluster scoring script
    command = [
        "nohup", "accelerate", "launch", "--config_file", accelerate_config_path,
        "src/SRLM/record_loss.py",
        "--model_name_or_path", "{}/{}".format(model_dir, base_model_name),
        "--tokenizer_path", "{}/{}".format(model_dir, base_model_name),
        "--mode", "chat",
        "--candidates_path", "data/main_results/candidates/{}/{}_output_64.jsonl".format(base_model_name, dataset_name),
        "--scored_path", "data/main_results/candidates/{}/{}_entropy_scored.jsonl".format(base_model_name, dataset_name),
        "--dpo_path", "data/main_results/candidates/{}/{}_entropy_dpo.jsonl".format(base_model_name, dataset_name),
        "--few_shot_cot", "False",
        "--use_format_filter", "True",
        "--batch_size", "2",
        "--max_model_len", "2048",
        "--seed", "42"
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
    params["dataset_name"] = "data/main_results/candidates/{}/{}_entropy_dpo.jsonl".format(base_model_name, dataset_name)
    params["output_dir"] = "data/main_results/models/{}-DPO-{}-entropy".format(base_model_name, dataset_name)
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
        "--model_name_or_path", "data/main_results/models/{}-DPO-{}-entropy".format(base_model_name, dataset_name),
        "--tokenizer_path", "{}/{}".format(model_dir, base_model_name),
        "--output_dir", "",
        "--mode", "chat",
        "--dataset_name", dataset_name,
        "--bf16", "True",
        "--few_shot_cot", "False",
        "--per_device_eval_batch_size", "8",
        "--max_new_tokens", "{}".format(str(max_new_tokens)),
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
    for base_model_name in [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Meta-Llama-3-8B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]:
        for dataset_name in [
            "wmt24pp_de",
            "wmt24pp_fr",
            "wmt24pp_ru",
            "wmt24pp_es",
            "cnn_dailymail",
            "pubmed_summary",
        ]:
            if base_model_name == "Qwen2.5-3B-Instruct" and dataset_name in ["cnn_dailymail", "pubmed_summary"]:
                continue
            if base_model_name == "Qwen2.5-1.5B-Instruct" and dataset_name in ["pubmed_summary"]:
                continue
            try:
                define_system_vars()
                compute_entropy_scores(base_model_name, dataset_name)
                train(base_model_name, dataset_name)
                evaluate(
                    base_model_name, 
                    dataset_name, 
                    max_new_tokens=800 if (base_model_name == "Qwen2.5-7B-Instruct" and "wmt" in dataset_name) else 512
                )
            except Exception as e:
                print(f"An error occurred while processing {base_model_name} on {dataset_name}: {e}")
                quit()
        