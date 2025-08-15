import os
import subprocess
import tempfile
import yaml

def compute_self_scores(base_model_name, dataset_name):
    # Define the command to run the SRLM cluster scoring script
    command = [
        "nohup", "python", "src/SRLM/critic.py",
        "--model_name_or_path", "/mnt/{}/rubickjiang/proj_storage/huggingface_models/{}".format(os.environ["MACLAB_NAS_NAME"], base_model_name),
        "--tokenizer_path", "/mnt/{}/rubickjiang/proj_storage/huggingface_models/{}".format(os.environ["MACLAB_NAS_NAME"], base_model_name),
        "--mode", "chat",
        "--candidates_path", "/mnt/{}/rubickjiang/codes/open-r1/data/retry_candidates/{}/{}_output_64.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--scored_path", "/mnt/{}/rubickjiang/codes/open-r1/data/retry_candidates/{}/{}_self_scored.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--dpo_path", "/mnt/{}/rubickjiang/codes/open-r1/data/retry_candidates/{}/{}_self_dpo.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--few_shot_cot", "False",
        "--batch_size", "4",
        "--return_sequences", "1",
        "--max_model_len", "3072",
        "--max_new_tokens", "512",
        "--seed", "42"
    ]

    # Run the command
    subprocess.run(command, check=True)

def train(base_model_name, dataset_name):
    # Define the command to run the DPO training script
    yaml_file = "/mnt/{}/rubickjiang/codes/open-r1/src/configs/dpo/{}.yaml".format(os.environ["MACLAB_NAS_NAME"], base_model_name)
    assert os.path.exists(yaml_file)
    with open(yaml_file, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    params["model_name_or_path"] = "/mnt/{}/rubickjiang/proj_storage/huggingface_models/{}".format(os.environ["MACLAB_NAS_NAME"], base_model_name)
    params["dataset_name"] = "/mnt/{}/rubickjiang/codes/open-r1/data/retry_candidates/{}/{}_self_dpo.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name)
    params["output_dir"] = "/mnt/{}/rubickjiang/codes/open-r1/data/retry_models/{}-DPO-{}-self".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmpfile:
        yaml.dump(params, tmpfile, allow_unicode=True)
        temp_file_name = tmpfile.name

    command = [
        "nohup", "accelerate", "launch", "--config_file", "/mnt/{}/rubickjiang/codes/accelerate_config/config_ds3.yaml".format(os.environ["MACLAB_NAS_NAME"]),
        "src/SRLM/dpo.py",
        "--config", temp_file_name,
    ]

    # Run the command
    subprocess.run(command, check=True)
    os.unlink(temp_file_name)  # Clean up the temporary file

def evaluate(base_model_name, dataset_name):
    # Define the command to run the evaluation script
    command = [
        "nohup", "accelerate", "launch", "--config_file", "/mnt/{}/rubickjiang/codes/accelerate_config/config_acc.yaml".format(os.environ["MACLAB_NAS_NAME"]),
        "src/open_r1/evaluation.py",
        "--model_name_or_path", "/mnt/{}/rubickjiang/codes/open-r1/data/retry_models/{}-DPO-{}-self".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--tokenizer_path", "/mnt/{}/rubickjiang/proj_storage/huggingface_models/{}".format(os.environ["MACLAB_NAS_NAME"], base_model_name),
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
        # "Llama-3.2-1B-Instruct",
        "Qwen2.5-1.5B-Instruct", 
        # "Llama-3.2-3B-Instruct", 
        # "Qwen2.5-3B-Instruct",
        # "Meta-Llama-3-8B-Instruct",
        # "Qwen2.5-7B-Instruct"
    ]:
        for dataset_name in [
            "wmt24pp_de", 
            # "wmt24pp_zh", 
            # "wmt24pp_fr", 
            # "wmt24pp_ru",
            # "wmt24pp_es",
            # "cnn_dailymail",
            # "pubmed_summary"
        ]:
            try:
                define_system_vars()
                # generate_sr_candidates(base_model_name, dataset_name)
                compute_self_scores(base_model_name, dataset_name)
                train(base_model_name, dataset_name)
                evaluate(base_model_name, dataset_name)
            except Exception as e:
                print(f"An error occurred while processing {base_model_name} on {dataset_name}: {e}")
                quit()
    