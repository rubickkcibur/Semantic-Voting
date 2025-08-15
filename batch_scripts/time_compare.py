import os
import subprocess
import tempfile
import yaml
import time

def compute_self_scores(base_model_name, dataset_name):
    # Define the command to run the SRLM cluster scoring script
    command = [
        "nohup", "python", "src/SRLM/critic.py",
        "--model_name_or_path", "/mnt/{}/rubickjiang/proj_storage/huggingface_models/{}".format(os.environ["MACLAB_NAS_NAME"], base_model_name),
        "--tokenizer_path", "/mnt/{}/rubickjiang/proj_storage/huggingface_models/{}".format(os.environ["MACLAB_NAS_NAME"], base_model_name),
        "--mode", "chat",
        "--candidates_path", "/mnt/{}/rubickjiang/codes/open-r1/data/SR_candidates/{}/{}_output_64.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
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

def compute_entropy_scores(base_model_name, dataset_name):
    # Define the command to run the SRLM cluster scoring script
    command = [
        "nohup", "accelerate", "launch", "--config_file", "/mnt/{}/rubickjiang/codes/accelerate_config/config_acc.yaml".format(os.environ["MACLAB_NAS_NAME"]),
        "src/SRLM/record_loss.py",
        "--model_name_or_path", "/mnt/{}/rubickjiang/proj_storage/huggingface_models/{}".format(os.environ["MACLAB_NAS_NAME"], base_model_name),
        "--tokenizer_path", "/mnt/{}/rubickjiang/proj_storage/huggingface_models/{}".format(os.environ["MACLAB_NAS_NAME"], base_model_name),
        "--mode", "chat",
        "--candidates_path", "/mnt/{}/rubickjiang/codes/open-r1/data/SR_candidates/{}/{}_output_64.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--scored_path", "/mnt/{}/rubickjiang/codes/open-r1/data/retry_candidates/{}/{}_entropy_scored.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--dpo_path", "/mnt/{}/rubickjiang/codes/open-r1/data/retry_candidates/{}/{}_entropy_dpo.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--few_shot_cot", "False",
        "--batch_size", "2",
        "--max_model_len", "2048",
        "--seed", "42"
    ]

    # Run the command
    subprocess.run(command, check=True)

def compute_cluster_scores(base_model_name, dataset_name, min_cluster_size=5, min_samples=2):
    # Define the command to run the SRLM cluster scoring script
    command = [
        "nohup", "python", "src/SRLM/cluster_score.py",
        "--candidate_path", "/mnt/{}/rubickjiang/codes/open-r1/data/SR_candidates/{}/{}_output_64.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--output_path_scored_file", "/mnt/{}/rubickjiang/codes/open-r1/data/retry_candidates/{}/{}_scored.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--output_path_dpo_file", "/mnt/{}/rubickjiang/codes/open-r1/data/retry_candidates/{}/{}_dpo.jsonl".format(os.environ["MACLAB_NAS_NAME"], base_model_name, dataset_name),
        "--min_cluster_size", "{}".format(min_cluster_size),
        "--min_samples", "{}".format(min_samples),
        "--filter_length", "5",
        "--seed", "42"
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
    searching_pairs = [
        ("Llama-3.2-1B-Instruct", "wmt24pp_zh"),
        ("Llama-3.2-3B-Instruct", "wmt24pp_zh"),
        ("Meta-Llama-3-8B-Instruct", "wmt24pp_zh"),
        ("Qwen2.5-1.5B-Instruct", "wmt24pp_zh"),
        ("Qwen2.5-3B-Instruct", "wmt24pp_zh"),
        ("Qwen2.5-7B-Instruct", "wmt24pp_zh"),
    ]
    for base_model_name, dataset_name in searching_pairs:
        try:
            define_system_vars()

            start = time.time()
            compute_entropy_scores(base_model_name, dataset_name)
            print("Time Recoded: Entropy Scores Computation for {}: {:.2f} seconds".format(base_model_name, time.time() - start))

            start = time.time()
            compute_cluster_scores(
                base_model_name,
                dataset_name,
                min_cluster_size=5,
                min_samples=2
            )
            print("Time Recoded: Cluster Scores Computation for {}: {:.2f} seconds".format(base_model_name, time.time() - start))

            start = time.time()
            compute_self_scores(base_model_name, dataset_name)
            print("Time Recoded: Self Scores Computation for {}: {:.2f} seconds".format(base_model_name, time.time() - start))

        except Exception as e:
            print(f"An error occurred while processing {base_model_name} on {dataset_name}: {e}")
            quit()
            continue
