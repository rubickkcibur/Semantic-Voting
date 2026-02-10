import torch
import tqdm
import transformers
import jsonlines
import re
import transformers
import numpy as np
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import argparse

GLOBAL_SEED = 42

def is_valid_answer(txt):
    if txt is None or txt == "None":
        return False
    if not txt.isascii():
        return False
    return True

def extract_sentence_pairs(path):
    extracted_data = []
    def extract_pred(txt):
        pattern = r'boxed\{([^}]*)\}'
        results = re.findall(pattern, txt)
        if results:
            ret = results[0]
            # ret = ret.replace("\"", "")
            # ret = ret.replace("\'", "")
            ret = ret.strip()
            return ret
        else:
            return None
        
    def extract_prompt(txt, path):
        if "cnn_dailymail" in path:
            assert "'\nPlease summarize the article in three sentences." in txt
            prompt = txt.split("'\nPlease summarize the article in three sentences.")[0].strip()
            assert "Here is the news article:\n'" in prompt
            prompt = prompt.split("Here is the news article:\n'")[-1].strip()
            return prompt
        elif "pubmed_summary" in path:
            assert "'\nPlease summarize the article in one or two sentences." in txt
            prompt = txt.split("'\nPlease summarize the article in one or two sentences.")[0].strip()
            assert "Here is the medical article:\n'" in prompt
            prompt = prompt.split("Here is the medical article:\n'")[-1].strip()
            return prompt
        elif "xsum" in path:
            assert "'\nPlease give your summary." in txt
            prompt = txt.split("'\nPlease give your summary.")[0].strip()
            assert "Here is the news report:\n'" in prompt
            prompt = prompt.split("Here is the news report:\n'")[-1].strip()
            return prompt
        else:
            assert "'\nPlease think about how to translate step by step." in txt
            prompt = txt.split("'\nPlease think about how to translate step by step.")[0].strip()
            assert "sentence is: '" in prompt
            prompt = prompt.split("sentence is: '")[-1].strip()
            return prompt


    with jsonlines.open(path, "r") as f:
        for item in f:
            prompt = extract_prompt(item["prompt"], path)
            candidates = item["outputs"]
            preds = [extract_pred(candidate) for candidate in candidates]
            extracted_data.append({
                "original_prompt": item["prompt"],
                "original_candidates": item["outputs"],
                "extracted_prompt": prompt,
                "extracted_candidates": preds
            })
    return extracted_data

def compute_sentence_embeddings(sentence_pairs):
    device = "cuda:0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "unsup-simcse-bert-base-uncased",
        padding_side="left",
    )
    model = transformers.AutoModel.from_pretrained(
        "unsup-simcse-bert-base-uncased",
        torch_dtype=torch.float32
    ).to(device)
    model.eval()
    with tqdm.tqdm(total=len(sentence_pairs), desc="Computing sentence embeddings") as pbar:
        for item in sentence_pairs:
            pbar.update(1)
            cands = item["extracted_candidates"]
            valid_index = [i for i in range(len(cands)) if is_valid_answer(cands[i])]
            inputs = [cand for cand in cands if is_valid_answer(cand)]
            if len(valid_index) == 0:
                item["embeddings"] = []
                item["valid_index"] = []
            else:
                encodings = tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=512,).to(device)
                with torch.no_grad():
                    output = model(**encodings)
                    emb = torch.nn.functional.normalize(output.last_hidden_state[:, 0, :], p=2, dim=1)
                    emb = emb.detach().cpu().double().numpy()
                item["embeddings"] = emb
                item["valid_index"] = valid_index
    return sentence_pairs

def compute_cluster_scores(sentence_pairs, args):
    if args.output_path_dpo_file is not None and len(args.output_path_dpo_file) > 0:
        os.makedirs(os.path.dirname(args.output_path_dpo_file), exist_ok=True)
    with tqdm.tqdm(total=len(sentence_pairs), desc="Compute Cluster Scores") as pbar, jsonlines.open(args.output_path_dpo_file, "w") as fw:
        for item in sentence_pairs:
            pbar.update(1)
            valid_index = item["valid_index"]
            if len(valid_index) < args.filter_length:
                item["cosine_scores"] = [None] * len(item["extracted_candidates"])
                continue
            emb = item["embeddings"]
            sim_matrix = cosine_similarity(emb)
            clusteror = hdbscan.HDBSCAN(
                metric='precomputed', 
                min_cluster_size = args.min_cluster_size, 
                min_samples = args.min_samples, 
                provide_probabilities = True, 
                allow_single_cluster = True,
                # prediction_data = True
            )
            labels = clusteror.fit_predict(sim_matrix)
            clusters = dict()
            isolated = set()
            for label_idx, c in enumerate(labels):
                if c < 0:
                    isolated.add(label_idx)
                    continue
                if c not in clusters:
                    clusters[c] = []
                clusters[c].append(label_idx)
            if len(clusters.keys()) < 1:
                continue
            max_size = max([len(v) for (k, v) in clusters.items()])
            max_c_name = -1
            for k in clusters.keys():
                if len(clusters[k]) == max_size: 
                    max_c_name = k
            chosen_id = random.choice(clusters[max_c_name])
            original_chosen_id = valid_index[chosen_id]
            if len(clusters.keys()) < 2:
                if len(isolated) > 0:
                    rej_id = random.choice(list(isolated))
                    original_rej_id = valid_index[rej_id]
                else:
                    continue
            else:
                min_size = min([len(v) for (k, v) in clusters.items()])
                min_c_name = -1
                for k in clusters.keys():
                    if len(clusters[k]) == min_size: 
                        min_c_name = k
                rej_id = random.choice(clusters[min_c_name])
                original_rej_id = valid_index[rej_id]
            fw.write({
                "chosen": item["original_candidates"][original_chosen_id],
                "rejected": item["original_candidates"][original_rej_id],
                "prompt": item["original_prompt"]
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The arguments for generation with vLLM on a HuggingFace dataset")
    parser.add_argument(
        "--candidate_path",
        type=str,
        required=True,
        help="The path of the source file containing candidate sentences",
    )
    parser.add_argument(
        "--output_path_scored_file",
        type=str,
        required=True,
        help="The path of the output file to save the scored candidates",
    )
    parser.add_argument(
        "--output_path_dpo_file",
        type=str,
        default=None,
        help="The path of the output file to save the preference data for DPO",
    )
    parser.add_argument(
        "--min_cluster_size",
        type = int,
        default = 5,
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--filter_length",
        type = int,
        default = 5
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed for reproducibility (default: 114514)",
    )
    args = parser.parse_args()

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    #Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    sentence_pairs = extract_sentence_pairs(args.candidate_path)
    sentence_pairs = compute_sentence_embeddings(sentence_pairs)
    sentence_pairs = compute_cluster_scores(sentence_pairs, args)