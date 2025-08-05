from external_lib import mt5_model
import torch
import tqdm
import transformers
import jsonlines
import re
import transformers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import hdbscan
from scipy.stats import kendalltau, spearmanr
from datasets import load_dataset
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import umap

MACLAB_NAS_NAME = "maclabcv2"
GLOBAL_SEED = 42

def reward_model_score(pred_txt, sources, model, tokenizer):
    batch_size = 64
    pred_txt = [pred if pred is not None else "None" for pred in pred_txt]
    total_length = len(pred_txt)
    no_reference_scores = []
    for i in range(0, total_length, batch_size):
        batch_preds = pred_txt[i:i+batch_size]
        batch_source = sources[i:i+batch_size]
        batch_texts = [
            "source: " + source + " candidate: " + pred
            for pred, source in zip(batch_preds, batch_source)
        ]
        batch_inputs = tokenizer(
            batch_texts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda:0")
        with torch.no_grad():
            outputs = model(**batch_inputs)
            scores = outputs.predictions.detach().cpu().float().numpy().tolist()
        no_reference_scores.extend(scores)
    # reference_scores = []
    # with tqdm.tqdm(total=total_length, desc="Calculating MQM scores with reference") as pbar:
    #     for i in range(0, total_length, batch_size):
    #         batch_preds = pred_txt[i:i+batch_size]
    #         batch_source = kwargs_list["target"][i:i+batch_size]
    #         batch_refs = kwargs_list["source"][i:i+batch_size]
    #         batch_texts = [
    #             "source: " + source + " candidate: " + pred + " reference: " + ref
    #             for pred, source, ref in zip(batch_preds, batch_source, batch_refs)
    #         ]
    #         batch_inputs = tokenizer(
    #             batch_texts,
    #             max_length=512,
    #             truncation=True,
    #             padding="max_length",
    #             return_tensors="pt"
    #         ).to("cuda:0")
    #         with torch.no_grad():
    #             outputs = model(**batch_inputs)
    #             scores = outputs.predictions.detach().cpu().float().numpy().tolist()
    #         reference_scores.extend(scores)
    #         pbar.update(len(batch_texts))
    return no_reference_scores

def compute_metrix():
    data = []
    def extract_pred(txt):
        pattern = r'\\boxed\{([^}]*)\}'
        results = re.findall(pattern, txt)
        if results:
            ret = results[0]
            # ret = ret.replace("\"", "")
            # ret = ret.replace("\'", "")
            ret = ret.strip()
            return ret
        else:
            return None
    with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/open-r1/data/SR_candidates/wmt_output.jsonl", "r") as f:
        for item in f:
            prompt = item["prompt"].split("The Chinese sentence is:")[-1].split("Please think about how to translate step by step.")[0].strip()
            prompt = prompt.replace("\'", "").strip()
            candidates = item["outputs"]
            preds = [extract_pred(candidate) for candidate in candidates]
            data.append({
                "chinese": prompt,
                "candidates": preds
            })
    model = mt5_model.MT5ForRegression.from_pretrained("/mnt/{}/rubickjiang/proj_storage/huggingface_models/metricx-24-hybrid-large-v2p6-bfloat16".format(MACLAB_NAS_NAME), torch_dtype=torch.bfloat16)
    model.to("cuda:0")
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", trust_remote_code=True)
    tokenizer.padding_side = "left"
    with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/open-r1/data/SR_candidates/wmt_output_metrix.jsonl", "w") as f:
        with tqdm.tqdm(total=len(data)) as pbar:
            for item in data:
                sources = [item["chinese"]] * len(item["candidates"])
                cands = item["candidates"]
                scores = reward_model_score(cands, sources, model, tokenizer)
                f.write({
                    "chinese": item["chinese"],
                    "candidates": [cand if cand is not None else "None" for cand in item["candidates"]],
                    "scores": scores
                })
                pbar.update(1)

def sim_emb():
    device = "cuda:0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        padding_side="left",
    )
    model = transformers.AutoModel.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        torch_dtype=torch.bfloat16
    ).to(device)
    with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/open-r1/data/SR_candidates/wmt_output_metrix.jsonl", "r") as fr:
        data = [item for item in fr]
    model.eval()

    item = data[233]
    print(item["chinese"])
    none_idx = [i for i in range(len(item["candidates"])) if item["candidates"][i] == "None"]
    cands = [d for idx, d in enumerate(item["candidates"]) if idx not in none_idx]
    scores = [d for idx, d in enumerate(item["scores"]) if idx not in none_idx]
    encodings = tokenizer(
        cands,
        return_tensors="pt",
        padding='longest',
        truncation=True,
        max_length=512,).to(device)
    with torch.no_grad():
        output = model(**encodings)
        emb = torch.nn.functional.normalize(output.last_hidden_state[:, 0, :], p=2, dim=1)
        emb = emb.detach().cpu().float().numpy()

    reducer = umap.UMAP(n_neighbors=5, metric='cosine', n_components=2, random_state=GLOBAL_SEED, min_dist = 0.5)
    emb_umap = reducer.fit_transform(emb)
    sim_matrix = cosine_similarity(emb)

    # pca = PCA(n_components=20, svd_solver = "covariance_eigh")
    # emb_pca = pca.fit_transform(emb)

    # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    # emb_tsne = tsne.fit_transform(emb_pca)

    plt.figure(dpi=300)

    # sns.heatmap(sim_matrix, cmap="viridis")
    # plt.savefig("./cosine2.png")
    # quit()

    cm = plt.cm.get_cmap("OrRd_r")
    sc = plt.scatter(
        x = emb_umap[:, 0], 
        y = emb_umap[:, 1],
        c = scores,
        vmin = 1,
        vmax = 5,
        s = 5,
        cmap = cm)
    plt.colorbar(sc)
    plt.savefig("./test.png")
    quit()

def test_designed_metric():
    device = "cuda:0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        padding_side="left",
    )
    model = transformers.AutoModel.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        torch_dtype=torch.float32
    ).to(device)
    with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/open-r1/data/SR_candidates/wmt_output_metrix.jsonl", "r") as fr:
        data = [item for item in fr]
    model.eval()

    cnt = 0
    kendals = []
    spears = []
    min_sim = 1
    main_cluster_size = 0
    cluster_types = 0
    topK = {
        "top@1": 0,
        "top@2": 0,
        "top@3": 0,
        "top@4": 0,
        "top@5": 0
    }
    top_reward = 0
    tail_reward = 0
    win_rate = 0

    def is_invalid(txt):
        if txt is None or txt == "None":
            return True
        if not txt.isascii():
            return True
        return False

    with tqdm.tqdm(total = len(data)) as pbar:
        for item in data:
            pbar.update(1)
            none_idx = [i for i in range(len(item["candidates"])) if is_invalid(item["candidates"][i])]
            cands = [d for idx, d in enumerate(item["candidates"]) if idx not in none_idx]
            scores = [d for idx, d in enumerate(item["scores"]) if idx not in none_idx]
            if len(cands) < 5:
                continue
            encodings = tokenizer(
                cands,
                return_tensors="pt",
                padding='longest',
                truncation=True,
                max_length=512,).to(device)
            with torch.no_grad():
                output = model(**encodings)
                emb = torch.nn.functional.normalize(output.last_hidden_state[:, 0, :], p=2, dim=1)
                emb = emb.detach().cpu().double().numpy()
            
            sim_matrix = cosine_similarity(emb)
            if np.min(sim_matrix) < min_sim:
                min_sim = np.min(sim_matrix)

            # use the whole
            # clusters = {0: list(range(len(cands)))}
            #################

            # devide clusters
            clusteror = hdbscan.HDBSCAN(
                metric='precomputed', 
                min_cluster_size = 3, 
                min_samples = 3, 
                provide_probabilities = True, 
                allow_single_cluster = True,
                # prediction_data = True
            )
            labels = clusteror.fit_predict(sim_matrix)
            cluster_types += len(set(labels)) - 1 # -1 for noise
            main_cluster_size += max([len([i for i in labels if i == c]) for c in set(labels) if c != -1])
            clusters = dict() # {cluster_num -> list[belonged element idx]}
            for idx, c in enumerate(labels):
                if c < 0:
                    continue
                if c not in clusters:
                    clusters[c] = []
                clusters[c].append(idx)
            if len(clusters.keys()) > 1:
                max_size = max([len(v) for (k, v) in clusters.items()])
                max_c_name = -1
                for k in clusters.keys():
                    if len(clusters[k]) == max_size: 
                        max_c_name = k
                clusters = {max_c_name: clusters[max_c_name]}
            ############
            
            featured_clusters = {
                k: [emb[idx] for idx in v]
                for k, v in clusters.items()
            } # {cluster_num -> list[belonged element features]}

            center_feature = {
                k: np.stack(v).mean(axis=0) / np.linalg.norm(np.stack(v).mean(axis=0))
                for k, v in featured_clusters.items()
            } # {cluster_num -> cluster_avg_feature}

            cos_sim_clusters = {
                k: [np.dot(center_feature[k], f)/ (np.linalg.norm(center_feature[k]) * np.linalg.norm(f)) for f in v]
                for k, v in featured_clusters.items()
            } # {cluster_num -> list[belonged element ddistance to the cluster_avg_feature]}

            score_clusters = {
                k: [scores[idx] for idx in v]
                for k, v in clusters.items()
            } # {cluster_num -> list[belonged element scores]}
            assert len(score_clusters.keys()) == 1
            metric_pairs = []
            for k in score_clusters:
                scores = score_clusters[k]
                pseudo_metric = cos_sim_clusters[k]
                for s, p in zip(scores, pseudo_metric):
                    metric_pairs.append({
                        "score": s,
                        "pseudo": p
                    })
            score_rank = sorted(enumerate(metric_pairs), key=lambda x: x[1]["score"]) # Score 从小到大 (Score 越小越好)
            score_rank = [i for i, val in score_rank]
            pseudo_rank = sorted(enumerate(metric_pairs), key=lambda x: x[1]["pseudo"], reverse=True) # cos_sim 从大到小 (cos_sim越大越接近)
            pseudo_rank = [i for i, val in pseudo_rank]
            random_rank = list(range(len(metric_pairs)))
            random.shuffle(random_rank)

            if len(score_rank) < 5:
                continue

            r_tau, r_p = kendalltau(score_rank, random_rank)
            ps_tau, ps_p = kendalltau(score_rank, pseudo_rank)
            kendal_stats = {
                "r_tau": r_tau,
                "r_p": r_p,
                "ps_tau": ps_tau,
                "ps_p": ps_p,
            }
            kendals.append(kendal_stats)

            r_rho, r_p = spearmanr(score_rank, random_rank)
            ps_rho, ps_p = spearmanr(score_rank, pseudo_rank)
            spear_stats = {
                "r_rho": r_rho,
                "r_p": r_p if not np.isnan(r_p) else 1,
                "ps_rho": ps_rho,
                "ps_p": ps_p if not np.isnan(ps_p) else 1,
            }
            spears.append(spear_stats)

            topK["top@1"] += 1 if score_rank[0] == pseudo_rank[0] else 0
            topK["top@2"] += len(set(score_rank[:2]) & set(pseudo_rank[:2])) / 2
            topK["top@3"] += len(set(score_rank[:3]) & set(pseudo_rank[:3])) / 3
            topK["top@4"] += len(set(score_rank[:4]) & set(pseudo_rank[:4])) / 4
            topK["top@5"] += len(set(score_rank[:5]) & set(pseudo_rank[:5])) / 5

            # mid = len(pseudo_rank)//2
            higher = sum([metric_pairs[idx]["score"] for idx in pseudo_rank[:1]])
            lower = sum([metric_pairs[idx]["score"] for idx in pseudo_rank[-1:]])
            win_rate += 1 if higher > lower else 0
            top_reward += higher / 1
            tail_reward += lower / 1

    for k in topK:
        topK[k] /= len(kendals)
    print("TopK Rewards:", top_reward / len(kendals))
    print("Tail Rewards:", tail_reward / len(kendals))
    print("Win Rate:", win_rate / len(kendals))
    print("TopK Results:", topK)
    print("{} filtered results in total".format(len(kendals)))
    print("Avg Main cluster size:", main_cluster_size / len(data))
    print("Avg Cluster types:", cluster_types / len(data))
    print("Kentall_tau Results:\n")
    for k in kendals[0]:
        value = [obj[k] for obj in kendals]
        value = sum(value)/len(value)
        print(k + ":" + str(value))
    print("\nSpears Results:\n")
    for k in spears[0]:
        value = [obj[k] for obj in spears]
        value = sum(value)/len(value)
        print(k + ":" + str(value))
    print("min similarity:", min_sim)

seed = GLOBAL_SEED
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


test_designed_metric()