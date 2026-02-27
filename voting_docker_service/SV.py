import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

def compute_voting_score(embs):
    center_feature = np.stack(embs).mean(axis=0) / np.linalg.norm(np.stack(embs).mean(axis=0))
    cos_sim_cluster = [np.dot(center_feature, f)/ (np.linalg.norm(center_feature) * np.linalg.norm(f)) for f in embs]
    return cos_sim_cluster

def compute_scores(sentences, emb_model, emb_tokenizer, batch_size=32, clustering_method="whole", min_cluster_size=None, min_samples=None):
    if clustering_method == "HDBSCAN" and len(sentences) < min_cluster_size:
        return [None] * len(sentences)
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        inputs = emb_tokenizer(batch_sentences, padding='longest', truncation=True, max_length=1024, return_tensors="pt").to(emb_model.device)
        with torch.no_grad():
            outputs = emb_model(**inputs)
            emb = torch.nn.functional.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)
            emb = emb.detach().cpu().double().numpy()
            all_embeddings.append(emb)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    if clustering_method == "whole":
        scores = compute_voting_score(all_embeddings)
        return scores
    if clustering_method == "HDBSCAN":
        sim_matrix = cosine_similarity(all_embeddings)
        clusteror = hdbscan.HDBSCAN(
            metric='precomputed', 
            min_cluster_size = min_cluster_size, 
            min_samples = min_samples, 
            provide_probabilities = True, 
            allow_single_cluster = True,
        )
        labels = clusteror.fit_predict(sim_matrix)
        clusters = dict()
        for label_idx, c in enumerate(labels):
            if c < 0:
                continue
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(label_idx)
        max_size = max([len(v) for (k, v) in clusters.items()])
        max_c_name = -1
        for k in clusters.keys():
            if len(clusters[k]) == max_size: 
                max_c_name = k
        main_cluster = clusters[max_c_name]
        if len(main_cluster) <= 2:
            return [None] * len(sentences)
        featured_embs = [all_embeddings[idx] for idx in main_cluster]
        scores = compute_voting_score(featured_embs)
        final_scores = [None] * len(sentences)
        for idx, c in enumerate(main_cluster):
            final_scores[c] = scores[idx]
        return final_scores
    return [None] * len(sentences)