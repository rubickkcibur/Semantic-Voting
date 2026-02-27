import requests

model_name = "unsup-simcse-bert-base-uncased"

json_data = {
    "model_path": model_name,
}

response = requests.post("http://localhost:8111/load_model", json=json_data)
print(response.json())


cands = ["The cat is on the table.", "A cat is sitting on the table.", "The dog is in the garden.", "There is a cat on the table.", "The cat is on the roof.", "A dog is playing in the garden.", "The cat is sleeping on the table.", "The dog is barking in the garden.", "The cat is on the floor.", "A dog is running in the garden."]

json_data = {
    "candidates": cands,
    "clustering": "whole",
    "batch_size": 16,
    "HDBSCAN_min_cluster_size": 5,
    "HDBSCAN_min_samples": 2,
}

response = requests.post("http://localhost:8111/predict", json=json_data)
print(response.json())

