import torch
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config

