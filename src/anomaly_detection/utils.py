import torch

WANTED_FEATURES = [
    "Time",
    "Ask Price 1",
    "Ask Volume 1",
    "Bid Price 1",
    "Bid Volume 1",
    "Imbalance Index",
    "Frequency of Incoming Messages",
    "Cancellations Rate",
    "High Quoting Activity",
    "Unbalanced Quoting",
    "Low Execution Probability",
    "Trades Oppose Quotes",
    "Cancels Oppose Trades"
]
WANDB_ENTITY = "thebigbook"
WANDB_PROJECT = "anomaly_detection_DP_2025_zappe_dominik"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
