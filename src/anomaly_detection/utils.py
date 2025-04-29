import torch

# Constants used across the project
# Features that are to be kept in the dataset for training etc.
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

# Weights and Biases configuration
WANDB_ENTITY = "thebigbook"
WANDB_PROJECT = "anomaly_detection_DP_2025_zappe_dominik"

# Default random seed
RANDOM_SEED_FOR_REPRODUCIBILITY = 42

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
