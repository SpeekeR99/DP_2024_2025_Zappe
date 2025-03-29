import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.anomaly_detection.data.dataloader import load_data, load_data_reduced_dimensions
from src.anomaly_detection.data.sequences import create_sequences, undo_sequences
from src.anomaly_detection.models.training import train_torch_model
from src.anomaly_detection.data.result_transform import transform_ys
from src.anomaly_detection.data.results_file_io import store_results, load_results
from src.anomaly_detection.analysis.visuals import plot_anomalies, plot_eval_res
from src.anomaly_detection.utils import WANTED_FEATURES, RANDOM_SEED_FOR_REPRODUCIBILITY, device


class BaseAutoencoder(nn.Module):
    """
    Base class for Autoencoders
    Defines fit and decision_function methods for sklearn-style usage
    This is needed for the evaluation, which is created for sklearn-style models
    ** This class is not meant to be used directly, but to be inherited from **
    """
    def __init__(self):
        """
        Constructor
        """
        super(BaseAutoencoder, self).__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Output data
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, train_loader, test_loader, num_epochs=10, lr=1e-3, patience=10, log=True):
        """
        Train the model
        This function exists mostly just because of the evaluation, which is created for sklearn-style models
        :param train_loader: DataLoader with training data
        :param test_loader: DataLoader with test data
        :param num_epochs: Number of epochs
        :param lr: Learning rate
        :param patience: Number of epochs without improvement before early stopping
        :param log: Log the epochs and losses
        """
        self.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Every 10 epochs, decrease the learning rate by 0.1 of its current value (gamma=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        best_loss = float("inf")
        epochs_without_improvement = 0
        epochs_without_much_change = 0
        best_model_state = self.state_dict()
        val_loss_delta_thresh = 0.01 if self.__class__.__name__ == "TransformerAutoencoder" else 1e-5

        train_loss = 0.0
        val_loss = 0.0
        last_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0

            for data_batch in train_loader:
                data_batch = data_batch.to(device)
                optimizer.zero_grad()

                # Forward pass
                output = self(data_batch)
                loss = criterion(output, data_batch)

                # Backward pass
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                # Optimize
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            if log:
                wandb.log({"train_loss": train_loss})
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}\n\tTraining Loss: {train_loss}")

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data_batch in test_loader:
                    data_batch = data_batch.to(device)
                    output = self(data_batch)
                    loss = criterion(output, data_batch)
                    val_loss += loss.item()

            val_loss /= len(test_loader)

            if log:
                wandb.log({"val_loss": val_loss})
                if (epoch + 1) % 10 == 0:
                    print(f"\tValidation Loss: {val_loss}")

            # Early stopping
            if np.abs(val_loss - last_val_loss) < val_loss_delta_thresh:
                epochs_without_much_change += 1

                if epochs_without_much_change >= patience // 3:
                    if log:
                        print(f"Early stopping triggered after {epoch + 1} epochs (No improvement)")
                    break
            else:
                epochs_without_much_change = 0
            last_val_loss = val_loss

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = self.state_dict()
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    if log:
                        print(f"Early stopping triggered after {epoch + 1} epochs (Possible overfitting)")
                    self.load_state_dict(best_model_state)
                    break

            lr_scheduler.step()

        if log:
            wandb.log({"train_loss_final": train_loss, "val_loss_final": val_loss})

    def score_samples(self, x):
        """
        Returns the reconstruction error for input x
        This function exists mostly just because of the evaluation, which is created for sklearn-style models
        :param x: Input data
        :return: Reconstruction error
        """
        with torch.no_grad():
            x_reconstructed = self.forward(x)
            error = torch.mean((x - x_reconstructed) ** 2, dim=1)
        return error.cpu().numpy()

    def decision_function(self, x, contamination=0.01, y_scores=None):
        """
        Returns the decision function, which is basically the same a score_samples, just offset
        so that the 0 is at the decision boundary (which is driven by the contamination parameter)
        Basically, we want to match the Scikit-implementation, where lower score is more abnormal,
        bigger score is more normal (our score is loss, so it's the opposite!)
        We basically have to take -score_samples(x) + offset, where offset is the score at the contamination quantile
        :param x: Input data
        :param contamination: Contamination parameter
        :param y_scores: Y scores (if already computed, why not use it)
        :return: Decision function
        """
        if y_scores is None:
            y_scores = self.score_samples(x)

        threshold = len(y_scores) * (1 - contamination)
        offset = np.sort(y_scores)[int(threshold)]

        return -y_scores + offset

    def save_model(self, path):
        """
        Save the model
        :param path: Path to save the model
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Load the model
        :param path: Path to load the model
        """
        self.load_state_dict(torch.load(path))


class FFNNAutoencoder(BaseAutoencoder):
    """
    Feedforward Neural Network Autoencoder
    """
    def __init__(self, input_size, latent_space_size):
        """
        Constructor
        :param input_size: Dimensionality of the input data
        :param latent_space_size: Dimensionality of the latent space
        """
        super(FFNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, (input_size + latent_space_size) // 2),
            nn.ReLU(),
            nn.Linear((input_size + latent_space_size) // 2, latent_space_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_size, (input_size + latent_space_size) // 2),
            nn.ReLU(),
            nn.Linear((input_size + latent_space_size) // 2, input_size),
            nn.Sigmoid()
        )

    def score_samples(self, x):
        """
        Returns the reconstruction error for input x
        This function exists mostly just because of the evaluation, which is created for sklearn-style models
        :param x: Input data
        :return: Reconstruction error
        """
        with torch.no_grad():
            x = x.view(x.size(0), -1)  # Ensure flattened input
            x_reconstructed = self.forward(x)
            error = torch.mean((x - x_reconstructed) ** 2, dim=1)
        return error.cpu().numpy()


class CNNAutoencoder(BaseAutoencoder):
    """
    Convolutional Neural Network Autoencoder
    """
    def __init__(self, input_size, latent_space_size):
        """
        Constructor
        :param input_size: Dimensionality of the input data
        :param latent_space_size: Dimensionality of the latent space
        """
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, latent_space_size // 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(latent_space_size // 2, latent_space_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_space_size, latent_space_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(latent_space_size // 2, input_size, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensure output is in [0, 1] if normalized
        )


class PositionalEncoding(nn.Module):
    """
    Since Transformers lack inherent positional information, we'll use sinusoidal positional encoding
    """
    def __init__(self, d_model, max_len=5000):
        """
        Constructor
        :param d_model: Dimensionality of the model
        :param max_len: Maximum length of the sequence
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Output data
        """
        return x + self.pe[:, :x.size(1)]


class TransformerAutoencoder(BaseAutoencoder):
    """
    Transformer Autoencoder
    """
    def __init__(self, input_size, seq_len=300, d_model=64, num_layers=4, num_heads=8):
        """
        Constructor
        :param input_size: Input size
        :param seq_len: Sequence length
        :param d_model: Model dimensionality
        :param num_layers: Number of layers
        :param num_heads: Number of heads
        """
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model

        # Input Embedding
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        # Transformer Encoder & Decoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, input_size)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Output data
        """
        # x shape: (batch_Size, features, seq_len)
        x = x.permute(0, 2, 1)  # Change to (batch_size, seq_len, features)

        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Encoder
        memory = self.encoder(x)

        # Decoder
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        output = self.decoder(x, memory, tgt_mask=tgt_mask)
        # output = self.decoder(x, memory)
        output = self.output_layer(output)

        output = output.permute(0, 2, 1)  # Change back to (batch_size, features, seq_len)
        return output


def main(config, data_file_info):
    """
    Main function
    :param config: Configuration of the model
    :param data_file_info: Information about the data file
    """
    # Load the data file information
    DATE = data_file_info["date"]
    MARKET_SEGMENT_ID = data_file_info["market_segment_id"]
    SECURITY_ID = data_file_info["security_id"]

    # Load the config
    model_type = config["model_type"]
    num_epochs = config["epochs"]
    kfolds = config["kfolds"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    seq_len = config["seq_len"]
    latent_dim = config["latent_dim"]

    # Load the data
    print("Loading the data...")
    data = load_data_reduced_dimensions(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # Take smaller subset of the data (for local computer speed purposes)
    # data = data.head(1000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()
    num_features = data_numpy.shape[1]

    # Transform data to PyTorch tensors and normalize the data
    data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
    data_tensor = (data_tensor - data_tensor.mean(dim=0)) / data_tensor.std(dim=0)  # Normalize the datas
    if model_type == "ffnn":  # FFNN does not need sequences
        data_tensor = data_tensor.to(device)
    else:  # CNN/Transformer needs sequences
        data_tensor = create_sequences(data_tensor, seq_len=seq_len).to(device)
    data_loader = DataLoader(data_tensor, batch_size=batch_size)

    # Initialize the model
    print("Initializing the model...")
    if model_type == "ffnn":
        model = FFNNAutoencoder(input_size=num_features, latent_space_size=latent_dim).to(device)
    elif model_type == "cnn":
        model = CNNAutoencoder(input_size=num_features, latent_space_size=latent_dim).to(device)
    elif model_type == "transformer":
        model = TransformerAutoencoder(input_size=num_features, seq_len=seq_len, d_model=32, num_layers=2, num_heads=4).to(device)

    # Train the model
    print("Training the model...")
    y_scores, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax = train_torch_model(model, data_loader, config, num_epochs=num_epochs, lr=lr, kfolds=kfolds, eval=True)
    # y_scores = train_torch_model(model, data_loader, config, num_epochs=num_epochs, lr=lr, kfolds=kfolds, eval=False)

    # !!! ----------------------------------- Only relevant for CNN/Transformer ------------------------------------ !!!
    if isinstance(model, CNNAutoencoder) or isinstance(model, TransformerAutoencoder):
        # Remake results from sequences to original shapes (data_len + 1, seq_len) -> (data_len,)
        # Aka take the it as a sliding window, so there's 300 predicitions for each data point (except the first 299 points)
        # 1, 2, 3, ... , 299, 300, 300, 300, ... , 300
        # Practically undo the def create_sequences() function
        y_scores = undo_sequences(torch.tensor(y_scores), seq_len=seq_len)
    # !!! ----------------------------------- Only relevant for CNN/Transformer ------------------------------------ !!!

    y_pred, anomaly_proba = transform_ys(y_scores, contamination=0.01, lower_is_better=True)

    # Dump the raw results to results folder
    store_results(DATE, MARKET_SEGMENT_ID, SECURITY_ID, config, y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax)

    # # Load results (just for reassurance that the function works and that the results are stored correctly)
    # y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax = load_results(DATE, MARKET_SEGMENT_ID, SECURITY_ID, config)
    #
    # # Prepare data for plots
    # print("Plotting the results...")
    # time_idx = data.columns.get_loc("Time")
    # indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    # if model_type == "ffnn":
    #     model_names = ["FFNN Autoencoder"]
    #     short_model_names = ["FFNNAE"]
    # elif model_type == "cnn":
    #     model_names = ["CNN Autoencoder"]
    #     short_model_names = ["CNNAE"]
    # elif model_type == "transformer":
    #     model_names = ["Transformer Autoencoder"]
    #     short_model_names = ["TAE"]
    # em_vals = [em_val]
    # mv_vals = [mv_val]
    # em_curves = [em_curve]
    # mv_curves = [mv_curve]
    #
    # # Plot the evaluation results
    # plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)
    #
    # # Plot the anomalies
    # plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names[0], short_model_names[0], data_numpy, time_idx, indcs, y_pred, anomaly_proba, WANTED_FEATURES[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--market_id", type=str, default="XEUR")
    parser.add_argument("--date", type=str, default="20191202")
    parser.add_argument("--market_segment_id", type=str, default="688")
    parser.add_argument("--security_id", type=str, default="4128839")

    parser.add_argument("--model_type", type=str, default="cnn")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=4)

    args = parser.parse_args()

    data_file_info = {
        "market_id": args.market_id,
        "date": args.date,
        "market_segment_id": args.market_segment_id,
        "security_id": args.security_id
    }

    config = {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "kfolds": args.kfolds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "latent_dim": args.latent_dim
    }

    # Fixation of all the random seeds (for reproducibility)
    torch.manual_seed(RANDOM_SEED_FOR_REPRODUCIBILITY)
    np.random.seed(RANDOM_SEED_FOR_REPRODUCIBILITY)

    main(config, data_file_info)
