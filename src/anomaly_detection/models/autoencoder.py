import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.anomaly_detection.dataloader import load_data
from src.anomaly_detection.training import train_torch_model
from src.anomaly_detection.results_file_io import store_results, load_results
from src.anomaly_detection.visuals import plot_anomalies, plot_eval_res
from src.anomaly_detection.utils import DATE, MARKET_SEGMENT_ID, SECURITY_ID, WANTED_FEATURES, device


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
        best_model_state = self.state_dict()

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
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = self.state_dict()
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    if log:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                    self.load_state_dict(best_model_state)
                    break

            lr_scheduler.step()

    def decision_function(self, x):
        """
        Returns the reconstruction error for input x
        This function exists mostly just because of the evaluation, which is created for sklearn-style models
        :param x: Input data
        :return: Reconstruction error
        """
        with torch.no_grad():
            x = x.view(x.size(0), -1)  # Ensure flattened input
            x_reconstructed = self.forward(x)
            error = torch.mean((x - x_reconstructed) ** 2, dim=1)  # Compute MSE per sample
        return error.cpu().numpy()  # Convert to NumPy for sklearn-style usage


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

    def decision_function(self, x):
        """
        Returns the reconstruction error for input x
        This function exists mostly just because of the evaluation, which is created for sklearn-style models
        :param x: Input data
        :return: Reconstruction error
        """
        with torch.no_grad():
            x_reconstructed = self.forward(x)
            error = torch.mean((x - x_reconstructed) ** 2, dim=1)  # Compute MSE per sample
        return error.cpu().numpy()  # Convert to NumPy for sklearn-style usage


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
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        # x shape: (batch_size, features, seq_len)
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Encoder
        memory = self.encoder(x)

        # Decoder
        output = self.decoder(x, memory)
        output = self.output_layer(output)
        return output

    def decision_function(self, x):
        """
        Returns the reconstruction error for input x
        This function exists mostly just because of the evaluation, which is created for sklearn-style models
        :param x: Input data
        :return: Reconstruction error
        """
        with torch.no_grad():
            x_reconstructed = self.forward(x)
            error = torch.mean((x - x_reconstructed) ** 2, dim=2)
        return error.cpu().numpy()


def create_sequences(data, seq_len=300, transpose=False):
    """
    Create sequences from the data
    :param data: Data tensor
    :param seq_len: Sequence length
    :param transpose: Transpose the data
    :return: Sequences for CNN/Transformer models
    """
    # Create the sliding windows (output shape: [num_samples, seq_len, num_features])
    windows = torch.stack([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])
    # Funnily enough, the CNN expects the channels to be the second dimension and sequences to be the third
    # (batch_size, features, seq_len)
    # However, Transformer expects the sequences to be the second dimension and features to be the third
    # (batch_size, seq_len, features)
    # This little thing caused a lot of troubles later on in the code with evaluation
    if transpose:
        windows = windows.permute(0, 2, 1)
    return windows


def undo_sequences(data, seq_len=300):
    """
    Original expected result is (data_len,) (predictions - normal / anomaly)
    What we have in the data is (data_len - seq_len + 1, seq_len)
    Meaning we have (at average) 300 predictions per timestamp (taking context into account)
    (First seq_len - 1 timestamps have less predictions, as the sliding window moved)
    We need to take the first value as a value for the first timestamp (only one value)
    Second timestamp is in the first window and the second window -> average of that (two values)
    So forth up until seq_len, then every timestamp has seq_len values across seq_len windows
    :param data: Data tensor (y_scores expected)
    :param seq_len: Sequence length
    :return: Predictions, scores and anomaly probabilities in the right shapes
    """
    # Get the number of windows and the sequence length (300 in your case)
    num_windows, window_size = data.shape
    original_length = num_windows + seq_len - 1  # The original length is num_windows + seq_len - 1

    # Create an empty tensor to store the reconstructed data
    reconstructed_data = torch.zeros(original_length)

    # We will average over the overlapping windows
    for i in range(num_windows):
        for j in range(seq_len):
            reconstructed_data[i + j] += data[i, j]  # Add the value from the window to the appropriate timestamp

    # Divide by the number of windows that contribute to each timestamp
    for i in range(original_length):
        reconstructed_data[i] /= min(i + 1, seq_len, original_length - i)  # Normalize by the number of windows contributing to the value

    y_scores = reconstructed_data.cpu().numpy()
    y_pred = np.zeros_like(y_scores)

    # Calculate the y_pred based on the anomaly prediction of contamination (1 %)
    threshold = np.percentile(y_scores, 1)
    y_pred[y_scores < threshold] = -1
    y_pred[y_scores >= threshold] = 1
    # Calculate the anomaly probability
    y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    anomaly_proba = 1 - y_scores_norm  # The lower the original score, the higher "certainty" it is an anomaly

    return y_pred, y_scores, anomaly_proba


def main(config):
    """
    Main function
    """
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
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # Take smaller subset of the data (for local computer speed purposes)
    data = data.head(1000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()
    num_features = data_numpy.shape[1]

    # Transform data to PyTorch tensors and normalize the data
    data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
    data_tensor = (data_tensor - data_tensor.mean(dim=0)) / data_tensor.std(dim=0)  # Normalize the datas

    # Initialize the model
    print("Initializing the model...")
    if model_type == "ffnn":
        model = FFNNAutoencoder(input_size=num_features, latent_space_size=latent_dim).to(device)
    elif model_type == "cnn":
        model = CNNAutoencoder(input_size=num_features, latent_space_size=latent_dim).to(device)
    elif model_type == "transformer":
        model = TransformerAutoencoder(input_size=num_features, seq_len=seq_len, d_model=64, num_layers=4, num_heads=8).to(device)

    # Based on model, augment with sequences (or not)
    if isinstance(model, CNNAutoencoder) or isinstance(model, TransformerAutoencoder):
        data_tensor = create_sequences(data_tensor, seq_len=seq_len, transpose=isinstance(model, CNNAutoencoder))
    data_loader = DataLoader(data_tensor, batch_size=batch_size)

    # Train the model
    print("Training the model...")
    y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax = train_torch_model(model, data_loader, config, num_epochs=num_epochs, lr=lr, kfolds=kfolds, eval=True)

    # !!! ----------------------------------- Only relevant for CNN/Transformer ------------------------------------ !!!
    if isinstance(model, CNNAutoencoder) or isinstance(model, TransformerAutoencoder):
        # Remake results from sequences to original shapes (data_len + 1, seq_len) -> (data_len,)
        # Aka take the it as a sliding window, so there's 300 predicitions for each data point (except the first 299 points)
        # 1, 2, 3, ... , 299, 300, 300, 300, ... , 300
        # Practically undo the def create_sequences() function
        y_pred, y_scores, anomaly_proba = undo_sequences(torch.tensor(y_scores), seq_len=seq_len)
    # !!! ----------------------------------- Only relevant for CNN/Transformer ------------------------------------ !!!

    # Dump the raw results to results folder
    store_results(DATE, MARKET_SEGMENT_ID, SECURITY_ID, config, y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax)

    # Load results (just for reassurance that the function works and that the results are stored correctly)
    y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax = load_results(DATE, MARKET_SEGMENT_ID, SECURITY_ID, config)

    # Prepare data for plots
    print("Plotting the results...")
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    if model_type == "ffnn":
        model_names = ["FFNN Autoencoder"]
        short_model_names = ["FFNNAE"]
    elif model_type == "cnn":
        model_names = ["CNN Autoencoder"]
        short_model_names = ["CNNAE"]
    elif model_type == "transformer":
        model_names = ["Transformer Autoencoder"]
        short_model_names = ["TAE"]
    em_vals = [em_val]
    mv_vals = [mv_val]
    em_curves = [em_curve]
    mv_curves = [mv_curve]

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names[0], short_model_names[0], data_numpy, time_idx, indcs, y_pred, anomaly_proba, WANTED_FEATURES[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="cnn")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=4)

    args = parser.parse_args()

    config = {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "kfolds": args.kfolds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "latent_dim": args.latent_dim
    }

    main(config)
