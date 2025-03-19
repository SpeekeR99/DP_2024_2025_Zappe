import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.anomaly_detection.dataloader import load_data
from src.anomaly_detection.training import train_torch_model
from src.anomaly_detection.visuals import plot_anomalies, plot_eval_res
from src.anomaly_detection.utils import DATE, MARKET_SEGMENT_ID, SECURITY_ID, WANTED_FEATURES


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def fit(self, train_loader, num_epochs=10, lr=1e-3):
        """
        Train the model
        This function exists mostly just because of the evaluation, which is created for sklearn-style models
        :param train_loader: DataLoader with training data
        :param num_epochs: Number of epochs
        :param lr: Learning rate
        """
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for data_batch in train_loader:
                data_batch = data_batch.to(device)
                optimizer.zero_grad()

                # Forward pass
                output = self(data_batch)
                loss = criterion(output, data_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

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

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Output data
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    """
    Main function
    """
    # Load the data
    print("Loading the data...")
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # Take smaller subset of the data (for local computer speed purposes)
    data = data.head(10000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Transform data to PyTorch tensors and normalize the data
    data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
    data_tensor = (data_tensor - data_tensor.mean(dim=0)) / data_tensor.std(dim=0)  # Normalize the data
    batch_size = 32
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

    # Initialize the model
    print("Initializing the model...")
    latent_dimensions = 4
    ffnn_model = FFNNAutoencoder(input_size=data_tensor.shape[1], latent_space_size=latent_dimensions).to(device)

    # Train the model
    print("Training the model...")
    num_epochs = 10
    lr = 1e-5
    kfolds = 5

    y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax = train_torch_model(ffnn_model, data_loader, num_epochs=num_epochs, lr=lr, kfolds=kfolds, eval=True)

    # Prepare data for plots
    print("Plotting the results...")
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    model_names = ["FFNN Autoencoder"]
    short_model_names = ["FFNNAE"]
    em_vals = [em_val]
    mv_vals = [mv_val]
    em_curves = [em_curve]
    mv_curves = [mv_curve]

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "FFNN Autoencoder", "FFNNAE", data_numpy, time_idx, indcs, y_pred, anomaly_proba, WANTED_FEATURES[1:])


if __name__ == "__main__":
    main()
