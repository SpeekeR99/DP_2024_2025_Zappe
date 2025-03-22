import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from src.anomaly_detection.models.autoencoder import FFNNAutoencoder, CNNAutoencoder, TransformerAutoencoder, create_sequences, undo_sequences

from src.anomaly_detection.dataloader import load_data
from src.anomaly_detection.training import train_model, train_torch_model
from src.anomaly_detection.visuals import plot_anomalies, plot_eval_res
from src.anomaly_detection.utils import DATE, MARKET_SEGMENT_ID, SECURITY_ID, WANTED_FEATURES


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Main function
    """
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
    data_tensor = (data_tensor - data_tensor.mean(dim=0)) / data_tensor.std(dim=0)  # Normalize the data
    seq_len = 300
    data_tensor_seq = create_sequences(data_tensor, seq_len=seq_len, transpose=False)
    data_tensor_seq_transposed = create_sequences(data_tensor, seq_len=seq_len, transpose=True)
    batch_size = 32
    data_loader = DataLoader(data_tensor, batch_size=batch_size)
    data_loader_seq = DataLoader(data_tensor_seq, batch_size=batch_size)
    data_loader_seq_transposed = DataLoader(data_tensor_seq_transposed, batch_size=batch_size)

    # Initialize the models
    print("Initializing the models...")
    model_if = IsolationForest(contamination=0.01)
    model_ocsvm = OneClassSVM(kernel="rbf", gamma="scale")
    model_lof = LocalOutlierFactor(n_neighbors=32, contamination=0.01, novelty=True)
    latent_dimensions = 4
    model_ffnn_ae = FFNNAutoencoder(input_size=num_features, latent_space_size=latent_dimensions).to(device)
    model_cnn_ae = CNNAutoencoder(input_size=num_features, latent_space_size=latent_dimensions).to(device)
    model_t_ae = TransformerAutoencoder(input_size=num_features, seq_len=seq_len, d_model=64, num_layers=4, num_heads=8).to(device)

    # Train the models
    print("Training the models...")
    num_epochs = 10
    lr_e_5 = 1e-5
    lr_e_4 = 1e-4
    kfolds = 5

    print("Isolation Forest")
    y_pred_if, y_scores_if, anomaly_proba_if, em_val_if, mv_val_if, em_curve_if, mv_curve_if, t_if, axis_alpha_if, amax_if = train_model(model_if, data_numpy, kfolds=kfolds, eval=True)
    print("One-Class SVM")
    y_pred_ocsvm, y_scores_ocsvm, anomaly_proba_ocsvm, em_val_ocsvm, mv_val_ocsvm, em_curve_ocsvm, mv_curve_ocsvm, t_ocsvm, axis_alpha_ocsvm, amax_ocsvm = train_model(model_ocsvm, data_numpy, kfolds=kfolds, eval=True)
    print("Local Outlier Factor")
    y_pred_lof, y_scores_lof, anomaly_proba_lof, em_val_lof, mv_val_lof, em_curve_lof, mv_curve_lof, t_lof, axis_alpha_lof, amax_lof = train_model(model_lof, data_numpy, kfolds=kfolds, eval=True)
    print("FFNN Autoencoder")
    y_pred_ffnnae, y_scores_ffnnae, anomaly_proba_ffnnae, em_val_ffnnae, mv_val_ffnnae, em_curve_ffnnae, mv_curve_ffnnae, t_ffnnae, axis_alpha_ffnnae, amax_ffnnae = train_torch_model(model_ffnn_ae, data_loader, num_epochs=num_epochs, lr=lr_e_5, kfolds=kfolds, eval=True)
    print("CNN Autoencoder")
    y_pred_cnnae, y_scores_cnnae, anomaly_proba_cnnae, em_val_cnnae, mv_val_cnnae, em_curve_cnnae, mv_curve_cnnae, t_cnnae, axis_alpha_cnnae, amax_cnnae = train_torch_model(model_cnn_ae, data_loader_seq_transposed, num_epochs=num_epochs, lr=lr_e_4, kfolds=kfolds, eval=True)
    print("Transformer Autoencoder")
    y_pred_tae, y_scores_tae, anomaly_proba_tae, em_val_tae, mv_val_tae, em_curve_tae, mv_curve_tae, t_tae, axis_alpha_tae, amax_tae = train_torch_model(model_t_ae, data_loader_seq, num_epochs=num_epochs, lr=lr_e_4, kfolds=kfolds, eval=True)

    # Transform sequences back to original shapes
    y_pred_cnnae, y_scores_cnnae, anomaly_proba_cnnae = undo_sequences(y_scores_cnnae, seq_len=seq_len)
    y_pred_tae, y_scores_tae, anomaly_proba_tae = undo_sequences(y_scores_tae, seq_len=seq_len)

    # Prepare data for plots
    print("Plotting the results...")
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    model_names = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "FFNN Autoencoder", "CNN Autoencoder", "Transformer Autoencoder"]
    short_model_names = ["IF", "OCSVM", "LOF", "FFNNAE", "CNNAE", "TAE"]
    y_preds = [y_pred_if, y_pred_ocsvm, y_pred_lof, y_pred_ffnnae, y_pred_cnnae, y_pred_tae]
    anomaly_probas = [anomaly_proba_if, anomaly_proba_ocsvm, anomaly_proba_lof, anomaly_proba_ffnnae, anomaly_proba_cnnae, anomaly_proba_tae]
    em_vals = [em_val_if, em_val_ocsvm, em_val_lof, em_val_ffnnae, em_val_cnnae, em_val_tae]
    mv_vals = [mv_val_if, mv_val_ocsvm, mv_val_lof, mv_val_ffnnae, mv_val_cnnae, mv_val_tae]
    em_curves = [em_curve_if, em_curve_ocsvm, em_curve_lof, em_curve_ffnnae, em_curve_cnnae, em_curve_tae]
    mv_curves = [mv_curve_if, mv_curve_ocsvm, mv_curve_lof, mv_curve_ffnnae, mv_curve_cnnae, mv_curve_tae]
    t = t_if
    axis_alpha = axis_alpha_if
    amax = max(amax_if, amax_ocsvm, amax_lof, amax_ffnnae, amax_cnnae, amax_tae)

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    for model_name, short_model_name, y_pred, anomaly_proba in zip(model_names, short_model_names, y_preds, anomaly_probas):
        plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_name, short_model_name, data_numpy, time_idx, indcs, y_pred, anomaly_proba, WANTED_FEATURES[1:])


if __name__ == "__main__":
    main()
