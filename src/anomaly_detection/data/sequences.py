import torch


def create_sequences(data, seq_len=300):
    """
    Create sequences from the data
    :param data: Data tensor
    :param seq_len: Sequence length
    :return: Sequences for CNN/Transformer models
    """
    # Create the sliding windows (output shape: [num_samples, seq_len, num_features])
    windows = torch.stack([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])
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
    :return: Scores in the right shape
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

    return reconstructed_data.cpu().numpy()