#!/bin/bash

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Variables                                                                    |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
# Check the number of arguments
if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "Usage: $0 market_id date market_segment_id security_id"
    echo "Example: $0 XEUR 20210319 688 5578483"
    exit 1
fi

# Assign the arguments to variables
market_id=$1
date=$2
market_segment_id=$3
security_id=$4

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Grid Search                                                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘

model_ffnn="ffnn"
model_cnn="cnn"
model_transformer="transformer"
# General HP's
epochs=500
kfolds=5
# Model specific HP's
batch_sizes=(16 32 64)
lrs=(0.01 0.001 0.0001)
seq_lens=(32 64 128)  # FFNN does not have sequences
mock_seq_len=64  # For FFNN
latent_dims=(4 6 8)  # Transformer does not have latent dim
mock_latent_dim=4  # For Transformer

# FFNN
for batch_size in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
        for latent_dim in "${latent_dims[@]}"; do
            # Run the training on a given market, date, market_segment_id, security_id with the given model and its params
            echo "Running FFNN with batch_size=$batch_size, lr=$lr, latent_dim=$latent_dim"
            qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_ffnn",epochs="$epochs",kfolds="$kfolds",batch_size="$batch_size",lr="$lr",seq_len="$mock_seq_len",latent_dim="$latent_dim" torch_training_gpu.sh
        done
    done
done

# CNN
for batch_size in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
        for latent_dim in "${latent_dims[@]}"; do
            for seq_len in "${seq_lens[@]}"; do
                # Run the training on a given market, date, market_segment_id, security_id with the given model and its params
                echo "Running CNN with batch_size=$batch_size, lr=$lr, latent_dim=$latent_dim, seq_len=$seq_len"
                qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_cnn",epochs="$epochs",kfolds="$kfolds",batch_size="$batch_size",lr="$lr",seq_len="$seq_len",latent_dim="$latent_dim" torch_training_gpu.sh
            done
        done
    done
done

# Transformer
for batch_size in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
        for seq_len in "${seq_lens[@]}"; do
            # Run the training on a given market, date, market_segment_id, security_id with the given model and its params
            echo "Running Transformer with batch_size=$batch_size, lr=$lr, seq_len=$seq_len"
            qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_transformer",epochs="$epochs",kfolds="$kfolds",batch_size="$batch_size",lr="$lr",seq_len="$seq_len",latent_dim="$mock_latent_dim" torch_training_gpu.sh
        done
    done
done

# CPU
# qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_type",epochs="$epochs",kfolds="$kfolds",batch_size="$batch_size",lr="$lr",seq_len="$seq_len",latent_dim="$latent_dim" torch_training.sh
# GPU
# qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_type",epochs="$epochs",kfolds="$kfolds",batch_size="$batch_size",lr="$lr",seq_len="$seq_len",latent_dim="$latent_dim" torch_training_gpu.sh
