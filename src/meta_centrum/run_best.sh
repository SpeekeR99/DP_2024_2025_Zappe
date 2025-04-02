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
# |            Run best models                                                              |
# └─────────────────────────────────────────────────────────────────────────────────────────┘

model_if="if"
model_ocsvm="ocsvm"
model_lof="lof"
model_ffnn="ffnn"
model_cnn="cnn"
model_transformer="transformer"
# General HP's
epochs=500
kfolds=5
# Model specific HP's (Grid Search 1 results -> Normal Dimensionality)
if_n_estimators=100
if_max_samples=0.1
if_max_features=0.5
ocsvm_gamma="scale"
lof_n_neighbors=32
ffnn_batch_size=32
ffnn_lr=0.001
ffnn_latent_dim=8
cnn_batch_size=32
cnn_lr=0.001
cnn_latent_dim=8
cnn_seq_len=64
transformer_batch_size=64
transformer_lr=0.0001
transformer_seq_len=32
# Model specific HP's (Grid Search 2 results -> Reduced Dimensionality)
# if_n_estimators=100
# if_max_samples=0.1
# if_max_features=0.5
# ocsvm_gamma="scale"
# lof_n_neighbors=32
# ffnn_batch_size=16
# ffnn_lr=0.001
# ffnn_latent_dim=4
# cnn_batch_size=32
# cnn_lr=0.001
# cnn_latent_dim=4
# cnn_seq_len=64
# transformer_batch_size=32
# transformer_lr=0.0001
# transformer_seq_len=64
random_seeds=(0 1 2 3 4 5 6 7 8 9)

# Run the best models
for seed in "${random_seeds[@]}"; do
    echo "Running Best Models with seed=$seed"
    qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_if",kfolds="$kfolds",n_estimators="$if_n_estimators",max_samples="$if_max_samples",max_features="$if_max_features",gamma="$ocsvm_gamma",n_neighbors="$lof_n_neighbors",seed="$seed" scikit_training.sh
    qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_ocsvm",kfolds="$kfolds",n_estimators="$if_n_estimators",max_samples="$if_max_samples",max_features="$if_max_features",gamma="$ocsvm_gamma",n_neighbors="$lof_n_neighbors",seed="$seed" scikit_training.sh
    qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_lof",kfolds="$kfolds",n_estimators="$if_n_estimators",max_samples="$if_max_samples",max_features="$if_max_features",gamma="$ocsvm_gamma",n_neighbors="$lof_n_neighbors",seed="$seed" scikit_training.sh
    qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_ffnn",epochs="$epochs",kfolds="$kfolds",batch_size="$ffnn_batch_size",lr="$ffnn_lr",seq_len="$cnn_seq_len",latent_dim="$ffnn_latent_dim",seed="$seed" torch_training_gpu.sh
    qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_cnn",epochs="$epochs",kfolds="$kfolds",batch_size="$cnn_batch_size",lr="$cnn_lr",seq_len="$cnn_seq_len",latent_dim="$cnn_latent_dim",seed="$seed" torch_training_gpu.sh
    qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_transformer",epochs="$epochs",kfolds="$kfolds",batch_size="$transformer_batch_size",lr="$transformer_lr",seq_len="$transformer_seq_len",latent_dim="$cnn_latent_dim",seed="$seed" torch_training_gpu.sh
done
