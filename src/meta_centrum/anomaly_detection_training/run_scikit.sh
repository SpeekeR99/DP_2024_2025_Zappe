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

model_if="if"
model_ocsvm="ocsvm"
model_lof="lof"
# General HP's
kfolds=5
# Model specific HP's
n_estimatorss=(50, 100, 200)
mock_n_estimators=100
max_sampless=("auto", "1024", "0.1")
mock_max_samples="auto"
max_featuress=(0.5 0.75 1.0)
mock_max_features=1.0
gammas=("scale", "auto")
mock_gamma="auto"
n_neigborss=(16, 32)
mock_n_neighbors=32

# Isolation Forest
for n_estimators in "${n_estimatorss[@]}"; do
    for max_samples in "${max_sampless[@]}"; do
        for max_features in "${max_featuress[@]}"; do
            # Run the training on a given market, date, market_segment_id, security_id with the given model and its params
            qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_if",kfolds="$kfolds",n_estimators="$n_estimators",max_samples="$max_samples",max_features="$max_features",gamma="$mock_gamma",n_neighbors="$mock_n_neighbors" scikit_training.sh
        done
    done
done

# One-Class SVM
for gamma in "${gammas[@]}"; do
      # Run the training on a given market, date, market_segment_id, security_id with the given model and its params
      qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_ocsvm",kfolds="$kfolds",n_estimators="$mock_n_estimators",max_samples="$mock_max_samples",max_features="$mock_max_features",gamma="$gamma",n_neighbors="$mock_n_neighbors" scikit_training.sh
done

# Local Outlier Factor
for n_neighbors in "${n_neigborss[@]}"; do
    # Run the training on a given market, date, market_segment_id, security_id with the given model and its params
    qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_lof",kfolds="$kfolds",n_estimators="$mock_n_estimators",max_samples="$mock_max_samples",max_features="$mock_max_features",gamma="$mock_gamma",n_neighbors="$n_neighbors" scikit_training.sh
done

# Run the training on a given market, date, market_segment_id, security_id with the given model and its params
# qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_type",kfolds="$kfolds",n_estimators="$n_estimators",max_samples="$max_samples",max_features="$max_features",gamma="$gamma",n_neighbors="$n_neighbors" scikit_training.sh
