#!/bin/bash

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Variables                                                                    |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
# Check the number of arguments
if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "Usage: $0 market_id date market_segment_id security_id"
    echo "Example: $0 XEUR 20191202 688 4128839"
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

# TODO: Grid search for loops

# These variables would be set by the grid search in the for loops normally, hardcoded for now
model_type="if"
kfolds=5

# Run the training on a given market, date, market_segment_id, security_id with the given model and its params
qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_type",kfolds="$kfolds" scikit_training.sh
