#!/bin/bash

DATADIR=/storage/plzen1/home/zapped99/s3

alias aws='aws --endpoint-url https://s3.cl2.du.cesnet.cz'
export S3_ENDPOINT_URL="https://s3.cl2.du.cesnet.cz"
export AWS_SHARED_CREDENTIALS_FILE="$DATADIR/.aws/credentials"
export AWS_CONFIG_FILE="$DATADIR/.aws/config"

BUCKET="data"
DAY="2021-01-04"

aws s3 ls ${BUCKET}/${DAY}/all/ --endpoint-url https://s3.cl2.du.cesnet.cz | grep .csv | awk '{print $4}'
