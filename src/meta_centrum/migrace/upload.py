import sys
import boto3
from boto3.s3.transfer import TransferConfig
import time

with open("credentials", "r") as fp:
    lines = fp.readlines()
    access_key = lines[0].strip()
    secret_key = lines[1].strip()
    endpoint_url = lines[2].strip()
    region_name = lines[3].strip()

mb = 1024 ** 2
config = TransferConfig(multipart_threshold=50 * mb, multipart_chunksize=50 * mb, use_threads=True)

s3 = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    endpoint_url=endpoint_url,
    region_name=region_name
)

bucket_name = sys.argv[1]
local_path = sys.argv[2]
s3_path = sys.argv[3]

s3.upload_file(local_path, bucket_name, s3_path, Config=config)
