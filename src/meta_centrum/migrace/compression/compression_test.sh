#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=16:mem=256gb:scratch_local=512gb:cl_halmir=True
#PBS -N compression

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Variables                                                                    |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
DATADIR=/storage/plzen1/home/zapped99/s3
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/migrace/compression/jobs_info.$PBS_JOBID.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

BUCKET="data"

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Preparation                                                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
cd $SCRATCHDIR
cp $DATADIR/migrace/upload.py $SCRATCHDIR/upload.py
cp $DATADIR/migrace/download.py $SCRATCHDIR/download.py
cp $DATADIR/migrace/credentials $SCRATCHDIR/credentials
cp $DATADIR/migrace/compression/compression.py $SCRATCHDIR/compression.py

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Venv                                                                         |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
module add py-virtualenv
virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install boto3
pip install zstandard

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Download from S3 to Scratch                                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
echo -e "`date`\tdownloading file $FILE from S3" >> $DATADIR/migrace/compression/jobs_info.$PBS_JOBID.txt
python3 download.py $BUCKET 2021-01-04/all/${FILE} ${FILE}
echo -e "`date`\tdownloading from S3 completed" >> $DATADIR/migrace/compression/jobs_info.$PBS_JOBID.txt

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Run the compression test                                                     |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
echo -e "`date`\trunning compression test" >> $DATADIR/migrace/compression/jobs_info.$PBS_JOBID.txt
python3 compression.py ${FILE} >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
echo -e "`date`\tcompression test completed" >> $DATADIR/migrace/compression/jobs_info.$PBS_JOBID.txt

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Cleanup                                                                      |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
clean_scratch
