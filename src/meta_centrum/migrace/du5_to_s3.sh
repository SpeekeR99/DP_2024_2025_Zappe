#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=4:mem=64gb:scratch_local=100gb:cl_halmir=True
#PBS -N du5_to_s3

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Variables                                                                    |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
DATADIR=/storage/plzen1/home/zapped99/s3
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

BUCKET="data.raw"

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Preparation                                                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
cd $SCRATCHDIR
cp $DATADIR/migrace/upload.py $SCRATCHDIR/upload.py
cp $DATADIR/migrace/download.py $SCRATCHDIR/download.py
cp $DATADIR/migrace/credentials $SCRATCHDIR/credentials

cp -r /storage/plzen1/home/zapped99/.ssh ~/.ssh
cp -r /storage/plzen1/home/zapped99/.ssh $SCRATCHDIR/.ssh

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Copying files from DU5 to Scratch                                            |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
echo -e "`date`\tcopying file $FILE from $DIR" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

scp zapped99@ssh.du5.cesnet.cz:$DIR${FILE}_0 .
test -e ${FILE}_0 || { echo >&2 "File ${FILE}_0 does not exist!"; echo "${FILE}" >> $DATADIR/migrace/failed.txt; exit 1; }
cat ${FILE}_0

scp zapped99@ssh.du5.cesnet.cz:$DIR$FILE .
test -e $FILE || { echo >&2 "File $FILE does not exist!"; echo "${FILE}" >> $DATADIR/migrace/failed.txt; exit 1; }
md5sum $FILE >> md5sum.txt
cat md5sum.txt

echo -e "`date`\tscp completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

diff ${FILE}_0 md5sum.txt || { echo "MD5 sums do not match!"; echo "${FILE}" >> $DATADIR/migrace/failed.txt; exit 2; }
echo -e "`date`\tMD5 sums match" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Uploading from Scratch to S3                                                 |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
echo -e "`date`\tuploading file $FILE to S3" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
python3 upload.py $BUCKET $FILE $FILE
echo -e "`date`\tupload to S3 completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Donwloading from S3 (Verification)                                           |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
rm $FILE
rm md5sum.txt

echo -e "`date`\tdownloading file $FILE from S3" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
python3 download.py $BUCKET $FILE $FILE
echo -e "`date`\tdownload from S3 completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

test -e $FILE || { echo >&2 "File $FILE does not exist!"; echo "${FILE}" >> $DATADIR/migrace/failed.txt; exit 1; }
md5sum $FILE >> md5sum.txt
cat md5sum.txt

diff ${FILE}_0 md5sum.txt || { echo "MD5 sums do not match!"; echo "${FILE}" >> $DATADIR/migrace/failed.txt; exit 2; }
echo -e "`date`\tMD5 sums match" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Cleanup                                                                      |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
clean_scratch
