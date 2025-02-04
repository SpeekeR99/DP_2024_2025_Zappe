#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=4:mem=160gb:scratch_local=320gb:cl_halmir=True
#PBS -N grep_awk

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Variables                                                                    |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
HOMEDIR=/storage/plzen1/home/zapped99
DATADIR=/storage/plzen1/home/zapped99/s3
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

BUCKET="data"
# Expected input environmental variables: DAY, SEC, SECID

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Preparation                                                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
cd $SCRATCHDIR
cp $DATADIR/migrace/upload.py $SCRATCHDIR/upload.py
cp $DATADIR/migrace/download.py $SCRATCHDIR/download.py
cp $DATADIR/migrace/credentials $SCRATCHDIR/credentials
cp $DATADIR/migrace/grep_awk/grepSecurityID.awk $SCRATCHDIR/grepSecurityID.awk
cp $DATADIR/migrace/grep_awk/merge.py $SCRATCHDIR/merge.py
cp $DATADIR/migrace/grep_awk/preprocess.py $SCRATCHDIR/preprocess.py
cp $DATADIR/migrace/grep_awk/reconstruction.py $SCRATCHDIR/reconstruction.py

alias aws='aws --endpoint-url https://s3.cl5.du.cesnet.cz'
export S3_ENDPOINT_URL="https://s3.cl5.du.cesnet.cz"
export AWS_SHARED_CREDENTIALS_FILE="$HOMEDIR/.aws/credentials"
export AWS_CONFIG_FILE="$HOMEDIR/.aws/config"

aws s3 ls ${BUCKET}/${DAY}/all/ --endpoint-url https://s3.cl5.du.cesnet.cz | grep .csv | awk '{print $4}' >> csv_files.txt
cat csv_files.txt
cat csv_files.txt >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
OUTDIR=${SEC}-${SECID}
mkdir ${OUTDIR}

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Venv                                                                         |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
module add py-virtualenv
virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install boto3
pip install pandas
pip install numpy

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Donwloading from S3 and processing with grep / awk                           |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
for FILE in $(cat csv_files.txt); do
    echo -e "`date`\tdownloading file $DAY/all/$FILE from S3" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
    python3 download.py $BUCKET ${DAY}/all/${FILE} $FILE
    echo -e "`date`\tdownloading from S3 completed" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt

    echo  -e "`date`\tprocessing file $FILE" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
    
    if [[ $(wc -l < ${FILE}) -ge 2 ]] ; then
		BASE=${FILE//.csv/}
		OUTFILE=${OUTDIR}/${BASE}_${SEC}_${SECID}.csv
		HEADER=`head -1 ${FILE}`
		if [[ ${HEADER,,} =~ "securityid" ]] ; then
			awk -v id=${SECID} -f grepSecurityID.awk ${FILE} > ${OUTFILE}
			MSGS=$(wc -l < ${FILE})
			((MSGS--))
			echo -e "`date`\t${OUTFILE} (${MSGS} messages)" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
		else
			echo -e "`date`\t${FILE} - no SecurityID, skipping." >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
		fi
	else
		echo -e "`date`\t${FILE} - only the header line, skipping." >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
	fi

	echo -e "`date`\tprocessing completed" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
	rm $FILE
done

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Merge and preprocess for visualization                                       |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
cp merge.py ${OUTDIR}
cp preprocess.py ${OUTDIR}
cp reconstruction.py ${OUTDIR}
cp upload.py ${OUTDIR}
cp credentials ${OUTDIR}
cd ${OUTDIR}

ADD=`ls | grep OrderAdd_`
MOD=`ls | grep OrderModify_`
DEL=`ls | grep OrderDelete_`
MODSP=`ls | grep OrderModifySamePrio_`
FULL=`ls | grep FullOrderExecution_`
PART=`ls | grep PartialOrderExecution_`
EXEC=`ls | grep ExecutionSummary_`
echo -e "`date`\tmerging files" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
python3 merge.py ${ADD} ${MOD} ${DEL} ${MODSP} ${FULL} ${PART} ${EXEC}
echo -e "`date`\tmerging completed" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt

DATEWITHOUTDASHES=$(echo $DAY | sed 's/-//g')
FILENAME=${DATEWITHOUTDASHES}-${SEC}-${SECID}.csv

echo -e "`date`\tpreprocessing merged file" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
python3 preprocess.py ${FILENAME}
echo -e "`date`\tpreprocessing completed" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt

echo -e "`date`\treconstructing OB for lobster" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
# python3 reconstruction.py ${DATEWITHOUTDASHES} ${SEC} ${SECID}
echo -e "`date`\treconstruction completed" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Uploading from Scratch to S3                                                 |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
for FILE in $(ls | grep .csv); do
	echo -e "`date`\tuploading file $FILE to S3" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
	python3 upload.py $BUCKET $FILE ${DAY}/${OUTDIR}/${FILE}
	echo -e "`date`\tupload to S3 completed" >> $DATADIR/migrace/grep_awk/jobs_info.$PBS_JOBID.txt
done

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Cleanup                                                                      |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
clean_scratch
