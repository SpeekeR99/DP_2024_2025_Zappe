#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=8:0:0
#PBS -l select=1:ncpus=4:mem=512gb:scratch_local=512gb:cl_halmir=True
#PBS -N parse_pcap

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Variables                                                                    |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
DATADIR=/storage/plzen1/home/zapped99/s3
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

BUCKET_RAW="data.raw"
BUCKET_PROCESSED="data"
DAY=$(echo $FILE | cut -d'_' -f6 | cut -c1-8 | sed 's/\(....\)\(..\)\(..\)/\1-\2-\3/')

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Preparation                                                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
cd $SCRATCHDIR
cp -r $DATADIR/EOBI_parser $SCRATCHDIR
cp $DATADIR/migrace/upload.py $SCRATCHDIR/upload.py
cp $DATADIR/migrace/download.py $SCRATCHDIR/download.py
cp $DATADIR/migrace/credentials $SCRATCHDIR/credentials

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Download from S3 to Scratch                                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
echo -e "`date`\tdownloading file $FILE from S3" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
python3 download.py $BUCKET_RAW ${FILE}.rar ${FILE}.rar
echo -e "`date`\tdownloading from S3 completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Unrar the file                                                               |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
echo -e "`date`\tunraring file $FILE" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
unrar x ${FILE}.rar
echo -e "`date`\tunraring completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
rm ${FILE}.rar

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Parse the pcap                                                               |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
cd EOBI_parser/emdda1063
chmod 777 emdda
FILE_2302=$(echo $FILE | sed 's/2350/2302/g')

echo -e "`date`\tparsing pcap file $FILE_2302" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
LD_LIBRARY_PATH=${DATADIR}/EOBI_parser/lib/ ./emdda -stats -pcap ../../${FILE_2302}.pcap -t ../../RDDFastTemplates-1.1.xml > ${FILE_2302}.txt
echo -e "`date`\tparsing completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

cd ../../
mkdir parser_output
cd EOBI_parser/R9.0
chmod 777 deobi

echo -e "`date`\tparsing pcap file $FILE" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
LD_LIBRARY_PATH=${DATADIR}/EOBI_parser/lib/ ./deobi -s eobi.xml -c ../jp-all-8.1.1.conf --input-file ../../${FILE}.pcap --output-directory ../../parser_output
echo -e "`date`\tparsing completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

echo -e "`date`\tparsing pcap file $FILE for summary only" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
LD_LIBRARY_PATH=${DATADIR}/EOBI_parser/lib/ ./deobi --summary-only -s eobi.xml -c ../jp-all-8.1.1.conf --input-file ../../${FILE}.pcap > summary_${FILE}.txt
echo -e "`date`\tparsing for summary completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Upload csv's to S3                                                           |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
cd ../../
echo -e "`date`\tuploading file summary_${FILE}.txt to S3" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
python3 upload.py $BUCKET_PROCESSED EOBI_parser/R9.0/summary_${FILE}.txt ${DAY}/all/summary_${FILE}.txt
echo -e "`date`\tuploading to S3 completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

echo -e "`date`\tuploading file ${FILE_2302}.txt to S3" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
python3 upload.py $BUCKET_PROCESSED EOBI_parser/emdda1063/${FILE_2302}.txt ${DAY}/all/${FILE_2302}.txt
echo -e "`date`\tuploading to S3 completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt

cp upload.py parser_output
cp credentials parser_output
cd parser_output
ls -l | grep ".csv" | awk '{print $9}' > upload.txt
for CSV in $(cat upload.txt)
do
    echo -e "`date`\tuploading file $CSV to S3" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
    python3 upload.py $BUCKET_PROCESSED $CSV ${DAY}/all/${CSV}
    echo -e "`date`\tuploading to S3 completed" >> $DATADIR/migrace/jobs_info.$PBS_JOBID.txt
done

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Cleanup                                                                      |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
clean_scratch
