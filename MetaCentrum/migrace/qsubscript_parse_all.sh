#!/bin/bash

for FILE in $(cat files.txt)
do
    FILE=$(echo $FILE | cut -d'.' -f1)
    echo $FILE
    qsub -v FILE=$FILE parse_pcap.sh
done

# qsub -v FILE="2350_00_D_03_A_20210121" parse_pcap.sh
