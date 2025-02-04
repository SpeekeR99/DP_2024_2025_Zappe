#!/bin/bash

ssh -tt zapped99@ssh.du5.cesnet.cz "ls -l ./VO_zcu_thebigbook-disk_only-archive-shared/data/Eurex/" | grep -v ".rar_0" | grep ".rar" | awk '{print $9}' > files.txt

for FILE in $(cat files.txt)
do
    echo $FILE
    qsub -v DIR="~/VO_zcu_thebigbook-disk_only-archive-shared/data/Eurex/",FILE=$FILE du5_to_s3.sh
done

# qsub -v DIR="~/VO_zcu_thebigbook-disk_only-archive-shared/data/Eurex/",FILE="2350_00_D_03_A_20210104.rar" du5_to_s3.sh
