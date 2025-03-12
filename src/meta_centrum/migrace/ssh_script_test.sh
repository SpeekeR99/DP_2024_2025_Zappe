#!/bin/bash

# echo "Before SSH"
# ssh -tt zapped99@ssh.du5.cesnet.cz<<EOT
# ls -l
# exit
# EOT
# echo "After SSH"
# scp zapped99@ssh.du5.cesnet.cz:~/VO_zcu_thebigbook-disk_only-archive-shared/data/Eurex/crc.py .

FILE="2350_2021-06-01_00-00-00"
FILE_2302=$(echo $FILE | sed 's/2350/2302/g')

echo "$FILE"
echo "$FILE_2302"