#!/bin/bash

# qsub -v DAY="2021-01-04",SEC="FGBX",SECID="5315926" grep_awk_csv.sh
# qsub -v DAY="2021-01-04",SEC="FGBL",SECID="5578483" grep_awk_csv.sh
# qsub -v DAY="2021-01-04",SEC="FGBL",SECID="5894217" grep_awk_csv.sh
# qsub -v DAY="2021-01-04",SEC="FGBL",SECID="6183278" grep_awk_csv.sh
# qsub -v DAY="2021-01-04",SEC="FGBL",SECID="4128839" grep_awk_csv.sh

for D in 04 05 06 07 08 11 12 13 14 15 18 19 20 21 22 25 26 27 28 29
do
    qsub -v DAY="2021-01-${D}",SEC="FGBL",SECID="5578483" grep_awk_csv.sh
    qsub -v DAY="2021-01-${D}",SEC="FGBL",SECID="5894217" grep_awk_csv.sh
    qsub -v DAY="2021-01-${D}",SEC="FGBL",SECID="6183278" grep_awk_csv.sh
    qsub -v DAY="2021-01-${D}",SEC="FGBL",SECID="4128839" grep_awk_csv.sh
done