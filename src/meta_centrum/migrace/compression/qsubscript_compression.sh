# 1 GB
# qsub -v FILE="2350_00_D_03_A_20210104.InstrumentSummary.MDInstrumentEntryGrp.csv" compression_test.sh
# 20 GB
# qsub -v FILE="2350_00_D_03_A_20210104.InstrumentSummary.csv" compression_test.sh
# 60 GB
qsub -v FILE="2350_00_D_03_A_20210104.SnapshotOrder.csv" compression_test.sh
# 140 GB
qsub -v FILE="2350_00_D_03_A_20210104.OrderModify.csv" compression_test.sh