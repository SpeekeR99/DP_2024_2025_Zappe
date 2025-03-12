import pandas as pd
import numpy as np
import time

# Config
infile = "merged/20191202-FGBL-4128839.csv"
outfile = "data/20191202-FGBL-4128839-ob.csv"
incols = ["PARENT_ID", "ID", "TrdRegTSTimeIn", "TrdRegTSTimePriority", "Side", "Price", "DisplayQty", "op", "Trans", "Prio"]
outcols = ['i', 'Price', 'DisplayQty', 'Q', 'od', 'do', "Trans", "Prio"]
delim = ","
ops_to_include = ['A', 'D', 'XXX', 'YYY', 'E', 'MS', 'PE']  # Operations to include in final OB

# Load data from CSV
Qdbb = pd.read_csv(infile, delimiter=delim, usecols=incols)

# Filter rows based on op
Qdbb = Qdbb[Qdbb["op"].isin(ops_to_include)]
Qdbb.reset_index(drop=True, inplace=True)

# Update Price column based on Side
Qdbb.loc[Qdbb["Side"] == 'S', "Price"] *= -1

# Add new column 'i' and 'Q'
Qdbb['i'] = np.arange(1, len(Qdbb) + 1)
Qdbb['Q'] = Qdbb['DisplayQty'].astype(int)

# Initialize columns 'od' and 'do'
Qdbb['od'] = Qdbb['i']
Qdbb['do'] = len(Qdbb) + 1

tic = time.time()

# Iterate over unique Price values
for i in Qdbb["Price"].unique():
    pom = Qdbb.index[Qdbb["Price"] == i]
    Qdbb.loc[pom, "DisplayQty"] = np.cumsum(Qdbb.loc[pom, "DisplayQty"])
    Qdbb.loc[pom[:-1], 'do'] = Qdbb.loc[pom[1:], 'i'].values

Qdbb['DisplayQty'] = Qdbb['DisplayQty'].astype(int)
toc = time.time()
print('Elapsed time:', toc - tic)

# Export to CSV
Qdbb[outcols].to_csv(outfile, index=False)
