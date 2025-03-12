import sys
import time
import heapq
import csv
import pandas as pd

print(sys.argv)
DATE = "20191202"
# DATE = sys.argv[1]
INSTRUMENT = "FGBL"
# MARKET_ID = sys.argv[2]
SECURITY_ID = "4128839"
# SECURITY_ID = sys.argv[3]

INPUT_FILE_PATH = f"data/{DATE}-{INSTRUMENT}-{SECURITY_ID}-ob.csv"
OUTPUT_FILE_PATH = f"../../A7/process-eobi-data/data/lobster_test.csv"

# Load data from CSV
data = pd.read_csv(INPUT_FILE_PATH)

print(data.head())

# Initialize data structures
max_index = data["do"].max()
lobster_buy = [[] for _ in range(max_index)]
lobster_sell = [[] for _ in range(max_index)]
timestamps = []

print(len(lobster_buy))
print(len(lobster_sell))

tic = time.time()

# Process data
levels = 30
# Iterate over each row
for i, row in data.iterrows():
    timestamp = int(row["Prio"])
    timestamps.append(timestamp)

    qty = int(row["DisplayQty"])
    if qty == 0:  # Skip zero quantity orders
        continue

    price = float(row["Price"])
    if price < 0:
        # Sell order: Use lobster_sell and store price as negative for max-heap behavior
        lobster = lobster_sell
    else:
        # Buy order: Use lobster_buy, keep price as is
        lobster = lobster_buy
    price = -price

    od_value = int(row["od"])
    do_value = int(row["do"])

    for j in range(od_value, do_value):
        heap = lobster[j]

        if len(heap) < levels:
            heapq.heappush(heap, (price, qty))
        # This next line works for both BUY and SELL orders, because the prices are negative for buy orders
        # Thus the max-heap behavior is achieved for buy orders, and min-heap for sell orders
        # This way the WORST price for both is always at the top of the heap (the one we compare against)
        elif price > heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, (price, qty))

# Sort the levels correspondingly to the price (depends on buy / sell, best buy price is highest, best sell lowest)
# Be aware that the prices are NEGATIVE for BUY orders -> thus sorting does not have to be REVERSED
for i in range(max_index):
    lobster_buy[i] = sorted(lobster_buy[i])
    lobster_sell[i] = sorted(lobster_sell[i])

timestamps = sorted(timestamps)
print(len(timestamps))

toc = time.time()
print("Elapsed time:", toc - tic)
print("Processing done")

# Time,Ask Price 1, Ask Volume 1, Bid Price 1, Bid Volume 1, ...
lobster_header = "Time,"
for i in range(levels):
    lobster_header += f"Ask Price {i + 1},Ask Volume {i + 1},Bid Price {i + 1},Bid Volume {i + 1},"
lobster_header = lobster_header[:-1]  # Remove last comma

tic = time.time()

# Export to CSV
with open(OUTPUT_FILE_PATH, "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(lobster_header.split(","))  # Write header

    for i in range(1, max_index):
        # row = [i]
        row = [timestamps[i - 1]]  # Start with index
        for level in range(levels):
            # Add sell levels
            if level < len(lobster_sell[i]):
                # Sell prices are already positive
                row.extend([f"{lobster_sell[i][level][0]:.8f}", f"{lobster_sell[i][level][1]:.8f}"])
            else:
                row.extend(["", ""])  # Empty values

            # Add buy levels
            if level < len(lobster_buy[i]):
                # Buy prices are negative -> make them positive
                row.extend([f"{-lobster_buy[i][level][0]:.8f}", f"{lobster_buy[i][level][1]:.8f}"])
            else:
                row.extend(["", ""])  # Empty values

        writer.writerow(row)  # Write row to CSV

toc = time.time()
print("Elapsed time:", toc - tic)
print("Export done")
