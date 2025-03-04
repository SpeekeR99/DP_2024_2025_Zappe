import sys
import json
import time
import csv
import numpy as np
import pandas as pd
import datetime

MARKET_ID = "XEUR"
DATE = "20191202"
MARKET_SEGMENT_ID = "688"
SECURITY_ID = "4128839"

if len(sys.argv) == 5:
    MARKET_ID = sys.argv[1]
    DATE = sys.argv[2]
    MARKET_SEGMENT_ID = sys.argv[3]
    SECURITY_ID = sys.argv[4]

ORDER_ADD = 13100
ORDER_MODIFY = 13101
ORDER_MODIFY_SAME_PRIORITY = 13106
ORDER_DELETE = 13102
ORDER_MASS_DELETE = 13103
PARTIAL_ORDER_EXECUTION = 13105
FULL_ORDER_EXECUTION = 13104
# EXECUTION_SUMMARY = 13202
# AUCTION_BEST_BID_OFFER = 13500
# AUCTION_CLEARING_PRICE = 13501
# TOP_OF_BOOK = 13504
# PRODUCT_STATE_CHANGE = 13300
# INSTRUMENT_STATE_CHANGE = 13301
# CROSS_REQUEST = 13502
# QUOTE_REQUEST = 13503
# ADD_COMPLEX_INSTRUMENT = 13400
# TRADE_REPORT = 13201
# TRADE_REVERSAL = 13200
# PRODUCT_SUMMARY = 13600
# INSTRUMENT_SUMMARY = 13601
# SNAPSHOT_ORDER = 13602
# HEARTBEAT = 13001

# XEUR_20210319_589_5336359_detailed
DATE = "20210319"
MARKET_SEGMENT_ID = "589"
SECURITY_ID = "5336359"

print("Loading data...")
tic = time.time()
# TODO: Change the path if needed
with open(f"process-eobi-data/data/{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_detailed.json", "r") as fp:
    data = json.load(fp)
tac = time.time()
print(f"Data loaded in {tac - tic:.2f} seconds")

# Key - Timestamp | Value - (What to do, (Price, Quantity, Side, ...))
instructions = {}

print("Processing data...")
tic = time.time()
for i, part in enumerate(data):
    print(f"Processing part {i + 1}/{len(data)}")

    for transaction_array in part["Transactions"]:
        for transaction in transaction_array:
            if transaction["MessageHeader"]["TemplateID"] == ORDER_ADD:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                # SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                # OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
                Price = float(transaction["OrderDetails"]["Price"]) / 1e8

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("ADD", (Price, DisplayQty, Side)))
                else:
                    instructions[TrdRegTSTimeIn] = [("ADD", (Price, DisplayQty, Side))]

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MODIFY:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                TrdRegTSPrevTimePriority = transaction["TrdRegTSPrevTimePriority"]
                PrevPrice = float(transaction["PrevPrice"]) / 1e8
                PrevDisplayQty = float(transaction["PrevDisplayQty"]) / 1e8
                # SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                # OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
                Price = float(transaction["OrderDetails"]["Price"]) / 1e8

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("MODIFY", (Price, DisplayQty, Side, PrevPrice, PrevDisplayQty)))
                else:
                    instructions[TrdRegTSTimeIn] = [("MODIFY", (Price, DisplayQty, Side, PrevPrice, PrevDisplayQty))]

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MODIFY_SAME_PRIORITY:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                # TransactTime = transaction["TransactTime"]
                PrevDisplayQty = float(transaction["PrevDisplayQty"]) / 1e8
                # SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                # OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
                # Pad6 = "\x00\x00\x00\x00\x00\x00"
                Price = float(transaction["OrderDetails"]["Price"]) / 1e8

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("MODIFY_SAME_PRIORITY", (Price, DisplayQty, Side, Price, PrevDisplayQty)))
                else:
                    instructions[TrdRegTSTimeIn] = [("MODIFY_SAME_PRIORITY", (Price, DisplayQty, Side, Price, PrevDisplayQty))]

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_DELETE:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                # TransactTime = transaction["TransactTime"]
                # SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                # OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
                Price = float(transaction["OrderDetails"]["Price"]) / 1e8

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("DELETE", (Price, DisplayQty, Side)))
                else:
                    instructions[TrdRegTSTimeIn] = [("DELETE", (Price, DisplayQty, Side))]

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MASS_DELETE:
                SecurityID = transaction["SecurityID"]
                TransactTime = transaction["TransactTime"]

                if TransactTime in instructions:
                    instructions[TransactTime].append(("ORDER_MASS_DELETE", (SecurityID)))
                else:
                    instructions[TransactTime] = [("ORDER_MASS_DELETE", (SecurityID))]

            elif transaction["MessageHeader"]["TemplateID"] == PARTIAL_ORDER_EXECUTION:
                Side = transaction["Side"]
                # OrdType = transaction["OrdType"] if transaction["OrdType"] is not None else "NOVALUE"
                # AlgorithmicTradeIndicator = transaction["AlgorithmicTradeIndicator"] if transaction["AlgorithmicTradeIndicator"] is not None else "NOVALUE"
                # Pad1 = "\x00"
                # TrdMatchID = transaction["TrdMatchID"]
                Price = float(transaction["Price"]) / 1e8
                TrdRegTSTimePriority = transaction["TrdRegTSTimePriority"]
                # SecurityID = transaction["SecurityID"]
                LastQty = float(transaction["LastQty"]) / 1e8
                LastPx = float(transaction["LastPx"]) / 1e8

                if TrdRegTSTimePriority in instructions:
                    instructions[TrdRegTSTimePriority].append(("PARTIAL_ORDER_EXECUTION", (LastPx, LastQty, Side)))
                else:
                    instructions[TrdRegTSTimePriority] = [("PARTIAL_ORDER_EXECUTION", (LastPx, LastQty, Side))]

            elif transaction["MessageHeader"]["TemplateID"] == FULL_ORDER_EXECUTION:
                Side = transaction["Side"]
                # OrdType = transaction["OrdType"] if transaction["OrdType"] is not None else "NOVALUE"
                # AlgorithmicTradeIndicator = transaction["AlgorithmicTradeIndicator"] if transaction["AlgorithmicTradeIndicator"] is not None else "NOVALUE"
                # Pad1 = "\x00"
                # TrdMatchID = transaction["TrdMatchID"]
                Price = float(transaction["Price"]) / 1e8
                TrdRegTSTimePriority = transaction["TrdRegTSTimePriority"]
                # SecurityID = transaction["SecurityID"]
                LastQty = float(transaction["LastQty"]) / 1e8
                LastPx = float(transaction["LastPx"]) / 1e8

                if TrdRegTSTimePriority in instructions:
                    instructions[TrdRegTSTimePriority].append(("FULL_ORDER_EXECUTION", (LastPx, LastQty, Side)))
                else:
                    instructions[TrdRegTSTimePriority] = [("FULL_ORDER_EXECUTION", (LastPx, LastQty, Side))]

tac = time.time()
max_index = 0
for key, array in instructions.items():
    max_index += len(array)
print(f"Processed {max_index} transactions in {tac - tic:.2f} seconds")
print("Processing done")

# Variable "data" is now useless, free up memory
del data

# Sort the "orderbook" by timestamp
instructions = dict(sorted(instructions.items()))

# Create true orderbook now
print("Creating orderbook...")

# Oddly enough, the day starts with 33 FULL_ORDER_EXECUTIONs or PARTIAL_ORDER_EXECUTIONs
# but I have nothing in my lobster to trade on yet -> Ignoring until first ADD

keys_to_delete = []
to_be_added = []
for i, (key, array) in enumerate(instructions.items()):
    to_be_deleted = []
    done = False
    for j, value in enumerate(array):
        if value[0] != "ADD":
            to_be_deleted.append(j)
            to_be_added.append(value)
        else:
            done = True
            break

    for j in sorted(to_be_deleted, reverse=True):
        del array[j]

    if done:
        break
    else:
        keys_to_delete.append(key)

for key in keys_to_delete:
    del instructions[key]

first_key = next(iter(instructions))
# Add the to_be_added to the first key
for value in to_be_added:
    instructions[first_key].append(value)

# Go through each key and each array of instructions and sort the arrays, so that the "ADD" instructions are first
for key, array in instructions.items():
    instructions[key] = sorted(array, key=lambda x: x[0] == "ADD")

# Initialize data structures
max_index = len(instructions)
lobster_buy = [[] for _ in range(max_index)]
lobster_sell = [[] for _ in range(max_index)]
timestamps = []
cancellations = {}
levels = 100

tic = time.time()
for i, (timestamp, array) in enumerate(instructions.items()):
    # Print progress
    if i % 50_000 == 0:
        print(f"Processing order {i}/{max_index}")

    # Copy previous state
    if i > 0:
        lobster_buy[i] = lobster_buy[i - 1].copy()
        lobster_sell[i] = lobster_sell[i - 1].copy()

    # Save timestamp
    timestamps.append(timestamp)
    if timestamp not in cancellations:
        cancellations[timestamp] = 0

    for j, value in enumerate(array):

        # # THIS SEEMS LIKE THE BEST SOLUTION, BUT DOESNT YIELD GOOD RESULTS --------------------------------------------
        #
        if (value[0] == "DELETE" or value[0] == "FULL_ORDER_EXECUTION" or value[0] == "PARTIAL_ORDER_EXECUTION"
                                 or value[0] == "MODIFY" or value[0] == "MODIFY_SAME_PRIORITY"):
            if value[0] == "MODIFY" or value[0] == "MODIFY_SAME_PRIORITY":
                _, _, side, price, display_qty = value[1]
            else:
                price, display_qty, side = value[1]
            if side == 1:
                lobster = lobster_buy
            else:
                lobster = lobster_sell

            if value[0] == "DELETE":
                cancellations[timestamp] += 1

            # Find the order in the heap
            found = False
            for k, (p, q) in enumerate(lobster[i]):
                if p == price:
                    found = True
                    break

            # Delete the order
            if found:
                lobster[i][k] = (price, q - display_qty)
                if q - display_qty <= 0:
                    del lobster[i][k]

            # else:
            #     print(f"Order not found: {value}")
            #     print(f"Lobster: {lobster[i]}")

        if value[0] == "ADD" or value[0] == "MODIFY" or value[0] == "MODIFY_SAME_PRIORITY":
            if value[0] == "MODIFY" or value[0] == "MODIFY_SAME_PRIORITY":
                price, display_qty, side, _, _ = value[1]
            else:
                price, display_qty, side = value[1]
            if side == 1:
                lobster = lobster_buy
            else:
                lobster = lobster_sell

            for p, q in lobster[i]:
                if p == price:
                    q += display_qty
                    break
            else:  # For-Else
                if len(lobster[i]) < levels:
                    lobster[i].append((price, display_qty))
                elif side == 1:
                    # Find the worst price
                    worst_price = min(lobster_buy[i], key=lambda x: x[0])[0]
                    if price > worst_price:
                        # Replace the worst price
                        for k, (p, q) in enumerate(lobster_buy[i]):
                            if p == worst_price:
                                lobster_buy[i][k] = (price, display_qty)
                                break
                elif side == 2:
                    # Find the worst price
                    worst_price = max(lobster_sell[i], key=lambda x: x[0])[0]
                    if price < worst_price:
                        # Replace the worst price
                        for k, (p, q) in enumerate(lobster_sell[i]):
                            if p == worst_price:
                                lobster_sell[i][k] = (price, display_qty)
                                break

        if value[0] == "ORDER_MASS_DELETE":
            # Delete all orders
            lobster_buy[i] = []
            lobster_sell[i] = []
        #
        # # THIS SEEMS LIKE THE BEST SOLUTION, BUT DOESNT YIELD GOOD RESULTS --------------------------------------------

        # if value[0] == "ADD":
        #     price, display_qty, side = value[1]
        #     if side == 1:
        #         lobster = lobster_buy
        #     else:
        #         lobster = lobster_sell
        #
        #     for p, q in lobster[i]:
        #         if p == price:
        #             q += display_qty
        #             break
        #     else:  # For-Else
        #         if len(lobster[i]) < levels:
        #             lobster[i].append((price, display_qty))
        #         elif side == 1:
        #             # Find the worst price
        #             worst_price = min(lobster_buy[i], key=lambda x: x[0])[0]
        #             if price > worst_price:
        #                 # Replace the worst price
        #                 for k, (p, q) in enumerate(lobster_buy[i]):
        #                     if p == worst_price:
        #                         lobster_buy[i][k] = (price, display_qty)
        #                         break
        #         elif side == 2:
        #             # Find the worst price
        #             worst_price = max(lobster_sell[i], key=lambda x: x[0])[0]
        #             if price < worst_price:
        #                 # Replace the worst price
        #                 for k, (p, q) in enumerate(lobster_sell[i]):
        #                     if p == worst_price:
        #                         lobster_sell[i][k] = (price, display_qty)
        #                         break
        #
        # elif value[0] == "MODIFY" or value[0] == "MODIFY_SAME_PRIORITY":
        #     price, display_qty, side, prev_price, prev_display_qty = value[1]
        #     if side == 1:
        #         lobster = lobster_buy
        #     else:
        #         lobster = lobster_sell
        #
        #     # Find the order in the heap
        #     found = False
        #     for k, (p, q) in enumerate(lobster[i]):
        #         if p == prev_price:
        #             found = True
        #             break
        #
        #     # Modify the order
        #     if found and price == prev_price:
        #         lobster[i][k] = (price, lobster[i][k][1] - prev_display_qty + display_qty)
        #         # if lobster[i][k][1] <= 0:
        #         #     del lobster[i][k]
        #     elif found:
        #         lobster[i][k] = (prev_price, lobster[i][k][1] - prev_display_qty)
        #         # if lobster[i][k][1] <= 0:
        #         #     del lobster[i][k]
        #
        #         found = False
        #         for l, (p, q) in enumerate(lobster[i]):
        #             if p == price:
        #                 found = True
        #                 break
        #         else:
        #             # if len(lobster[i]) < levels:
        #             #     lobster[i].append((price, display_qty))
        #             # elif side == 1:
        #             #     # Find the worst price
        #             #     worst_price = min(lobster_buy[i], key=lambda x: x[0])[0]
        #             #     if price > worst_price:
        #             #         # Replace the worst price
        #             #         for m, (p, q) in enumerate(lobster_buy[i]):
        #             #             if p == worst_price:
        #             #                 lobster_buy[i][m] = (price, display_qty)
        #             #                 break
        #             # elif side == 2:
        #             #     # Find the worst price
        #             #     worst_price = max(lobster_sell[i], key=lambda x: x[0])[0]
        #             #     if price < worst_price:
        #             #         # Replace the worst price
        #             #         for m, (p, q) in enumerate(lobster_sell[i]):
        #             #             if p == worst_price:
        #             #                 lobster_sell[i][m] = (price, display_qty)
        #             #                 break
        #             # Find the worst price
        #             # else:
        #             worst_price = max(lobster[i], key=lambda x: x[0])[0]
        #             if price < worst_price:
        #                 # Replace the worst price
        #                 for m, (p, q) in enumerate(lobster[i]):
        #                     if p == worst_price:
        #                         lobster[i][m] = (price, display_qty)
        #                         break
        #
        #         if found:
        #             lobster[i][l] = (price, lobster[i][l][1] + display_qty)
        #
        #     # # Not found at all
        #     # else:
        #     #     if len(lobster[i]) < levels:
        #     #         lobster[i].append((price, display_qty))
        #     #     elif side == 1:
        #     #         # Find the worst price
        #     #         worst_price = min(lobster_buy[i], key=lambda x: x[0])[0]
        #     #         if price > worst_price:
        #     #             # Replace the worst price
        #     #             for l, (p, q) in enumerate(lobster_buy[i]):
        #     #                 if p == worst_price:
        #     #                     lobster_buy[i][l] = (price, display_qty)
        #     #                     break
        #     #     elif side == 2:
        #     #         # Find the worst price
        #     #         worst_price = max(lobster_sell[i], key=lambda x: x[0])[0]
        #     #         if price < worst_price:
        #     #             # Replace the worst price
        #     #             for l, (p, q) in enumerate(lobster_sell[i]):
        #     #                 if p == worst_price:
        #     #                     lobster_sell[i][l] = (price, display_qty)
        #     #                     break
        #
        # elif value[0] == "DELETE" or value[0] == "FULL_ORDER_EXECUTION" or value[0] == "PARTIAL_ORDER_EXECUTION":
        #     price, display_qty, side = value[1]
        #     if side == 1:
        #         lobster = lobster_buy
        #     else:
        #         lobster = lobster_sell
        #
        #     if value[0] == "DELETE":
        #         cancellations[timestamp] += 1
        #
        #     # Find the order in the heap
        #     found = False
        #     for k, (p, q) in enumerate(lobster[i]):
        #         if p == price:
        #             found = True
        #             break
        #
        #     # Delete the order
        #     if found:
        #         lobster[i][k] = (price, q - display_qty)
        #         if lobster[i][k][1] <= 0:
        #             del lobster[i][k]
        #
        # elif value[0] == "ORDER_MASS_DELETE":
        #     # Delete all orders
        #     lobster_buy[i] = []
        #     lobster_sell[i] = []

tac = time.time()
print(f"Created orderbook in {tac - tic:.2f} seconds")

# Variable "instructions" is now useless, free up memory
del instructions

# Export to lobster csv
print("Preparing for export to CSV...")

# Sort the levels correspondingly to the price (depends on buy / sell, best buy price is highest, best sell lowest)
# Be aware that the prices are NEGATIVE for BUY orders -> thus sorting does not have to be REVERSED
for i in range(max_index):
    lobster_buy[i] = sorted(lobster_buy[i], reverse=True)
    lobster_sell[i] = sorted(lobster_sell[i])

# Time,Ask Price 1, Ask Volume 1, Bid Price 1, Bid Volume 1, ...
levels = 30
lobster_header = "Time,"
for i in range(levels):
    lobster_header += f"Ask Price {i + 1},Ask Volume {i + 1},Bid Price {i + 1},Bid Volume {i + 1},"
lobster_header = lobster_header[:-1]  # Remove last comma
lobster_header = lobster_header.split(",")

OUTPUT_FILE_PATH = "pokus_lobsteru.csv"

print("Exporting to CSV...")

tic = time.time()
# Export to CSV
with open(OUTPUT_FILE_PATH, "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(lobster_header)  # Write header

    for i in range(max_index):
        # row = [i]
        row = [timestamps[i]]  # Start with timestamp
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
                row.extend([f"{lobster_buy[i][level][0]:.8f}", f"{lobster_buy[i][level][1]:.8f}"])
            else:
                row.extend(["", ""])  # Empty values

        writer.writerow(row)  # Write row to CSV

toc = time.time()
print(f"Exported to CSV in {toc - tic:.2f} seconds")

# # Variable "lobster_buy" and "lobster_sell" are now useless, free up memory
# del lobster_buy
# del lobster_sell
#
# # Add imbalance index, frequency of incoming messages, cancellations rate to the CSV
# print("Augmenting CSV with imbalance index, frequency of incoming messages, cancellations rate...")
# tic = time.time()
#
# class Config:
#     @staticmethod
#     def calc_nansec_from_time(time: str) -> int:
#         """
#         :param time: time in format hh:mm:ss.nnnnnnnnn or hh:mm:ss
#         Returns number of nanoseconds from time in format hh:mm:ss.nnnnnnnnn or hh:mm:ss
#         """
#         time = time.split(":")
#         nansec = time[-1].split(".")
#         return int(int(time[0]) * 36e11 + int(time[1]) * 6e10 + int(nansec[0]) * 1e9 + (int(nansec[1].ljust(9, "0")) if len(nansec) == 2 else 0))
#
#     @staticmethod
#     def calc_time_from_nansec(nansecs: int) -> str:
#         """
#         :param nansecs: number of nanoseconds
#         Returns time in format hh:mm:ss.nnnnnnnnn from given nanoseconds
#         """
#         s = nansecs // 1e9
#         delta = datetime.timedelta(seconds=s)
#         ns = str(int(nansecs % 1e9)).zfill(9)
#         datetime_obj = (datetime.datetime.min + delta).time()
#         time_formatted = datetime_obj.strftime('%H:%M:%S')
#         time = time_formatted + "." + ns
#         return time
#
#
# def imbalance_index_vectorized(asks, bids, alpha=0.5, level=3):
#     """
#     Calculate imbalance index for a given orderbook.
#     :param asks: numpy matrix of ask sizes (volumes)
#     :param bids: numpy matrix of bid sizes (volumes)
#     :param alpha: parameter for imbalance index
#     :param level: number of levels to consider
#     :return: imbalance index
#     """
#     assert asks.shape[1] >= level and bids.shape[1] >= level, "Not enough levels in orderbook"
#     assert alpha > 0, "Alpha must be positive"
#     assert level > 0, "Level must be positive"
#
#     # Calculate imbalance index
#     V_bt = np.sum(bids[:, :level] * np.exp(-alpha * np.arange(0, level)), axis=1)
#     V_at = np.sum(asks[:, :level] * np.exp(-alpha * np.arange(0, level)), axis=1)
#     return (V_bt - V_at) / (V_bt + V_at)
#
#
# def get_frequency_of_all_incoming_actions(timestamps, time=300):
#     """
#     Returns the frequency of incoming actions (messages) in the last *time* seconds for all timestamps
#     :param timestamps: Array of timestamps (in nano seconds)
#     :param time: Time window in seconds (default value is 300)
#     :return: List of frequencies of incoming actions (messages) for all timestamps
#     """
#     # Convert time to nanoseconds
#     time_ns = time * 1e9
#
#     # Convert timestamps to a numpy array
#     timestamps = np.array(timestamps)
#
#     # Initialize two pointers at the end of the timestamps array
#     start, end = len(timestamps) - 1, len(timestamps) - 1
#
#     # Initialize an array to hold the frequencies
#     freqs = np.zeros_like(timestamps)
#
#     # Calculate the frequency for each timestamp
#     for start in range(len(timestamps)-1, -1, -1):
#         while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
#             end -= 1
#         freqs[start] = start - end
#
#     return freqs / time
#
#
# data = pd.read_csv(OUTPUT_FILE_PATH, delimiter=",")
#
# ask_columns = [f'Ask Volume {i}' for i in range(1, levels+1)]
# bid_columns = [f'Bid Volume {i}' for i in range(1, levels+1)]
#
# # Imbalance index
# lobster_data_matrix = data[ask_columns + bid_columns].values
# imbalance_indices = imbalance_index_vectorized(lobster_data_matrix[:, :levels], lobster_data_matrix[:, levels:], alpha=0.5, level=levels)
#
# # Frequency of incoming messages
# time_window = 300
# timestamps_orig = data["Time"].tolist()
# temp = [Config.calc_time_from_nansec(t) for t in timestamps_orig]
# timestamps = [Config.calc_nansec_from_time(t) for t in temp]
# freqs = get_frequency_of_all_incoming_actions(timestamps, time=time_window)
#
# # Cancellations rate
# # Sliding average over the last *time_window* seconds
# cancellation_rate = np.zeros_like(timestamps, dtype=float)
# cancellations = list(cancellations.values())
# for i in range(len(timestamps)):
#     start = max(0, i - time_window)
#     end = i
#     cancellation_rate[i] = np.sum([cancellations[j] for j in range(start, end)]) / time_window
