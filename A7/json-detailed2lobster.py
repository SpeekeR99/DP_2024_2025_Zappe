import sys
import json
import time
import csv

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
EXECUTION_SUMMARY = 13202
AUCTION_BEST_BID_OFFER = 13500
AUCTION_CLEARING_PRICE = 13501
TOP_OF_BOOK = 13504
# PRODUCT_STATE_CHANGE = 13300
INSTRUMENT_STATE_CHANGE = 13301
# CROSS_REQUEST = 13502
# QUOTE_REQUEST = 13503
# ADD_COMPLEX_INSTRUMENT = 13400
TRADE_REPORT = 13201
# TRADE_REVERSAL = 13200
# PRODUCT_SUMMARY = 13600
INSTRUMENT_SUMMARY = 13601
# SNAPSHOT_ORDER = 13602
# HEARTBEAT = 13001

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
                SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
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
                SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
                Price = float(transaction["OrderDetails"]["Price"]) / 1e8

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("MODIFY", (Price, DisplayQty, Side, PrevPrice, PrevDisplayQty)))
                else:
                    instructions[TrdRegTSTimeIn] = [("MODIFY", (Price, DisplayQty, Side, PrevPrice, PrevDisplayQty))]

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MODIFY_SAME_PRIORITY:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                TransactTime = transaction["TransactTime"]
                PrevDisplayQty = float(transaction["PrevDisplayQty"]) / 1e8
                SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
                Pad6 = "\x00\x00\x00\x00\x00\x00"
                Price = float(transaction["OrderDetails"]["Price"]) / 1e8

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("MODIFY_SAME_PRIORITY", (Price, DisplayQty, Side, Price, PrevDisplayQty)))
                else:
                    instructions[TrdRegTSTimeIn] = [("MODIFY_SAME_PRIORITY", (Price, DisplayQty, Side, Price, PrevDisplayQty))]

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_DELETE:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                TransactTime = transaction["TransactTime"]
                SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
                Price = float(transaction["OrderDetails"]["Price"]) / 1e8

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("DELETE", (Price, DisplayQty, Side)))
                else:
                    instructions[TrdRegTSTimeIn] = [("DELETE", (Price, DisplayQty, Side))]

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MASS_DELETE:
                SecurityID = transaction["SecurityID"]
                TransactTime = transaction["TransactTime"]
                pass

            elif transaction["MessageHeader"]["TemplateID"] == PARTIAL_ORDER_EXECUTION:
                Side = transaction["Side"]
                OrdType = transaction["OrdType"] if transaction["OrdType"] is not None else "NOVALUE"
                AlgorithmicTradeIndicator = transaction["AlgorithmicTradeIndicator"] if transaction["AlgorithmicTradeIndicator"] is not None else "NOVALUE"
                Pad1 = "\x00"
                TrdMatchID = transaction["TrdMatchID"]
                Price = float(transaction["Price"]) / 1e8
                TrdRegTSTimePriority = transaction["TrdRegTSTimePriority"]
                SecurityID = transaction["SecurityID"]
                LastQty = float(transaction["LastQty"]) / 1e8
                LastPx = float(transaction["LastPx"]) / 1e8

                if TrdRegTSTimePriority in instructions:
                    instructions[TrdRegTSTimePriority].append(("PARTIAL_ORDER_EXECUTION", (LastPx, LastQty, Side)))
                else:
                    instructions[TrdRegTSTimePriority] = [("PARTIAL_ORDER_EXECUTION", (LastPx, LastQty, Side))]

            elif transaction["MessageHeader"]["TemplateID"] == FULL_ORDER_EXECUTION:
                Side = transaction["Side"]
                OrdType = transaction["OrdType"] if transaction["OrdType"] is not None else "NOVALUE"
                AlgorithmicTradeIndicator = transaction["AlgorithmicTradeIndicator"] if transaction["AlgorithmicTradeIndicator"] is not None else "NOVALUE"
                Pad1 = "\x00"
                TrdMatchID = transaction["TrdMatchID"]
                Price = float(transaction["Price"]) / 1e8
                TrdRegTSTimePriority = transaction["TrdRegTSTimePriority"]
                SecurityID = transaction["SecurityID"]
                LastQty = float(transaction["LastQty"]) / 1e8
                LastPx = float(transaction["LastPx"]) / 1e8

                if TrdRegTSTimePriority in instructions:
                    instructions[TrdRegTSTimePriority].append(("FULL_ORDER_EXECUTION", (LastPx, LastQty, Side)))
                else:
                    instructions[TrdRegTSTimePriority] = [("FULL_ORDER_EXECUTION", (LastPx, LastQty, Side))]

            elif transaction["MessageHeader"]["TemplateID"] == EXECUTION_SUMMARY:
                SecurityID = transaction["SecurityID"]
                AggressorTime = transaction["AggressorTime"]
                RequestTime = transaction["RequestTime"]
                ExecID = transaction["ExecID"]
                LastQty = float(transaction["LastQty"]) / 1e8
                AggressorSide = transaction["AggressorSide"]
                Pad1 = "\x00"
                TradeCondition = transaction["TradeCondition"] if transaction["TradeCondition"] is not None else "NOVALUE"
                Pad4 = "\x00\x00\x00\x00"
                LastPx = float(transaction["LastPx"]) / 1e8
                RestingHiddenQty = f"{float(transaction['RestingHiddenQty']) / 1e8}:.8f" if (transaction["RestingHiddenQty"] is not None) and (transaction["RestingHiddenQty"] != 0) and (transaction["RestingHiddenQty"] != "0") else "NOVALUE"
                RestingCxlQty = f"{float(transaction['RestingCxlQty']) / 1e8}:.8f" if (transaction["RestingCxlQty"] is not None) and (transaction["RestingCxlQty"] != 0) and (transaction["RestingHiddenQty"] != "0") else "NOVALUE"
                pass

            elif transaction["MessageHeader"]["TemplateID"] == AUCTION_BEST_BID_OFFER:
                TransactTime = transaction["TransactTime"]
                SecurityID = transaction["SecurityID"]
                BidPx = float(transaction["BidPx"]) / 1e8
                OfferPx = float(transaction["OfferPx"]) / 1e8
                BidSize = f'{float(transaction["BidSize"]) / 1e8}:.8f' if transaction["BidSize"] is not None else "NOVALUE"
                OfferSize = f'{float(transaction["OfferSize"]) / 1e8}:.8f' if transaction["OfferSize"] is not None else "NOVALUE"
                PotentialSecurityTradingEvent = transaction["PotentialSecurityTradingEvent"] if transaction["PotentialSecurityTradingEvent"] is not None else "NOVALUE"
                BidOrdType = transaction["BidOrdType"] if transaction["BidOrdType"] is not None else "NOVALUE"
                OfferOrdType = transaction["OfferOrdType"] if transaction["OfferOrdType"] is not None else "NOVALUE"
                Pad5 = "\x00\x00\x00\x00\x00"
                pass

            elif transaction["MessageHeader"]["TemplateID"] == AUCTION_CLEARING_PRICE:
                TransactTime = transaction["TransactTime"]
                SecurityID = transaction["SecurityID"]
                LastPx = f'{float(transaction["LastPx"]) / 1e8}:.8f' if transaction["LastPx"] is not None else "NOVALUE"
                LastQty = f'{float(transaction["LastQty"]) / 1e8}:.8f' if transaction["LastQty"] is not None else "NOVALUE"
                ImbalanceQty = f'{float(transaction["ImbalanceQty"]) / 1e8}:.8f' if transaction["ImbalanceQty"] is not None else "NOVALUE"
                SecurityTradingStatus = transaction["SecurityTradingStatus"] if transaction["SecurityTradingStatus"] is not None else "NOVALUE"
                PotentialSecurityTradingEvent = transaction["PotentialSecurityTradingEvent"] if transaction["PotentialSecurityTradingEvent"] is not None else "NOVALUE"
                Pad6 = "\x00\x00\x00\x00\x00\x00"
                pass

            elif transaction["MessageHeader"]["TemplateID"] == TOP_OF_BOOK:
                TransactTime = transaction["TransactTime"]
                SecurityID = transaction["SecurityID"]
                BidPx = float(transaction["BidPx"]) / 1e8
                OfferPx = float(transaction["OfferPx"]) / 1e8
                BidSize = f'{float(transaction["BidSize"]) / 1e8}:.8f' if transaction["BidSize"] is not None else "NOVALUE"
                OfferSize = f'{float(transaction["OfferSize"]) / 1e8}:.8f' if transaction["OfferSize"] is not None else "NOVALUE"
                pass

            elif transaction["MessageHeader"]["TemplateID"] == INSTRUMENT_STATE_CHANGE:
                SecurityID = transaction["SecurityID"]
                SecurityStatus = transaction["SecurityStatus"]
                SecurityTradingStatus = transaction["SecurityTradingStatus"]
                MarketCondition = transaction["MarketCondition"]
                FastMarketIndicator = transaction["FastMarketIndicator"]
                SecurityTradingEvent = transaction["SecurityTradingEvent"] if transaction["SecurityTradingEvent"] is not None else "NOVALUE"
                SoldOutIndicator = transaction["SoldOutIndicator"] if transaction["SoldOutIndicator"] is not None else "NOVALUE"
                Pad2 = "\x00\x00"
                TransactTime = transaction["TransactTime"]
                pass

            elif transaction["MessageHeader"]["TemplateID"] == TRADE_REPORT:
                SecurityID = transaction["SecurityID"]
                TransactTime = transaction["TransactTime"]
                LastQty = float(transaction["LastQty"]) / 1e8
                LastPx = float(transaction["LastPx"]) / 1e8
                TrdMatchID = transaction["TrdMatchID"]
                MatchType = transaction["MatchType"]
                MatchSubType = transaction["MatchSubType"]
                AlgorithmicTradeIndicator = transaction["AlgorithmicTradeIndicator"] if transaction["AlgorithmicTradeIndicator"] is not None else "NOVALUE"
                TradeCondition = transaction["TradeCondition"] if transaction["TradeCondition"] is not None else "NOVALUE"
                pass

            elif transaction["MessageHeader"]["TemplateID"] == INSTRUMENT_SUMMARY:
                SecurityID = transaction["SecurityID"]
                LastUpdateTime = transaction["LastUpdateTime"]
                TrdRegTSExecutionTime = transaction["TrdRegTSExecutionTime"]
                TotNoOrders = transaction["TotNoOrders"]
                SecurityStatus = transaction["SecurityStatus"]
                SecurityTradingStatus = transaction["SecurityTradingStatus"]
                MarketCondition = transaction["MarketCondition"]
                FastMarketIndicator = transaction["FastMarketIndicator"]
                SecurityTradingEvent = transaction["SecurityTradingEvent"]
                SoldOutIndicator = transaction["SoldOutIndicator"]
                ProductComplex = transaction["ProductComplex"]
                NoMDEntries = transaction["NoMDEntries"]
                Pad6 = "\x00\x00\x00\x00\x00\x00"
                pass

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
for i, (key, array) in enumerate(instructions.items()):
    to_be_deleted = []
    done = False
    for j, value in enumerate(array):
        if value[0] != "ADD":
            to_be_deleted.append(j)
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

# Initialize data structures
max_index = 0
for key, array in instructions.items():
    max_index += len(array)
lobster_buy = [[] for _ in range(max_index)]
lobster_sell = [[] for _ in range(max_index)]
timestamps = []
levels = 100

tic = time.time()
for i, (timestamp, array) in enumerate(instructions.items()):

    if i % 50_000 == 0:
        print(f"Processing order {i}/{max_index}")

    for j, value in enumerate(array):
        # Copy previous state
        if i > 0:
            lobster_buy[i + j] = lobster_buy[i + j - 1].copy()
            lobster_sell[i + j] = lobster_sell[i + j - 1].copy()

        timestamps.append(timestamp)

        if value[0] == "ADD":
            price, display_qty, side = value[1]
            if side == 1:
                lobster = lobster_buy
                price = -price
            else:
                lobster = lobster_sell

            for p, q in lobster[i]:
                if p == price:
                    q += display_qty
                    break
            else:
                if len(lobster[i]) < levels:
                    lobster[i].append((price, display_qty))
                else:
                    # Find the worst price
                    worst_price = max(lobster[i], key=lambda x: x[0])[0]
                    if price < worst_price:
                        # Replace the worst price
                        for j, (p, q) in enumerate(lobster[i]):
                            if p == worst_price:
                                lobster[i][j] = (price, display_qty)
                                break

        elif value[0] == "MODIFY" or value[0] == "MODIFY_SAME_PRIORITY":
            price, display_qty, side, prev_price, prev_display_qty = value[1]
            if side == 1:
                lobster = lobster_buy
                price = -price
            else:
                lobster = lobster_sell

            # Find the order in the heap
            found = False
            for j, (p, q) in enumerate(lobster[i]):
                if p == prev_price:
                    found = True
                    break

            # Modify the order
            if found and price == prev_price:
                lobster[i][j] = (price, lobster[i][j][1] - prev_display_qty + display_qty)
            # elif found:
            #     lobster[i][j] = (prev_price, lobster[i][j][1] - prev_display_qty)
            #
            #     found = False
            #     for j, (p, q) in enumerate(lobster[i]):
            #         if p == price:
            #             found = True
            #             break
            #     else:
            #         # Find the worst price
            #         worst_price = max(lobster[i], key=lambda x: x[0])[0]
            #         if price < worst_price:
            #             # Replace the worst price
            #             for j, (p, q) in enumerate(lobster[i]):
            #                 if p == worst_price:
            #                     lobster[i][j] = (price, display_qty)
            #                     break
            #
            #     if found:
            #         lobster[i][j] = (price, lobster[i][j][1] + display_qty)

        elif value[0] == "DELETE" or value[0] == "FULL_ORDER_EXECUTION" or value[0] == "PARTIAL_ORDER_EXECUTION":
            price, display_qty, side = value[1]
            if side == 1:
                lobster = lobster_buy
                price = -price
            else:
                lobster = lobster_sell

            # Find the order in the heap
            found = False
            for j, (p, q) in enumerate(lobster[i]):
                if p == price:
                    found = True
                    break

            # Delete the order
            if found:
                lobster[i][j] = (price, q - display_qty)
                if lobster[i][j][1] <= 0:
                    del lobster[i][j]

tac = time.time()
print(f"Created orderbook in {tac - tic:.2f} seconds")

# Variable "instructions" is now useless, free up memory
del instructions

# Export to lobster csv
print("Exporting to CSV...")

# Sort the levels correspondingly to the price (depends on buy / sell, best buy price is highest, best sell lowest)
# Be aware that the prices are NEGATIVE for BUY orders -> thus sorting does not have to be REVERSED
for i in range(max_index):
    lobster_buy[i] = sorted(lobster_buy[i])
    lobster_sell[i] = sorted(lobster_sell[i])

# Time,Ask Price 1, Ask Volume 1, Bid Price 1, Bid Volume 1, ...
levels = 30
lobster_header = "Time,"
for i in range(levels):
    lobster_header += f"Ask Price {i + 1},Ask Volume {i + 1},Bid Price {i + 1},Bid Volume {i + 1},"
lobster_header = lobster_header[:-1]  # Remove last comma

OUTPUT_FILE_PATH = "pokus_lobsteru.csv"

tic = time.time()
# Export to CSV
with open(OUTPUT_FILE_PATH, "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(lobster_header.split(","))  # Write header

    # Starting from 10, because first few indices are weird anyways, people trying to buy for cheap and sell for high
    for i in range(10, max_index):
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
                row.extend([f"{-lobster_buy[i][level][0]:.8f}", f"{lobster_buy[i][level][1]:.8f}"])
            else:
                row.extend(["", ""])  # Empty values

        writer.writerow(row)  # Write row to CSV

toc = time.time()
print(f"Exported to CSV in {toc - tic:.2f} seconds")
