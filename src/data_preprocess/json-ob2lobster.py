import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import json

# Default values
MARKET_ID = "XEUR"
DATE = "20191202"
MARKET_SEGMENT_ID = "688"
SECURITY_ID = "4128839"

# User defined values
if len(sys.argv) == 5:
    MARKET_ID = sys.argv[1]
    DATE = sys.argv[2]
    MARKET_SEGMENT_ID = sys.argv[3]
    SECURITY_ID = sys.argv[4]


def parse_lobster(part, fp, levels=30, part_one=False):
    """
    Parse the order book data from JSON format to CSV format
    :param part: Which part of the downloaded data to parse
    :param fp: File pointer to write the data to
    :param levels: Levels of the order book to parse
    :param part_one: If this is the first part of the data, write the header
    """
    # If this is the first part of the data, write the header
    if part_one:
        header = "Time,"
        for i in range(levels):
            header += f"Ask Price {i + 1},Ask Volume {i + 1},Bid Price {i + 1},Bid Volume {i + 1},"
        header = header[:-1]
        fp.write(f"{header}\n")

    # Parse the data
    for i, entry in enumerate(part):
        ask_bid_zip = f"{entry['Timestamp']},"

        for level in range(levels):

            if level < len(entry['Sell']):
                ask_bid_zip += f"{float(entry['Sell'][level]['Price'])/1e8:.8f},{float(entry['Sell'][level]['Quantity'])/1e8:.8f},"
            else:
                ask_bid_zip += ",,"

            if level < len(entry['Buy']):
                ask_bid_zip += f"{float(entry['Buy'][level]['Price'])/1e8:.8f},{float(entry['Buy'][level]['Quantity'])/1e8:.8f},"
            else:
                ask_bid_zip += ",,"

        ask_bid_zip = ask_bid_zip[:-1]
        fp.write(f"{ask_bid_zip}\n")


print("Loading data...")
with open(f"data/{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_orderbook.json", "r") as fp:
    data = json.load(fp)
print("Data loaded")

fp = open(f"data/{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_orderbook.csv", "w")

for i, part in enumerate(data):
    print(f"Processing part {i + 1}/{len(data)}")

    parse_lobster(part, fp, part_one=(i == 0))

fp.close()
