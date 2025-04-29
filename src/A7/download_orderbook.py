import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime
import json
import requests
import time
import random

if len(sys.argv) != 5:
    print("Usage: python download_eobi.py <MARKET_ID> <DATE> <MARKET_SEGMENT_ID> <SECURITY_ID>")
    print("Example: python download_eobi.py XEUR 20210104 691 5315926")
    exit(1)

MARKET_ID = sys.argv[1]
DATE = sys.argv[2]
MARKET_SEGMENT_ID = sys.argv[3]
SECURITY_ID = sys.argv[4]

if not os.path.exists("a7token.txt"):
    print("Please create a file named 'a7token.txt' with your API token.")
    exit(1)

userId = "zapped99@ntis.zcu.cz"  # Change this to your user ID
with open("a7token.txt", "r") as file:  # Change this to your token file
    API_TOKEN = file.read().rstrip()

url = "https://a7.deutsche-boerse.com/api/v1/"
header = {"Authorization": "Bearer " + API_TOKEN}

# 1e9 is for converting seconds to nanoseconds
start_in_nano = str(int(datetime.strptime(DATE, "%Y%m%d").replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1e9))

levels = 30
part = 0

if not os.path.exists("data"):
    os.makedirs("data")

fp = open(f"data/{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_orderbook.json", "w")
fp.write("[\n")

while True:
    # Request for the first 1 million transact times
    request = f"ob/{MARKET_ID}/{DATE}/{MARKET_SEGMENT_ID}/{SECURITY_ID}?&limit=1000000&from={start_in_nano}&levels={str(levels)}&orderbook=aggregated"

    response = requests.get(url=url + request, headers=header)

    # Check if the response is empty
    if len(response.json()) == 0:
        break

    print(f"Status: {response.status_code}")
    print(f"Downloaded {len(response.json())} entries")

    # Update the "last time" variable for downloading in parts
    previous_start_in_nano = start_in_nano
    print(response.json()[-1])
    start_in_nano = str(int(response.json()[-1]["Timestamp"]) + 1)

    # This should never happen, but just in case
    if previous_start_in_nano > start_in_nano:
        print("Time is not increasing!")

    # Check if the start time is the same as the previous one -> downloaded all transact times
    if previous_start_in_nano == start_in_nano:
        break

    print(f"From {datetime.fromtimestamp(int(previous_start_in_nano) / 1000000000)} to {datetime.fromtimestamp(int(start_in_nano) / 1000000000)}")

    # Write the response to the file
    if response.status_code == 200:
        fp.write(json.dumps(response.json(), indent=1))
        fp.write(",")
    else:
        print("Connection error!")
        exit(1)

    part += 1
    # Politely wait for a random time between 5 and 15 seconds (to avoid bombarding the API with requests)
    time.sleep(5 + random.random() * 10)

print(f"Done downloading order book in {part} parts")

fp.seek(fp.tell() - 1, 0)
fp.write("\n]")
fp.close()

print(f"Done writing to {MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_orderbook.json")
