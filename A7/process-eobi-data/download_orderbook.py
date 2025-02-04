from datetime import datetime
import json
import requests
import sys
import time
import random

print(sys.argv)
# MARKET_ID = "XEUR"
MARKET_ID = sys.argv[1]
# DATE = "20210104"
# DATE = "20191202"
DATE = sys.argv[2]
# MARKET_SEGMENT_ID = "691"
# MARKET_SEGMENT_ID = "688"
MARKET_SEGMENT_ID = sys.argv[3]
# SECURITY_ID = "5315926"
# SECURITY_ID = "4128839"
SECURITY_ID = sys.argv[4]

userId = "zapped99@ntis.zcu.cz"
with open('a7token.txt', 'r') as file:
    API_TOKEN = file.read().rstrip()

url = 'https://a7.deutsche-boerse.com/api/v1/'
header = {'Authorization': 'Bearer ' + API_TOKEN}

start_in_nano = str(int(datetime.strptime(DATE, "%Y%m%d").replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000 * 1000 * 1000))
levels = 30

part = 0

fp = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_orderbook.json", "w")
fp.write("[\n")
while True:
    request = f"ob/{MARKET_ID}/{DATE}/{MARKET_SEGMENT_ID}/{SECURITY_ID}?&limit=1000000&from={start_in_nano}&levels={str(levels)}&orderbook=aggregated"
    response = requests.get(url=url + request, headers=header)
    if len(response.json()) == 0:
        break

    print(f"Status: {response.status_code}")
    print(f"Downloaded {len(response.json())} entries")

    previous_start_in_nano = start_in_nano
    print(response.json()[-1])
    start_in_nano = str(int(response.json()[-1]["Timestamp"]) + 1)

    if previous_start_in_nano > start_in_nano:
        print("Time is not increasing!")
    if previous_start_in_nano == start_in_nano:
        break

    print(f"From {datetime.fromtimestamp(int(previous_start_in_nano) / 1000000000)} to {datetime.fromtimestamp(int(start_in_nano) / 1000000000)}")

    if response.status_code == 200:
        fp.write(json.dumps(response.json(), indent=1))
        fp.write(",")
    else:
        print("Connection error!")
        exit(1)

    part += 1
    time.sleep(5 + random.random() * 10)

print(f"Done downloading order book in {part} parts")

fp.seek(fp.tell() - 1, 0)
fp.write("\n]")
fp.close()

print(f"Done writing to {MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_orderbook.json")
