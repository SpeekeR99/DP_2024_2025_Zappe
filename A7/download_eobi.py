import sys
from datetime import datetime
import json
import requests

# XETR 20211217 52408 2504501
# MARKET_ID = "XEUR"
# MARKET_ID = sys.argv[1]
MARKET_ID = "XETR"
# DATE = "20210104"
# DATE = "20191202"
# DATE = sys.argv[2]
DATE = "20211217"
# DATE = "20210319"
# MARKET_SEGMENT_ID = "691"
# MARKET_SEGMENT_ID = "688"
# MARKET_SEGMENT_ID = sys.argv[3]
MARKET_SEGMENT_ID = "52408"
# MARKET_SEGMENT_ID = "1176"
# SECURITY_ID = "5315926"
# SECURITY_ID = "4128839"
# SECURITY_ID = sys.argv[4]
SECURITY_ID = "2504501"
# SECURITY_ID = "5337760"

userId = "zapped99@ntis.zcu.cz"
with open('a7token.txt', 'r') as file:
    API_TOKEN = file.read().rstrip()

url = 'https://a7.deutsche-boerse.com/api/v1/'
header = {'Authorization': 'Bearer ' + API_TOKEN}

last_time = str(int(datetime.strptime(DATE, "%Y%m%d").replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000 * 1000 * 1000))

part = 0

fp = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_detailed.json", "w")
fp.write("[\n")
while True:
    request = f"eobi/{MARKET_ID}/{DATE}/{MARKET_SEGMENT_ID}/{SECURITY_ID}?mode=reference&limit=1000000&from={last_time}"
    request_detailed = f"eobi/{MARKET_ID}/{DATE}/{MARKET_SEGMENT_ID}/{SECURITY_ID}?mode=detailed&limit=1000000&from={last_time}"

    response = requests.get(url=url + request, headers=header)
    if len(response.json()["TransactTimes"]) == 0:
        break

    response_detailed = requests.get(url=url + request_detailed, headers=header)

    print(f"Status: {response.status_code} and {response_detailed.status_code}")
    print(f"Downloaded {len(response.json()['TransactTimes'])} transact times")

    previous_last_time = last_time
    last_time = str(int(response.json()["TransactTimes"][-1]) + 1)

    if previous_last_time > last_time:
        print("Time is not increasing!")
    if previous_last_time == last_time:
        break

    print(f"From {datetime.fromtimestamp(int(previous_last_time) / 1000000000)} to {datetime.fromtimestamp(int(last_time) / 1000000000)}")

    if response.status_code == 200 and response_detailed.status_code == 200:
        fp.write(json.dumps(response_detailed.json(), indent=1))
        fp.write(",")
    else:
        print("Connection error!")
        exit(1)

    part += 1

print(f"Done downloading transact times in {part} parts")

fp.seek(fp.tell() - 1, 0)
fp.write("\n]")
fp.close()

print(f"Done writing to {MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_detailed.json")
