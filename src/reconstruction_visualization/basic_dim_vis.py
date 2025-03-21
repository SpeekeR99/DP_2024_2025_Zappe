import os
import pandas
import matplotlib.pyplot as plt

DATE = "20191202"
MARKET_SEGMENT_ID = "688"
SECURITY_ID = "4128839"

FILE_PATH = f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_lobster_augmented.csv"
if not os.path.exists(f"img/features/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}"):
    os.makedirs(f"img/features/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}")

data = pandas.read_csv(FILE_PATH)

# Visualize all the columns in the dataset with "Time" on the x-axis
for column in data.columns:
    if column == "Time" or "Ask Price" in column or "Bid Price" in column or "Ask Volume" in column or "Bid Volume" in column:
        if column != "Ask Price 1" and column != "Bid Price 1" and column != "Ask Volume 1" and column != "Bid Volume 1":
            continue
    plt.figure(figsize=(20, 10))
    plt.title(column)

    plt.scatter(data["Time"], data[column], color="black", label=column)

    plt.grid()
    plt.savefig(f"img/features/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}/{column}.png")
    plt.show()
