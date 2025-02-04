import sys
import csv
import heapq
import time
from datetime import datetime

INT_MIN = -1 * (sys.maxsize + 1)


def get_nansec_from_time(base_date: datetime, timestamp: str):
    """
    @param base_date: Datetime object representing the beginning of the day
    @param timestamp: UTC Timestamp
    Returns number of nanoseconds from the beginning of day (base_date) from UTC Timestamp or INT_MIN if Timestamp invalid
    """
    if timestamp == "NOVALUE":
        return INT_MIN

    timestamp = int(timestamp)
    tstamp_date = datetime.utcfromtimestamp(timestamp // 1e9)
    tstamp_nansec = int(str(int(timestamp % 1e9)).zfill(9))

    difference = tstamp_date - base_date
    res = int(difference.total_seconds() * 1e9 + tstamp_nansec)

    return res


def transform_qty(source_qty: str, qty_rounder: int):
    """
    @param source_qty: Quantity to be transformed
    @param qty_rounder: Multiplier
    Returns quantity multiplied by qty_rounder
    """
    if source_qty == "NOVALUE":
        return "NaN"
    return int(round(float(source_qty)*qty_rounder))


class TaggedLine:
    """
    Class representing each line in all files. Line is defined by operation and line itself
    """
    def __init__(self, line, source_op):
        """
        Constructor
        :param line: Line from file
        :param source_op: Source operation
        """
        self.line = line
        self.source_op = source_op


def tagged_lines(file, source_op, delim):
    """
    Generator function for reading lines from file
    :param file: File to read
    :param source_op: Source operation
    :param delim: Delimiter
    Yields TaggedLines
    """
    reader = csv.DictReader((line.replace('\x00', '') for line in file), delimiter=delim)
    for line in reader:
        yield TaggedLine(line, source_op)


def main():
    """
    Main function
    """
    # Config
    # -----------------------------------
    delim = ','
    output_format_params = ["i", "PARENT_ID", "ID", "TrdRegTSTimeIn", "TrdRegTSTimePriority", "Side", "Price", "DisplayQty", "op", "Trans", "Prio"]
    qty_rounder = 10000

    data_files = {
        sys.argv[1]: "A",
        sys.argv[2]: "M",
        sys.argv[3]: "D",
        sys.argv[4]: "MS",
        sys.argv[5]: "FE",
        sys.argv[6]: "PE",
        sys.argv[7]: "ES",
        # "2350_00_D_03_A_20191202.OrderAdd_FGBL_4128839.csv": "A",
        # "2350_00_D_03_A_20191202.OrderModify_FGBL_4128839.csv": "M",
        # "2350_00_D_03_A_20191202.OrderDelete_FGBL_4128839.csv": "D",
        # "2350_00_D_03_A_20191202.OrderModifySamePrio_FGBL_4128839.csv": "MS",
        # "2350_00_D_03_A_20191202.FullOrderExecution_FGBL_4128839.csv": "FE",
        # "2350_00_D_03_A_20191202.PartialOrderExecution_FGBL_4128839.csv": "PE",
        # # "2350_00_D_03_A_20191202.InstrumentSummary_FGBL_4128839.csv": "IS",
        # "2350_00_D_03_A_20191202.ExecutionSummary_FGBL_4128839.csv": "ES"
    }

    temp_name = list(data_files.keys())[0].split(".")
    date = temp_name[0].split("_")[-1]
    instrument = temp_name[1].split("_")[1]
    security = temp_name[1].split("_")[2]
    base_date = datetime(int(date[:4]), int(date[4:6]), int(date[6:]))

    outfile = date+"-"+instrument+"-"+security+".csv"
    # -----------------------------------

    # Merge sorted data files
    print("Merging files...")
    merge_list = []
    open_files = []
    for filename, op in data_files.items():
        f_input = open(filename)
        open_files.append(f_input)
        merge_list.append(tagged_lines(f_input, op, delim))

    tic = time.perf_counter()
    with open(outfile, 'w', newline="", encoding="utf-8") as output_file:
        csv_output = csv.writer(output_file, delimiter=delim)
        csv_output.writerow(output_format_params)
        sorted_lines = heapq.merge(*merge_list, key=lambda x: (int(x.line["PARENT_ID"]), int(x.line["ID"])) if x.source_op != "PE" else (int(x.line["PARENT_ID"]), get_nansec_from_time(base_date, x.line["TrdRegTSTimePriority"])))

        # Load order book state from previous day (first n Add orders with same TimeIn timestamp)
        i = 1
        first_line = next_line = next(sorted_lines)
        inic_time = first_line.line["TrdRegTSTimeIn"]
        tin_nansec = get_nansec_from_time(base_date, inic_time)

        csv_output.writerow([
            i,
            first_line.line["PARENT_ID"],
            first_line.line["ID"],
            tin_nansec,
            get_nansec_from_time(base_date, first_line.line["TrdRegTSTimePriority"]),
            "B" if first_line.line["Side"] == "1" else "S",
            first_line.line["Price"],
            transform_qty(first_line.line["DisplayQty"], qty_rounder),
            "A",
            tin_nansec - 1,
            first_line.line["TrdRegTSTimePriority"]
        ])

        while True:
            try:
                next_line = next(sorted_lines)

                if next_line.source_op != "A":
                    i += 1
                    break

                if next_line.line["SecurityID"] != security:
                    continue

                i += 1
                cur_time = next_line.line["TrdRegTSTimeIn"]

                if cur_time != inic_time:
                    break

                csv_output.writerow([
                    i,
                    next_line.line["PARENT_ID"],
                    next_line.line["ID"],
                    tin_nansec,
                    get_nansec_from_time(base_date, next_line.line["TrdRegTSTimePriority"]),
                    "B" if next_line.line["Side"] == "1" else "S",
                    next_line.line["Price"],
                    transform_qty(next_line.line["DisplayQty"], qty_rounder),
                    "A",
                    tin_nansec - 1,
                    next_line.line["TrdRegTSTimePriority"]
                ])

            except StopIteration:
                break

        # Load the rest
        while True:
            if next_line.line["SecurityID"] == security:
                lines = []
                if next_line.source_op == "M":  # Modify
                    time_in = get_nansec_from_time(base_date, next_line.line["TrdRegTSTimeIn"])
                    time_prio = get_nansec_from_time(base_date, next_line.line["TrdRegTSTimePriority"])
                    new_line_1 = [  # Delete
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        time_in,
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSPrevTimePriority"]),
                        "B" if next_line.line["Side"] == "1" else "S",
                        next_line.line["PrevPrice"],
                        transform_qty(next_line.line["PrevDisplayQty"], -1 * qty_rounder),
                        "YYY",
                        time_prio,
                        next_line.line["TrdRegTSPrevTimePriority"]
                    ]

                    i += 1
                    new_line_2 = [  # Add
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        time_in,
                        time_prio,
                        "B" if next_line.line["Side"] == "1" else "S",
                        next_line.line["Price"],
                        transform_qty(next_line.line["DisplayQty"], qty_rounder),
                        "XXX",
                        time_prio,
                        next_line.line["TrdRegTSTimePriority"]
                    ]

                    lines = [new_line_1, new_line_2]

                elif next_line.source_op == "A":
                    time_prio = get_nansec_from_time(base_date, next_line.line["TrdRegTSTimePriority"])
                    lines = [[
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSTimeIn"]),
                        time_prio,
                        "B" if next_line.line["Side"] == "1" else "S",
                        next_line.line["Price"],
                        transform_qty(next_line.line["DisplayQty"], qty_rounder),
                        "A",
                        time_prio,
                        next_line.line["TrdRegTSTimePriority"]
                    ]]

                elif next_line.source_op == "D":
                    lines = [[
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSTimeIn"]),
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSTimePriority"]),
                        "B" if next_line.line["Side"] == "1" else "S",
                        next_line.line["Price"],
                        transform_qty(next_line.line["DisplayQty"], -1 * qty_rounder),
                        "D",
                        get_nansec_from_time(base_date, next_line.line["TransactTime"]),
                        next_line.line["TrdRegTSTimePriority"]
                    ]]

                elif next_line.source_op == "MS":
                    lines = [[
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSTimeIn"]),
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSTimePriority"]),
                        "B" if next_line.line["Side"] == "1" else "S",
                        next_line.line["Price"],
                        transform_qty(next_line.line["DisplayQty"], qty_rounder) - transform_qty(next_line.line["PrevDisplayQty"], qty_rounder),
                        "MS",
                        get_nansec_from_time(base_date, next_line.line["TransactTime"]),
                        next_line.line["TrdRegTSTimePriority"]
                    ]]

                elif next_line.source_op == "FE":
                    lines = [[
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        "NOVALUE",
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSTimePriority"]),
                        "B" if next_line.line["Side"] == "1" else "S",
                        next_line.line["Price"],
                        transform_qty(next_line.line["LastQty"], -1 * qty_rounder),
                        "E",
                        INT_MIN,
                        next_line.line["TrdRegTSTimePriority"]
                    ]]

                elif next_line.source_op == "PE":
                    lines = [[
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        "NOVALUE",
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSTimePriority"]),
                        "B" if next_line.line["Side"] == "1" else "S",
                        next_line.line["Price"],
                        transform_qty(next_line.line["LastQty"], -1 * qty_rounder),
                        "PE",
                        INT_MIN,
                        next_line.line["TrdRegTSTimePriority"]
                    ]]

                elif next_line.source_op == "IS":
                    lines = [[
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        "NOVALUE",
                        get_nansec_from_time(base_date, next_line.line["TrdRegTSExecutionTime"]),
                        "",
                        "NaN",
                        next_line.line["TotNoOrders"],
                        "Check",
                        get_nansec_from_time(base_date, next_line.line["LastUpdateTime"]),
                        next_line.line["TrdRegTSExecutionTime"]
                    ]]

                elif next_line.source_op == "ES":
                    lines = [[
                        i,
                        next_line.line["PARENT_ID"],
                        next_line.line["ID"],
                        get_nansec_from_time(base_date, next_line.line["RequestTime"]),
                        "NOVALUE",
                        "B" if next_line.line["AggressorSide"] == "1" else "S",
                        next_line.line["LastPx"],
                        transform_qty(next_line.line["LastQty"], qty_rounder),
                        "Sum",
                        get_nansec_from_time(base_date, next_line.line["ExecID"]),
                        "NOVALUE"
                    ]]

                i += 1
                csv_output.writerows(lines)

            try:
                next_line = next(sorted_lines)

            except StopIteration:
                break

    # Clean
    print("Closing opened files...")

    for file in open_files:
        file.close()
    toc = time.perf_counter()

    print(f"Elapsed time: {(toc - tic):0.9f} seconds")


if __name__ == "__main__":
    main()
