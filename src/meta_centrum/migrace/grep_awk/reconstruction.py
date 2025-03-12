import pandas as pd
import numpy as np
import datetime
import sys


class Config:
    """
    Configuration settings and useful functions
    """
    # Server config
    addr = "127.0.0.1"
    port = 1234

    # Data config
    df_cols = ["Price", "DisplayQty", "Q", "od", "do", "Trans", "Prio"]
    df_cols_type = {"Price": np.float64, "DisplayQty": np.int64, "Q": np.int64, "od": np.int64, "do": np.int64, "Trans": np.int64, "Prio": str}
    delim = ","

    @staticmethod
    def calc_nansec_from_time(time: str) -> int:
        """
        :param time: time in format hh:mm:ss.nnnnnnnnn or hh:mm:ss
        Returns number of nanoseconds from time in format hh:mm:ss.nnnnnnnnn or hh:mm:ss
        """
        time = time.split(":")
        nansec = time[-1].split(".")
        return int(int(time[0]) * 36e11 + int(time[1]) * 6e10 + int(nansec[0]) * 1e9 + (int(nansec[1].ljust(9, "0")) if len(nansec) == 2 else 0))

    @staticmethod
    def calc_time_from_nansec(nansecs: int) -> str:
        """
        :param nansecs: number of nanoseconds
        Returns time in format hh:mm:ss.nnnnnnnnn from given nanoseconds
        """
        s = nansecs // 1e9
        delta = datetime.timedelta(seconds=s)
        ns = str(int(nansecs % 1e9)).zfill(9)
        datetime_obj = (datetime.datetime.min + delta).time()
        time_formatted = datetime_obj.strftime('%H:%M:%S')
        time = time_formatted + "." + ns
        return time


class OB:
    """
    Order book
    """
    def __init__(self, instrument, security, date) -> None:
        """
        Constructor
        :param instrument: Instrument
        :param security: Security ID
        :param date: Date
        """
        self.__instrument = instrument
        self.__security = security
        self.__date = date
        self.__data = pd.DataFrame()
        self.__min_timestamp = 0  # First timestamp of day
        self.__timestamp = 0
        self.__bookA = pd.DataFrame()
        self.__bookB = pd.DataFrame()
        self.__executes = {}
        self.__changed = False  # True if new data source valid, False otherwise (used in API func)
        self.change_data_df(self.__instrument, self.__security, self.__date)
    
    def get_instrument(self):
        return self.__instrument

    def get_security(self):
        return self.__security

    def get_date(self):
        return self.__date

    def get_timestamp(self):
        return self.__timestamp

    def get_bookA(self):
        return self.__bookA

    def get_bookB(self):
        return self.__bookB

    def get_executes(self):
        return self.__executes

    def get_changed(self):
        return self.__changed
    
    def set_timestamp(self, timestamp):
        self.__timestamp = timestamp
        
    def change_data_df(self, instrument, security, date) -> None:
        """
        Load new dataframe for given instrument, securityID, date and calculate initial OB
        Keeps previous state if new file doesnt exist
        :param instrument: Instrument
        :param security: Security ID
        :param date: Date
        """
        try:
            temp = pd.read_csv(f"{date}-{instrument}-{security}-ob.csv", delimiter=Config.delim, usecols=Config.df_cols, dtype=Config.df_cols_type)
            if not temp.empty:
                self.__data = temp
                self.__instrument = instrument
                self.__security = security
                self.__date = date
                self.__timestamp = int(self.__data.iloc[0]["Trans"])  # Set inital time (first time of day)
                self.__min_timestamp = self.__timestamp
                self.__changed = True
        except Exception as e:
            print(e)
            self.__changed = False
    
    def calc_order_book_state(self, seq) -> None:
        """
        Calculate order book for given time sequence
        :param seq: [id1, id2], id2 - index of row with timestamp to which the OB should be reconstructed
        """
        # Select all order executes in time sequence
        executes = self.__data.loc[seq[0]:seq[1]-2]
        executes = executes.loc[executes["Trans"] < 0, ["Price", "Q", "Prio"]]
        if not executes.empty:
            executes.rename(columns={'Q': 'Qty'}, inplace=True)
            executes['Type'] = executes['Price'].apply(lambda x: 'B' if x > 0 else 'A')
            executes['Qty'] = executes['Qty'] * -1
            executes.loc[executes['Type'] == 'A', 'Price'] *= -1

        od = seq[1]

        # Select rows valid at od
        ind = (self.__data['od'] <= od) & (self.__data['do'] > od)
        p = self.__data.loc[ind, 'Price'].values
        q = self.__data.loc[ind, 'DisplayQty'].values
        
        # Sort by price
        sorted_indices = np.argsort(p)
        p = p[sorted_indices]
        q = q[sorted_indices]
        
        # Separate bid and ask orders
        indB = (q > 0) & (p > 0)
        bookB = pd.DataFrame({'Price': p[indB], 'Qty': q[indB]})
        bookB = bookB.sort_values("Price", ascending=False)
        bookB.reset_index(drop=True, inplace=True)

        indA = (q > 0) & (p < 0)
        bookA = pd.DataFrame({'Price': -p[indA], 'Qty': q[indA]})
        bookA = bookA.sort_values("Price", ascending=True)
        bookA.reset_index(drop=True, inplace=True)

        self.__bookA = bookA
        self.__bookB = bookB
        self.__executes = executes.to_dict("records")

    def get_time_seq(self, time):
        """
        Returns time sequence and timestamp for nearest smaller timestamp to given time in dataframe.
        Sequence: [id1,id2], id2 - index of row with timestamp to which the OB should be reconstructed,
        id1 - index of row with first positive timestamp before id2. Timestamp - timestamp of row with id2
        :param time: timestamp
        """
        seq, tstamp = [], 0
        if time < self.__min_timestamp:
            time = self.__min_timestamp

        temp = (self.__data['Trans'] <= time) & (self.__data['Trans'] > 0)
        rows = self.__data.loc[temp]
        rows = rows[(~rows.duplicated(["Trans"], keep="last"))].tail(2)

        if len(rows) == 2:
            seq.append(rows.iloc[[0]]["od"].values[0])
            seq.append(rows.iloc[[1]]["od"].values[0])
            tstamp = rows.iloc[[1]]["Trans"].values[0]
        else:
            od = rows.iloc[[0]]["od"].values[0]
            seq = [od, od]
            tstamp = rows.iloc[[0]]["Trans"].values[0]

        return seq, tstamp

    def get_prev_time_seq(self, time):
        """
        Returns time sequence and timestamp for nearest smaller timestamp to currently selected timestamp in dataframe
        Sequence: [id1,id2], id2 - index of row with timestamp to which the OB should be reconstructed, 
        id1 - index of row with first positive timestamp before id2. Timestamp - timestamp of row with id2
        :param time: timestamp
        """
        seq, tstamp = [], 0
        temp = (self.__data['Trans'] <= time) & (self.__data['Trans'] > 0)
        rows = self.__data.loc[temp]
        rows = rows[(~rows.duplicated(["Trans"], keep="last"))]

        if len(rows) <= 2:
            od = rows.iloc[[0]]["od"].values[0]
            seq = [od, od]
            tstamp = rows.iloc[[0]]["Trans"].values[0]
        else:    
            rows = rows.iloc[-3:-1]
            seq.append(rows.iloc[[0]]["od"].values[0])
            seq.append(rows.iloc[[1]]["od"].values[0])
            tstamp = rows.iloc[[1]]["Trans"].values[0]

        return seq, tstamp

    def get_next_time_seq(self, time):
        """
        Returns time sequence and timestamp for nearest bigger timestamp to currently selected timestamp in dataframe
        Sequence: [id1,id2], id2 - index of row with timestamp to which the OB should be reconstructed, 
        id1 - index of row with first positive timestamp before id2. Timestamp - timestamp of row with id2
        :param time: timestamp
        """
        seq, tstamp = [], 0
        temp = (self.__data['Trans'] <= time) & (self.__data['Trans'] > 0)
        start = self.__data.loc[temp].tail(1)
        seq.append(start["od"].values[0])

        temp = (self.__data['Trans'] > time) & (self.__data['Trans'] > 0)
        rows = self.__data.loc[temp]
        rows = rows[(~rows.duplicated(["Trans"], keep="last"))]

        if rows.empty:
            seq.append(seq[0])
            tstamp = start["Trans"].values[0]
        else:
            stop = rows.head(1)
            seq.append(stop["od"].values[0])
            tstamp = stop["Trans"].values[0]

        return seq, tstamp


def main():
    """
    Main function
    """
    # Instrument, security, date
    # date = "20191202"
    date = sys.argv[1]
    # instrument = "FGBL"
    instrument = sys.argv[2]
    # security = "4128839"
    security = sys.argv[3]

    orderbook = OB(instrument, security, date)
    level_depth = 30
    header = ["Time"] + [f"Ask Price {i+1}" for i in range(level_depth)] + [f"Ask Volume {i+1}" for i in range(level_depth)] + [f"Bid Price {i+1}" for i in range(level_depth)] + [f"Bid Volume {i+1}" for i in range(level_depth)]
    lobster_data = []

    curr_time = orderbook.get_timestamp()
    seq, tstamp = orderbook.get_time_seq(curr_time)
    orderbook.calc_order_book_state(seq)
    next_seq, next_time = orderbook.get_next_time_seq(curr_time)
    while next_time != curr_time:
        A = orderbook.get_bookA()
        B = orderbook.get_bookB()

        lobster_data.append([curr_time] + list(A["Price"].head(level_depth)) + list(A["Qty"].head(level_depth)) + list(B["Price"].head(level_depth)) + list(B["Qty"].head(level_depth)))

        orderbook.calc_order_book_state(next_seq)
        curr_time = next_time
        next_seq, next_time = orderbook.get_next_time_seq(curr_time)

    lobster = pd.DataFrame(lobster_data, columns=header)
    print(lobster_data)
    lobster.to_csv(f"{date}-{instrument}-{security}-lobster.csv", index=False)


if __name__ == '__main__':
    main()
