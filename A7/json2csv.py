import json

MARKET_ID = "XEUR"
# DATE = "20210104"
DATE = "20191202"
# MARKET_SEGMENT_ID = "691"
MARKET_SEGMENT_ID = "688"
# SECURITY_ID = "5315926"
SECURITY_ID = "4128839"

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
with open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_detailed.json", "r") as fp:
    data = json.load(fp)
print("Data loaded")

print("Writing to csv...")
fp_order_add = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_ORDER_ADD.csv", "w")
fp_order_add.write("ID,TrdRegTSTimeIn,SecurityID,TrdRegTSTimePriority,DisplayQty,Side,OrdType,Price\n")
fp_order_modify = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_ORDER_MODIFY.csv", "w")
fp_order_modify.write("ID,TrdRegTSTimeIn,TrdRegTSPrevTimePriority,PrevPrice,PrevDisplayQty,SecurityID,TrdRegTSTimePriority,DisplayQty,Side,OrdType,Price\n")
fp_order_modify_same_priority = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_ORDER_MODIFY_SAME_PRIORITY.csv", "w")
fp_order_modify_same_priority.write("ID,TrdRegTSTimeIn,TransactTime,PrevDisplayQty,SecurityID,TrdRegTSTimePriority,DisplayQty,Side,OrdType,Pad6,Price\n")
fp_order_delete = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_ORDER_DELETE.csv", "w")
fp_order_delete.write("ID,TrdRegTSTimeIn,TransactTime,SecurityID,TrdRegTSTimePriority,DisplayQty,Side,OrdType,Price\n")
fp_order_mass_delete = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_ORDER_MASS_DELETE.csv", "w")
fp_order_mass_delete.write("ID,SecurityID,TransactTime\n")
fp_partial_order_execution = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_PARTIAL_ORDER_EXECUTION.csv", "w")
fp_partial_order_execution.write("ID,Side,OrdType,AlgorithmicTradeIndicator,Pad1,TrdMatchID,Price,TrdRegTSTimePriority,SecurityID,LastQty,LastPx\n")
fp_full_order_execution = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_FULL_ORDER_EXECUTION.csv", "w")
fp_full_order_execution.write("ID,Side,OrdType,AlgorithmicTradeIndicator,Pad1,TrdMatchID,Price,TrdRegTSTimePriority,SecurityID,LastQty,LastPx\n")
fp_execution_summary = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_EXECUTION_SUMMARY.csv", "w")
fp_execution_summary.write("ID,SecurityID,AggressorTime,RequestTime,ExecID,LastQty,AggressorSide,Pad1,TradeCondition,Pad4,LastPx,RestingHiddenQty,RestingCxlQty\n")
fp_auction_best_bid_offer = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_AUCTION_BEST_BID_OFFER.csv", "w")
fp_auction_best_bid_offer.write("ID,TransactTime,SecurityID,BidPx,OfferPx,BidSize,OfferSize,PotentialSecurityTradingEvent,BidOrdType,OfferOrdType,Pad5\n")
fp_auction_clearing_price = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_AUCTION_CLEARING_PRICE.csv", "w")
fp_auction_clearing_price.write("ID,TransactTime,SecurityID,LastPx,LastQty,ImbalanceQty,SecurityTradingStatus,PotentialSecurityTradingEvent,Pad6\n")
fp_top_of_book = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_TOP_OF_BOOK.csv", "w")
fp_top_of_book.write("ID,TransactTime,SecurityID,BidPx,OfferPx,BidSize,OfferSize\n")
fp_instrument_state_change = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_INSTRUMENT_STATE_CHANGE.csv", "w")
fp_instrument_state_change.write("ID,SecurityID,SecurityStatus,SecurityTradingStatus,MarketCondition,FastMarketIndicator,SecurityTradingEvent,SoldOutIndicator,Pad2,TransactTime\n")
fp_trade_report = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_TRADE_REPORT.csv", "w")
fp_trade_report.write("ID,SecurityID,TransactTime,LastQty,LastPx,TrdMatchID,MatchType,MatchSubType,AlgorithmicTradeIndicator,TradeCondition\n")
fp_instrument_summary = open(f"{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_INSTRUMENT_SUMMARY.csv", "w")
fp_instrument_summary.write("ID,SecurityID,LastUpdateTime,TrdRegTSExecutionTime,TotNoOrders,SecurityStatus,SecurityTradingStatus,MarketCondition,FastMarketIndicator,SecurityTradingEvent,SoldOutIndicator,ProductComplex,NoMDEntries,Pad6\n")

ID = 0
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

                fp_order_add.write(f"{ID},{TrdRegTSTimeIn},{SecurityID},{TrdRegTSTimePriority},{DisplayQty:.8f},{Side},{OrdType},{Price:.8f}\n")

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

                fp_order_modify.write(f"{ID},{TrdRegTSTimeIn},{TrdRegTSPrevTimePriority},{PrevPrice:.8f},{PrevDisplayQty:.8f},{SecurityID},{TrdRegTSTimePriority},{DisplayQty:.8f},{Side},{OrdType},{Price:.8f}\n")

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

                fp_order_modify_same_priority.write(f"{ID},{TrdRegTSTimeIn},{TransactTime},{PrevDisplayQty:.8f},{SecurityID},{TrdRegTSTimePriority},{DisplayQty:.8f},{Side},{OrdType},{Pad6},{Price:.8f}\n")

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_DELETE:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                TransactTime = transaction["TransactTime"]
                SecurityID = transaction["SecurityID"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]
                OrdType = transaction["OrderDetails"]["OrdType"] if transaction["OrderDetails"]["OrdType"] is not None else "NOVALUE"
                Price = float(transaction["OrderDetails"]["Price"]) / 1e8

                fp_order_delete.write(f"{ID},{TrdRegTSTimeIn},{TransactTime},{SecurityID},{TrdRegTSTimePriority},{DisplayQty:.8f},{Side},{OrdType},{Price:.8f}\n")

            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MASS_DELETE:
                SecurityID = transaction["SecurityID"]
                TransactTime = transaction["TransactTime"]

                fp_order_mass_delete.write(f"{ID},{SecurityID},{TransactTime}\n")

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

                fp_partial_order_execution.write(f"{ID},{Side},{OrdType},{AlgorithmicTradeIndicator},{Pad1},{TrdMatchID},{Price:.8f},{TrdRegTSTimePriority},{SecurityID},{LastQty:.8f},{LastPx:.8f}\n")

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

                fp_full_order_execution.write(f"{ID},{Side},{OrdType},{AlgorithmicTradeIndicator},{Pad1},{TrdMatchID},{Price:.8f},{TrdRegTSTimePriority},{SecurityID},{LastQty:.8f},{LastPx:.8f}\n")

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

                fp_execution_summary.write(f"{ID},{SecurityID},{AggressorTime},{RequestTime},{ExecID},{LastQty:.8f},{AggressorSide},{Pad1},{TradeCondition},{Pad4},{LastPx:.8f},{RestingHiddenQty},{RestingCxlQty}\n")

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

                fp_auction_best_bid_offer.write(f"{ID},{TransactTime},{SecurityID},{BidPx:.8f},{OfferPx:.8f},{BidSize},{OfferSize},{PotentialSecurityTradingEvent},{BidOrdType},{OfferOrdType},{Pad5}\n")

            elif transaction["MessageHeader"]["TemplateID"] == AUCTION_CLEARING_PRICE:
                TransactTime = transaction["TransactTime"]
                SecurityID = transaction["SecurityID"]
                LastPx = f'{float(transaction["LastPx"]) / 1e8}:.8f' if transaction["LastPx"] is not None else "NOVALUE"
                LastQty = f'{float(transaction["LastQty"]) / 1e8}:.8f' if transaction["LastQty"] is not None else "NOVALUE"
                ImbalanceQty = f'{float(transaction["ImbalanceQty"]) / 1e8}:.8f' if transaction["ImbalanceQty"] is not None else "NOVALUE"
                SecurityTradingStatus = transaction["SecurityTradingStatus"] if transaction["SecurityTradingStatus"] is not None else "NOVALUE"
                PotentialSecurityTradingEvent = transaction["PotentialSecurityTradingEvent"] if transaction["PotentialSecurityTradingEvent"] is not None else "NOVALUE"
                Pad6 = "\x00\x00\x00\x00\x00\x00"

                fp_auction_clearing_price.write(f"{ID},{TransactTime},{SecurityID},{LastPx},{LastQty},{ImbalanceQty},{SecurityTradingStatus},{PotentialSecurityTradingEvent},{Pad6}\n")

            elif transaction["MessageHeader"]["TemplateID"] == TOP_OF_BOOK:
                TransactTime = transaction["TransactTime"]
                SecurityID = transaction["SecurityID"]
                BidPx = float(transaction["BidPx"]) / 1e8
                OfferPx = float(transaction["OfferPx"]) / 1e8
                BidSize = f'{float(transaction["BidSize"]) / 1e8}:.8f' if transaction["BidSize"] is not None else "NOVALUE"
                OfferSize = f'{float(transaction["OfferSize"]) / 1e8}:.8f' if transaction["OfferSize"] is not None else "NOVALUE"

                fp_top_of_book.write(f"{ID},{TransactTime},{SecurityID},{BidPx:.8f},{OfferPx:.8f},{BidSize},{OfferSize}\n")

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

                fp_instrument_state_change.write(f"{ID},{SecurityID},{SecurityStatus},{SecurityTradingStatus},{MarketCondition},{FastMarketIndicator},{SecurityTradingEvent},{SoldOutIndicator},{Pad2},{TransactTime}\n")

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

                fp_trade_report.write(f"{ID},{SecurityID},{TransactTime},{LastQty:.8f},{LastPx:.8f},{TrdMatchID},{MatchType},{MatchSubType},{AlgorithmicTradeIndicator},{TradeCondition}\n")

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

                fp_instrument_summary.write(f"{ID},{SecurityID},{LastUpdateTime},{TrdRegTSExecutionTime},{TotNoOrders},{SecurityStatus},{SecurityTradingStatus},{MarketCondition},{FastMarketIndicator},{SecurityTradingEvent},{SoldOutIndicator},{ProductComplex},{NoMDEntries},{Pad6}\n")

            ID += 1

fp_order_add.close()
fp_order_modify.close()
fp_order_modify_same_priority.close()
fp_order_delete.close()
fp_order_mass_delete.close()
fp_partial_order_execution.close()
fp_full_order_execution.close()
fp_execution_summary.close()
fp_auction_best_bid_offer.close()
fp_auction_clearing_price.close()
fp_top_of_book.close()
fp_instrument_state_change.close()
fp_trade_report.close()
fp_instrument_summary.close()

print("Done writing to csv")
