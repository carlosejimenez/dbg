import numpy as np
import pandas as pd

from tools import dax


def get_dax_index(stock):
    return list(dax.keys()).index(stock)


class Market:
    def __init__(self, market_df):
        self.market = market_df
        self.alloc = [0 for x in range(29)]
        self.date = None
        self.time = None

    def buy(self, stock_val_tups):
        stock_val_tups = list(stock_val_tups)
        for stock, val in stock_val_tups:
            self.alloc[get_dax_index(stock)] += val

    def get_alloc(self):
        return list(zip(dax.keys(), self.alloc))

    def get_alloc_stock_names(self):
        stocks = []
        for stock, val in self.get_alloc():
            if val > 0:
                stocks.append(stock)
        return stocks

    def iterate(self):
        last_prices = list(self.market.iloc[0][2:-1])
        self.date, self.time = self.market.iloc[0][:2]
        start_date = self.date + self.time
        yield last_prices
        for index, row in self.market.iterrows():
            self.date, self.time = row[:2]
            if self.date + self.time == start_date:
                continue
            return_array = np.divide(np.subtract(list(row[2:-1]), last_prices), list(last_prices))
            last_prices = list(row[2:-1])
            # self.date, self.time = row[:2]
            self.update_alloc(list(return_array))
            yield last_prices

    def sell(self, stock_val_tups):
        sell_value = 0
        stock_val_tups = list(stock_val_tups)
        for stock, val in stock_val_tups:
            sell_value += sell_value
            self.alloc[get_dax_index(stock)] -= val
        return sell_value

    def liquidate_stocks(self, stocks):
        stocks = list(stocks)
        sell_value = sum([self.alloc[get_dax_index(stock)] for stock in stocks])
        for stock in stocks:
            self.alloc[get_dax_index(stock)] = 0
        return sell_value
            
    def update_alloc(self, return_array):
        self.alloc = list(np.add(np.multiply(self.alloc, return_array), self.alloc))
