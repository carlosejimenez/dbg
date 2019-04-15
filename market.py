import numpy as np
import pandas as pd

from tools import dax


class Market:
    def __init__(self, market_df, p=1):
        self.p = p
        self.market = market_df
        self.alloc = [0 for x in range(30)]
        self.ticks = list(dax.keys()) + ['Index']

        self.date = None
        self.time = None

    def buy(self, stock_val_tups, with_index=False):
        stock_val_tups = list(stock_val_tups)
        for stock, val in stock_val_tups:
            if with_index:
                self.alloc[stock] += val
            else:
                self.alloc[self.ticks.index(stock)] += val

    def get_alloc(self):
        return list(zip(self.ticks, self.alloc))

    def get_alloc_stock_names(self):
        stocks = []
        for stock, val in self.get_alloc():
            if val > 0:
                stocks.append(stock)
        return stocks

    def get_tick_from_index(self, index):
        assert 0 < index < len(self.ticks), f'{index} out of reasonable bounds. Only {len(self.ticks)} stocks on market.'
        return self.ticks[index]

    def iterate(self, tick_names=True):
        last_prices = [row[2:] for row in list(self.market.itertuples(index=False, name=None))[:self.p]]
        self.date, self.time = self.market.iloc[self.p][:2]
        if tick_names:
            yield list(map(lambda x: [(s, t) for s, t in zip(self.ticks, x)], last_prices))
        else:
            yield last_prices
        for row in list(self.market.itertuples(index=False, name=None))[self.p:]:
            return_array = np.divide(np.subtract(list(row[2:]), last_prices[-1]), list(last_prices[-1]))
            self.update_alloc(list(return_array))
            last_prices = last_prices[1:] + [row[2:]]
            self.date, self.time = row[0:2]
            if tick_names:
                yield list(map(lambda x: [(s, t) for s, t in zip(self.ticks, x)], last_prices))
            else:
                yield last_prices

    def sell(self, stock_val_tups, with_index=False):
        stock_val_tups = list(stock_val_tups)
        for stock, val in stock_val_tups:
            if with_index:
                self.alloc[stock] -= val
            else:
                self.alloc[self.ticks.index(stock)] -= val
        return sum([v for s, v in stock_val_tups])

    def liquidate_stocks(self, stocks, with_index=False):
        stocks = list(stocks)
        if with_index:
            sell_value = sum([self.alloc[stock] for stock in stocks])
        else:
            sell_value = sum([self.alloc[self.ticks.index(stock)] for stock in stocks])
        for stock in stocks:
            if with_index:
                self.alloc[stock] = 0
            else:
                self.alloc[self.ticks.index(stock)] = 0
        return sell_value
            
    def update_alloc(self, return_array):
        self.alloc = list(np.add(np.multiply(self.alloc, return_array), self.alloc))
