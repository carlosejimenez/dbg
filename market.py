import numpy as np
import pandas as pd

from tools import dax


def get_dax_index(stock):
    return list(dax.keys()).index(stock)


class Market:
    def __init__(self, market_df):
        self.market = market_df
        self.alloc = [0 for x in range(29)]

    def buy(self, vols):
        for stock, vol in vols:
            self.alloc[get_dax_index(stock)] += vol

    def sell(self, vols):
        for stock, vol in vols:
            self.alloc[get_dax_index(stock)] -= vol


