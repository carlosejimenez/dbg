import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV
import tools
import operator

from market import Market
from lstm import Lstm


if __name__ == '__main__':
    first_day = '2018-12-01'
    end_date = '2019-01-01'

    column = 'Price'
    stock = 'DAI'
    stock2 = 'BMW'
    stock3 = 'SAP'

    p = 10

    # # my_df = tools.make_return_df('BMW', first_day, end_date, interval=60, ignore_missing_data=True, difference=False)
    # my_df = tools.make_price_df(stock, first_day, end_date, interval=60, ignore_missing_data=True)
    # my_df2 = tools.make_price_df(stock2, first_day, end_date, interval=60, ignore_missing_data=True)
    # my_df3 = tools.make_price_df(stock3, first_day, end_date, interval=60, ignore_missing_data=True)
    # merged_df = tools.combine_stock_dfs([my_df, my_df2, my_df3])
    # # x, y = tools.build_ar_x_y(my_df, p=2, column='Price')
    # x, y = tools.build_ar_x_y(merged_df, p, rest=2)
    # model = Lstm(time_steps=p, input_size=3, output_size=3)
    # # x_test, y_test = model.load_and_split_data(x, y, test_ratio=0.15, units=45, purge_ratio=0.05, epochs=10,
    # #                                            batch_size=50)
    # model.load_data(x, y, units=45, epochs=10, batch_size=50)
    # # y_pred = model.predict(x_test, y_test, graph=True, title=f'{stock} Price ({first_day} - {end_date})')
    # y_pred = model.predict(x)
    # print(y_pred)
    #
    def get_ith_col_x(array, i):
        new_array = []
        for arr in array:
            new_array.append([row[i] for row in arr])
        return new_array

    def get_ith_col_y(array, i):
        new_array = []
        for row in array:
            new_array.append(row[i])
        return new_array

    P = {'Index': 1}
    cash = 0

    my_df = tools.make_market_df(first_day, end_date, interval=30, ignore_missing_data=True)
    x_train, x_test = tools.split_dataframe(my_df, test_ratio=0.2)
    market = Market(x_test, p=p)
    x, y = tools.build_ar_x_y(x_train, p=p, rest=2)
    x = np.array(x).reshape((-1, p, 30))
    y = np.array(y).reshape((-1, 30, 1))
    models = [RidgeCV(alphas=[0.2, 0.3, 0.6], cv=6, fit_intercept=True) for i in range(30)]
    for i in range(30):
        models[i].fit(get_ith_col_x(x, i), get_ith_col_y(y, i))

    # model = Lstm(time_steps=p, input_size=30, output_size=30)
    # model.load_data(x, y, units=50, epochs=20, batch_size=50)

    market.buy(P.items())
    trades = []
    for prices in market.iterate(index=False):
        pass
        # y_pred = [m.predict(get_ith_col_x(np.array(prices).reshape((-1, p, 30)), i)) for i, m in enumerate(models)]
        # y_pred = np.array(y_pred).flatten()
        # returns = np.divide(np.subtract(np.array(y_pred), np.array(prices[-1])), np.array(prices[-1]))
        # returns = returns.flatten()
        # max_return = max(returns)
        # if max_return > 0:
        #     max_index = list(returns).index(max_return)
        #     cash = market.liquidate_stocks(market.ticks)
        #     market.buy([(max_index, cash)], with_index=True)
        #     trades.append(max_index)

    cash = market.liquidate_stocks(market.ticks)
    print(cash)
    print(trades)

    # lstms = {tick: Lstm() for tick in market.ticks}
    # for i, key in enumerate(lstms.keys()):
    #     lstms[key].load_data(get_ith_col_x(x, i), get_ith_col_y(y, i), units=50, epochs=10, batch_size=50)
    # for hour in market.iterate():
    #     lstm_estimates = []
    #     for stock, price in hour:
    #         lstm_estimates.append(lstms[stock].predict(price))
    #     print(f'lstm:estimates: {lstm_estimates}')

