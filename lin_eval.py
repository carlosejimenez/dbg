import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV
from datetime import datetime
import tools
import operator

from market import Market
from lstm import Lstm


if __name__ == '__main__':
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
    first_day = '2019-01-01'
    end_date = '2019-02-01'
    p = 4
    interval = 60

    P = {'Index': 1}
    cash = 0
    my_df = tools.make_market_df(first_day, end_date, interval=interval, ignore_missing_data=True)
    dates = list(map(datetime.fromisoformat,
                     [str(x) + ' ' + str(y) for x, y in
                      list(my_df.get(['Date', 'Time']).itertuples(index=False, name=None))]))

    x_train, x_test, x_purge = tools.split_dataframe(my_df, test_ratio=0.2, purge_ratio=0.1)
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
    y_preds = []
    cash_y = []
    cash_x = []
    for prices in market.iterate(tick_names=False):
        y_pred = [m.predict(get_ith_col_x(np.array(prices).reshape((-1, p, 30)), i)) for i, m in enumerate(models)]
        y_pred = np.array(y_pred).flatten()
        y_preds.append(y_pred)
        returns = np.divide(np.subtract(np.array(y_pred), np.array(prices[-1])), np.array(prices[-1]))
        returns = returns.flatten()
        max_return = max(returns)
        max_index = list(returns).index(max_return)
        cash = market.liquidate_stocks(market.ticks)
        cash_y.append(cash)
        cash_x.append(datetime.fromisoformat(str(market.date) + ' ' + str(market.time)))
        market.buy([(max_index, cash)], with_index=True)
        trades.append(max_index)

    cash = market.liquidate_stocks(market.ticks)
    print(cash)

    x_test, y_test = tools.build_ar_x_y(x_test, p=p, rest=2)
    x_purge, y_purge = tools.build_ar_x_y(x_purge, p=p, rest=2)
    for i in range(0, 30):
        yt = np.array(get_ith_col_y(np.array(y).reshape(-1, 30), i))
        ypurge = [None]*len(yt) + list(np.array(get_ith_col_y(np.array(y_purge).reshape((-1, 30)), i))) + [None]*len(y_preds)
        yy = np.array(get_ith_col_y(np.array(y_test).reshape((-1, 30)), i))
        yt = list(yt) + [None]*len(y_purge)
        yp = [None]*len(yt)
        yt.extend(list(yy))
        yp = yp + list(np.array(get_ith_col_y(np.array(y_preds).reshape((-1, 30)), i)))
        plt.plot_date(x=dates[:len(yt)], y=yt, fmt="r-", label=f'True y')
        plt.plot_date(x=dates[:len(yp)], y=yp, fmt="b-", label=f'Predicted y')
        plt.plot_date(x=dates[:len(ypurge)], y=ypurge[:len(dates)], fmt='g-', label=f'Purged data')
        # plt.plot(x=dates, y=yt, color='navy', label=f'True y')
        # plt.plot(x=dates, y=yp[:-1], color='red', label=f'Predicted y')
        plt.title(f'Ridge: {market.get_tick_from_index(i)} - ({first_day} - {end_date}), p = {p}, interval = {interval}')
        plt.xlabel('Time')
        plt.ylabel(f'y')
        plt.legend()
        plt.show()
        plt.clf()

    plt.clf()
    plt.plot_date(x=cash_x, y=cash_y, fmt="y-", label=f'Monetary value')
    plt.title(f'Portfolio value - ({first_day} - {end_date}), p = {p}, interval = {interval}')
    plt.xlabel('Time')
    plt.ylabel(f'value')
    plt.legend()
    plt.show()
    plt.clf()

    # lstms = {tick: Lstm() for tick in market.ticks}
    # for i, key in enumerate(lstms.keys()):
    #     lstms[key].load_data(get_ith_col_x(x, i), get_ith_col_y(y, i), units=50, epochs=10, batch_size=50)
    # for hour in market.iterate():
    #     lstm_estimates = []
    #     for stock, price in hour:
    #         lstm_estimates.append(lstms[stock].predict(price))
    #     print(f'lstm:estimates: {lstm_estimates}')

