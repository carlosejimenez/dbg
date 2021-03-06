import numpy as np
import os
import pandas
from sklearn import linear_model
import requests
import statistics
from collections import namedtuple
from threading import Lock
from itertools import chain

from collections import OrderedDict
from datetime import datetime, timedelta

dax = OrderedDict([('WDI', 'DE0007472060'), ('DPW', 'DE0005552004'), ('DBK', 'DE0005140008'), ('RWE', 'DE0007037129'),
                   ('VNA', 'DE000A1ML7J1'), ('LHA', 'DE0008232125'), ('DB1', 'DE0005810055'), ('TKA', 'DE0007500001'),
                   ('EOAN', 'DE000ENAG999'), ('BAS', 'DE000BASF111'), ('MUV2', 'DE0008430026'), ('IFX', 'DE0006231004'),
                   ('BEI', 'DE0005200000'), ('1COV', 'DE0006062144'), ('CON', 'DE0005439004'), ('SAP', 'DE0007164600'),
                   ('HEI', 'DE0006047004'), ('SIE', 'DE0007236101'), ('HEN3', 'DE0006048432'), ('ALV', 'DE0008404005'),
                   ('VOW3', 'DE0007664039'), ('BAYN', 'DE000BAY0017'), ('ADS', 'DE000A1EWWW0'), ('FRE', 'DE0005785604'),
                   ('DAI', 'DE0007100000'), ('FME', 'DE0005785802'), ('DTE', 'DE0005557508'), ('BMW', 'DE0005190003'),
                   ('MRK', 'DE0006599905')])

urls = {'xetra': 'https://api.developer.deutsche-boerse.com/prod/xetra-public-data-set/1.0.0/xetra',
        'eurex': 'https://api.developer.deutsche-boerse.com/prod/eurex-public-data-set/1.0.0/eurex'}

data_columns = {'xetra': ['Mnemonic', 'Date', 'Time', 'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'TradedVolume',
                          'NumberOfTrades'],
                'eurex': ['Isin', 'SecurityType', 'MaturityDate', 'StrikePrice', 'PutOrCall', 'Date', 'Time',
                          'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'NumberOfContracts', 'NumberOfTrades']}

holidays = ['2018-01-01', '2018-07-13', '2018-03-30', '2018-04-02', '2018-05-01', '2018-05-21', '2018-10-03', '2018-12-25', '2018-12-26',
            '2019-01-01',

            # '2017-06-19', '2017-06-20', '2017-06-21', '2017-06-22', '2017-06-23', '2017-06-26', '2017-06-27',
            # '2017-06-28', '2017-06-29', '2017-06-30', '2017-10-03', '2017-10-23', '2017-10-24', '2017-10-30',
            # '2017-10-31', '2017-12-25', '2017-12-26', '2018-01-01', '2018-07-13'
            ]
            #todo remove these

lock = Lock()


def build_x_y(df, window, alpha, column='Return'):
    """
    Given a returns dataframe, we construct the EMA, SMA feature vectors returned as X, and the associated labels,
    offset 1, returned as Y.
    :param return_df: returns dataframe
    :param window: window for EMA, SMA
    :param alpha: alpha for EMA
    :param column: Column to build EMA and SMA over. Default "Return", can change to "Price".
    :return: X, Y tuple
    """
    assert type(df) == pandas.DataFrame

    ema_df = make_ema_df(df, window, alpha, column=column)
    sma_df = make_sma_df(df, window, column=column)

    df = df.loc[window:]
    ema_df = ema_df.loc[:len(ema_df) - 2]  # offset 1
    sma_df = sma_df.loc[:len(sma_df) - 2]  # offset 1

    assert len(df) == len(ema_df) == len(sma_df)
    return_df = df.reset_index(drop=True)

    Y = list(return_df[column])
    X = list(zip(ema_df['EMA'], sma_df['SMA']))

    return X, Y


def build_ar_x_y(array, p, lag=1, column=None, rest=0):
    """
    Makes an AR(p) process dataset, for a single array, dataframe, or dataframe column if it is specified.
    Essentially, this produces a shifted lag dataset.
    :param array: Array, or dataframe on which to construct the dataset.
    :param p: AR(p), the number of values contained in the dependent vars vector.
    :param column: optional, use for dataset parameter; which column should we construct X, Y for?
    :param rest: For use with dataframes. used to cut first few values from dataframe. (ie date/time for market_df)
    :return: two lists, x and y, where x is a list of vectors, and y is the associated labels.
    """
    lag -= 1  # The list splicing offsets us by one already.
    if column:
        array = array[column].tolist()
    elif type(array) is pandas.DataFrame:
        array = [row[rest:] for row in list(array.itertuples(index=False, name=None))[0:]]
    else:
        array = array.tolist()
    assert p < len(array)
    x = []
    y = []
    for i in range(p+lag, len(array)):
        y.append(array[i])
        x.append(np.array(array[i-p-lag:i]).flatten())
    return x, y


def col_index(column):
    """
    Get index of a column name based upon the 'xetra' data_columns, used to write files.
    :param column: string name of column to get index
    :return: integer (index of column in 'xetra' data_columns.
    """
    if column in data_columns['xetra']:
        return data_columns['xetra'].index(column)
    else:
        return None


def download(date, api, api_key, dirpath, stock_query=None):
    """download feather archive files for all DAX stocks from Xetra for a particular date (YYYY-MM-DD).
    downloaded data schema is 'Mnemonic', 'Date', 'Time', 'StartPrice', 'MaxPrice',
     'MinPrice', 'EndPrice', 'TradedVolume', 'NumberOfTrades'.

     :param date: Date as YYYY-MM-DD
     :param api: api to query
     :param api_key: key to use for api
     :param dirpath: Path to save price data.
     :param stock_query: Optional, can be str, e.g. 'BMW'; or a list of of mnemonics, e.g. ['BMW', 'SAP'], default is
     the DAX index.

    stdout is stocks that failed to write, most likely from an api error.
    """
    dirpath = './' + dirpath + api + '/'
    os.makedirs(dirpath, exist_ok=True)
    url = urls[api]
    columns = data_columns[api]

    if stock_query:
        if type(stock_query) is str:
            stock_query = [stock_query]
    else:
        stock_query = dax

    for stock_name in stock_query:
        filename = dirpath + f'{stock_name}-{date}.feather'
        if os.path.isfile(filename):
            continue

        headers = {
            'X-DBP-APIKEY': api_key,
        }

        params = (
            ('date', f'{date}'), ('limit', 1000), ('isin', f'{dax[stock_name]}')
        )

        try:
            lock.acquire()
            response = requests.get(url, headers=headers, params=params)
            lock.release()
            if response.status_code is not 200:
                print(f'\nresponse not 200, instead {response.status_code}, raising exception')
                raise Exception
            print(f'\rSaving file {stock_name} for {date}', end='')
            response_trimmed = [{i: minute[i] for i in columns} for minute in response.json()]
            df = pandas.DataFrame(response_trimmed, columns=columns)
            df.to_feather(filename)
            response.close()
        except:
            print(f'\r{api}-{stock_name} failed to write for date {date}')


def get_signed_ratio(predictions, actuals):
    """
    Given predictions and the actual values of those predictions comes up with the signed ratio as a form of evaluation.
    :param predictions: a list of predictions.
    :param actuals: a list of measured results to compare against the actuals.
    :return: a ratio of how frequently the predictions and the actuals share the same sign.
    """
    ratio = 0
    if len(predictions) != len(actuals):
        raise ValueError(f'length of predictions not equal, {len(predictions)} != {len(actuals)}')
    for predict, act in zip(predictions, actuals):
        if predict > 0 and act > 0:
            ratio += 1
        elif predict < 0 and act < 0:
            ratio += 1
    return ratio /len(actuals)


def make_ema_df(data, window, alpha, column='Return'):
    """Given a return dataframe, returns the ema for each point in the dataframe based on parameters.

    :param data: a return dataframe
    :param window: int, ema's moving window
    :param alpha: weight in (0, 1)
    :param column: Column name, default "Return", can change to "Price"
    :return: ema dataframe with original dataframe - window points
    """
    if len(data) < window:
        raise ValueError(f'Data input shorter than window, ema cannot be computed.')
    if not 0 <= alpha <= 1:
        raise ValueError(f'alpha: {alpha} not in bounds of (0, 1)')
    # splicing does NOT work the same in pandas as in Python. Pandas include spliced value when head splicing.
    ema = [statistics.mean(data[column].loc[:window-1])]
    # window-2: because we already added a value to ema.
    ema_df = data[['Mnemonic', 'Date', 'Time']].loc[window-1:]
    # Pandas excludes the spliced value whe tail splicing.
    returns = data[column].loc[window:]
    for value in returns:
        ema_point = alpha * value + (1-alpha) * ema[-1]
        ema.append(ema_point)

    # resets index for dataframe so we can iterate from 0
    ema_df = ema_df.reset_index(drop=True)
    ema_df['EMA'] = ema
    return ema_df

def make_market_df(start='2019-01-01', end=None, interval=30, ignore_missing_data=False):
    global stock, market_index
    mnemonics = dax.keys()
    securities = []
    for stock in mnemonics:
        sec = make_price_df(stock, start, end, interval, ignore_missing_data)
        securities.append(sec)
    market_index = make_index_price_df(*securities)
    return market_index

def make_market_return_df(market_index):
    stocks = seperate_index(market_index)
    return_df = []
    for stock in stocks:
        return_df.append(price_to_return_df(stock))
    market_index_return = combine_stock_dfs(return_df, 'Return')
    return market_index_return

def make_price_df(stock, start, end=None, interval=30, ignore_missing_data=False, dirpath='./'):
    """Given a stock, a start date, end date optional, we return a prices dataframe with column headings
    ['Mnemonic', 'Date', 'Time', 'Price'].
    :param stock: Mnemonic string
    :param start: YYYY-MM-DD string
    :param end: YYYY-MM-DD string
    :param interval: integer divisor of 60
    :param dirpath: string
    :return: pandas.Dataframe()
    """
    if 60 % interval != 0:
        raise ValueError(f'Interval of {interval} not a divisor of 60.')

    end = end if end else start
    dirpath = dirpath if dirpath[-1] == '/' else dirpath + '/'
    data = []
    for date in trading_daterange(start, end):
        filename = dirpath + 'xetra/' + f'{stock}-{date}.feather'  # We always assume that the file is in ./xetra
        if not ignore_missing_data:
            if not os.path.isfile(filename):
                # If data does not yet exist, we download it first.
                download(date, 'xetra', open('apikey', 'r').readline().strip(), dirpath, stock)
        if os.path.isfile(filename):
            df = pandas.read_feather(filename, columns=data_columns['xetra'])
            data.extend(df.values.tolist())
    buckets = OrderedDict()
    for row in data:
        date = row[col_index('Date')]
        hour, minutes = row[col_index('Time')].split(':')
        if int(hour) < 7:  # Price adjustements are recorded as pre-open trades. We omit them from analysis.
            continue
        price = row[col_index('EndPrice')]
        volume = row[col_index('NumberOfTrades')]
        minute_val = interval * (int(minutes) // interval)
        minute_val = str(minute_val) if minute_val >= 10 else '0' + str(minute_val)
        key = f'{date} {hour}:{minute_val}'
        # by using a dictionary we let the data structure do the bucketing for us.
        if key not in buckets:
            buckets[key] = [(volume, price)]
        else:
            buckets[key].append((volume, price))
    price_list = []
    price_items = list(buckets.items())
    yesterday = price_items[0][0].split(' ')[0]  # first date in the sequence.
    for index in range(0, len(buckets)):
        key, prices = price_items[index]
        date, time = key.split(' ')
        if date != yesterday:  # We omit prices from inter-day trading.
            yesterday = date
            continue
        avg_price = get_weighted_avg(prices)
        price_list.append([stock, date, time, avg_price])
    price_df = pandas.DataFrame(price_list, columns=['Mnemonic', 'Date', 'Time', 'Price'])
    return price_df


def get_weighted_avg(tuples_list):
    """
    Creates a weighted average from a list of weight, value tuples.
    :param tuples_list: [(volume, price) ...] typically will be used to make a weighted average.
    :return: a float.
    """
    size = sum([w[0] for w in tuples_list])
    avg = sum(map(lambda w: w[0]*w[1], tuples_list))/size
    return avg


def make_sma_df(data, window, column="Return"):
    """Given a return dataframe, returns the sma for each point in the dataframe.
    :param data: return dataframe
    :param window: int, sma's moving window
    :param column: Column to index, Default "Return", can change to "Price".
    :return:
    """
    if len(data) < window:
        raise ValueError(f'Data input shorter than window, ema cannot be computed.')
    # window-2: because we initialize sma with a value already.
    sma_df = data[['Mnemonic', 'Date', 'Time']].loc[window - 1:]
    # splicing does NOT work the same in pandas as in Python. Pandas includes spliced value when head splicing.
    window_list = list(data[column].loc[:window - 1])
    sma = [statistics.mean(window_list)]
    # Pandas excludes the spliced value whe tail splicing.
    returns = data[column].loc[window:]
    for value in returns:
        window_list = window_list[1:] + window_list[:1]
        window_list[-1] = value
        sma.append(statistics.mean(window_list))

    # resets index for dataframe so we can iterate from 0
    sma_df = sma_df.reset_index(drop=True)
    sma_df['SMA'] = sma
    return sma_df


def price_to_return(prices, difference=False):
    """
    Converts a price array to a corresponding return array. Useful for analysis that uses both values.
    :param prices: prices from prediction or y
    :param difference: Can calculate price difference instead of return if set True.
    :return: return nparray..
    """
    return_list = []
    for index in range(1, len(prices)):
        old_price = prices[index-1]
        price = prices[index]
        if difference:
            ret = price - float(old_price)
        else:
            ret = (price - float(old_price)) / float(old_price)
        return_list.append(ret)
    return np.array(return_list)


def price_to_return_df(price_df, difference=False):
    """
    Converts a price dataframe to a corresponding return dataframe. Useful for analysis that uses both values.
    :param price_df: price dataframe from make_price_df()
    :param difference: Can calculate price difference instead of return if set True.
    :return: return dataframe, analogous to make_return_df.
    """
    return_list = []
    price_items = price_df['Price'].tolist()
    metadata = price_df.get(['Mnemonic', 'Date', 'Time']).values.tolist()
    for index in range(1, len(price_items)):
        old_price = price_items[index-1]
        price = price_items[index]
        if difference:
            ret = price - float(old_price)
        else:
            ret = (price - float(old_price)) / float(old_price)
        metadata[index].append(ret)
        return_list.append(metadata[index])
    return_df = pandas.DataFrame(return_list, columns=['Mnemonic', 'Date', 'Time', 'Return'])
    return return_df


def make_return_df(stock, start, end=None, interval=30, dirpath='./', ignore_missing_data=False, difference=False):
    """Given a stock, a start date, end date optional, we return a returns dataframe with column headings
    ['Mnemonic', 'Date', 'Time', 'Return'].
    :param stock: Mnemonic string
    :param start: YYYY-MM-DD string
    :param end: YYYY-MM-DD string
    :param interval: integer divisor of 60
    :param dirpath: string
    :return: pandas.Dataframe()
    :param difference: bool
    """
    if 60 % interval != 0:
        raise ValueError(f'Interval of {interval} not a divisor of 60.')

    end = end if end else start
    dirpath = dirpath if dirpath[-1] == '/' else dirpath + '/'
    data = []
    for date in trading_daterange(start, end):
        filename = dirpath + 'xetra/' + f'{stock}-{date}.feather'  # We always assume that the file is in ./xetra
        if not ignore_missing_data:
            if not os.path.isfile(filename):
                # If data does not yet exist, we download it first.
                download(date, 'xetra', open('apikey', 'r').readline().strip(), dirpath, stock)
        if os.path.isfile(filename):
            df = pandas.read_feather(filename, columns=data_columns['xetra'])
            data.extend(df.values.tolist())
    buckets = OrderedDict()
    for row in data:
        date = row[col_index('Date')]
        hour, minutes = row[col_index('Time')].split(':')
        if int(hour) < 7:  # Price adjustements are recorded as pre-open trades. We omit them from analysis.
            continue
        price = row[col_index('EndPrice')]
        minute_val = interval * (int(minutes) // interval)
        minute_val = str(minute_val) if minute_val >= 10 else '0' + str(minute_val)
        key = f'{date} {hour}:{minute_val}'
        # by using a dictionary we let the data structure do the bucketing for us.
        if key not in buckets:
            buckets[key] = (int(minutes), price)
        else:
            # Here we are only keeping the first data point in the window, in order to calculate the return later.
            continue
    return_list = []
    price_items = list(buckets.items())
    yesterday = price_items[0][0].split(' ')[0]  # first date in the sequence.
    for index in range(1, len(buckets)):
        old_price = price_items[index-1][1]
        key, price = price_items[index]
        date, time = key.split(' ')
        if date != yesterday:  # We omit returns from inter-day trading.
            yesterday = date
            continue
        if difference:
            ret = price[1] - float(old_price[1])
        else:
            ret = (price[1] - float(old_price[1])) / float(old_price[1])
        return_list.append([stock, date, time, ret])
    return_df = pandas.DataFrame(return_list, columns=['Mnemonic', 'Date', 'Time', 'Return'])
    return return_df


def split_data(X, Y, test_ratio, purge_ratio=0.0):
    # Slice out evaluation set.
    assert purge_ratio + test_ratio < 1
    evaluation_set_count = int(len(X) * test_ratio)
    purge_set_count = int(len(X) * purge_ratio)
    evaluation_set_x = X[len(X) - evaluation_set_count:]
    x = X[:len(X) - evaluation_set_count - purge_set_count]
    evaluation_set_y = Y[len(X) - evaluation_set_count:]
    y = Y[:len(X) - evaluation_set_count - purge_set_count]
    return np.array(x), np.array(evaluation_set_x), np.array(y), np.array(evaluation_set_y)


def split_dataframe(df, test_ratio, purge_ratio=0.0):
    # Slice out evaluation set.
    assert purge_ratio + test_ratio < 1
    df_cols = df.columns
    df = [list(val[1]) for val in df.iterrows()]  # extract only rows of real data without column names.
    evaluation_set_count = int(len(df) * test_ratio)
    purge_set_count = int(len(df) * purge_ratio)
    evaluation_set = df[len(df) - evaluation_set_count:]
    df_purge = df[len(df) - evaluation_set_count - purge_set_count:len(df) - evaluation_set_count]
    df = df[:len(df) - evaluation_set_count - purge_set_count]
    if purge_ratio > 0:
        return pandas.DataFrame(df, columns=df_cols), pandas.DataFrame(evaluation_set, columns=df_cols), \
               pandas.DataFrame(df_purge, columns=df_cols)
    else:
        return pandas.DataFrame(df, columns=df_cols), pandas.DataFrame(evaluation_set, columns=df_cols)


def trading_daterange(start, end):
    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)
    number_of_days = int((end - start).days) + 1
    if number_of_days < 0:
        raise ValueError(f'start date {start.date()} must be before end date {end.date()}')
    for days in range(number_of_days):
        day = (start + timedelta(days))
        if day.weekday() < 5 and str(day.date()) not in holidays:
            yield day.date()
        else:
            continue


def seperate_index(index_df):
    """
    Breaks a DataFrame for an Index into several DataFrames one for the index and Each Stock."
    :param index_df: A DataFrame from the function make_index_price_df.
    :return: A list of DataFrames with columns Date, Time, Price, and Mnemonic.
    """
    expected_columns = ['Date', 'Time', 'Price_Index']
    validate_df(index_df, expected_columns)

    dfs = []

    prices = index_df[index_df.columns[2:]]
    price_columns = prices.columns

    for price_name in price_columns:
        df = index_df[['Date', 'Time']]
        df[price_name] = index_df[price_name]

        price_column_name = df.columns[2]
        df['Mnemonic'] = price_name.lstrip('Price_')
        df = df.rename({price_column_name: 'Price'}, axis=1)

        dfs.append(df)

    return dfs


def make_index_price_df(*dfs):
    """
    Given a list of stock DataFrames will return a new DataFrame representing an index composed of those stocks.
    The index price is the average price of the stocks.
    Mnemonic columns are removed and price labels are changed from 'Price' to 'Price_Mnemonic'. e.g. Price_BMW.
    Not all stock are traded every minute. Thus, nan values are first forward filled and then backward filled. This
    implies that if you want form an index you should have stock prices over the same range. Otherwise, the prices
    will be inaccurate after filling.
    :param dfs: A list of DataFrames representing a stock price with columns 'Date', 'Time', and 'Price'.
    :return: A DataFrame with Date, Time, Stock_price(s), and Index_price.
    """
    if len(dfs) == 0:
        raise ValueError('Must pass in one or more DataFrames.')

    for df in dfs:
        expected_columns = ['Date', 'Time', 'Price']
        validate_df(df, expected_columns)

    index_df = combine_stock_dfs(dfs)

    # Add index prices to DataFrame.
    price_columns = index_df.columns[2:]
    index_prices = index_df[price_columns].apply(lambda row: sum(row)/len(row), axis=1)
    index_df['Price_Index'] = index_prices

    return index_df


def combine_stock_dfs(dfs, type='Price'):
    """
    combine_stock_dfs if given a list of security Dataframes returns a larger DataFrame that merged dates.
    Note that this function fills in empty prices (the stock wasn't traded that minute) by first forward filling then
    back filling.
    :param dfs: A list of security DataFrames to combine.
    :return:
    """
    if type is not 'Price' and type is not 'Return':
        raise ValueError('type must be either "Price" or "Return".')

    for df in dfs:
        expected_columns = ['Date', 'Time', type]
        validate_df(df, expected_columns)

    combined_df = dfs[0]
    combined_df = combined_df.rename(columns={f'{type}': f"{type}_{combined_df['Mnemonic'][0]}"})
    combined_df = combined_df.drop('Mnemonic', axis=1)
    for df in dfs[1:]:
        current_df = df
        current_df = current_df.rename(columns={f'{type}': f"{type}_{current_df['Mnemonic'][0]}"})
        current_df = current_df.drop('Mnemonic', axis=1)
        combined_df = pandas.merge(combined_df, current_df, how='outer', on=['Date', 'Time'])
    combined_df = combined_df.sort_values(by=['Date', 'Time'])
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    return combined_df


def validate_df(df, expected_columns):
    """
    Given an object checks to make sure that it is a DataFrame, has elements, and has the expected columns.
    :param df:
    :param expected_columns:
    :return:
    """
    if type(df) is not pandas.DataFrame:
        raise ValueError(f'Index should be DataFrame not {type(df)}')
    for column in expected_columns:
        if column not in df.columns:
            raise ValueError(f'DataFrame should have column: {column}.')
    if len(df) == 0:
        raise ValueError(f'Length of DataFrame must be greater than 0.')

class StatisticalArbitrage:
    betas = namedtuple('Beta', 'beta_not, beta')

    def __init__(self, index):
        expected_columns = ['Date', 'Time', 'Return_Index']
        validate_df(index, expected_columns)

        self.stock_names = self._get_stock_names(index)
        self.data = index
        self.hyper_parameters = pandas.DataFrame(columns=self.stock_names)
        self._update_hyper_parameters()


    def _get_stock_names(self, index):
        return_names = list(index.columns)
        del return_names[return_names.index('Date')]
        del return_names[return_names.index('Time')]
        del return_names[return_names.index('Return_Index')]
        return return_names


    def _update_hyper_parameters(self):
        for stock in self.stock_names:
            observations = len(self.data)
            reg_1 = linear_model.LinearRegression(fit_intercept=True)
            dependent_var = np.array(self.data[stock]).reshape(-1, 1)
            independent_var = np.array(self.data['Return_Index']).reshape(-1, 1)
            reg_1.fit(independent_var, dependent_var)
            beta =  reg_1.coef_[0][0]
            self.hyper_parameters.loc['Beta', stock]  = beta
            beta_0 =  reg_1.intercept_[0]
            self.hyper_parameters.loc['Beta_0', stock] = beta_0
            self.hyper_parameters.loc['alpha', stock] = beta_0*observations # that is divide by delta_time

            residuals = dependent_var - beta_0 - beta*independent_var
            X = np.cumsum(residuals)
            X_prime = X.copy()
            X = X[:len(X) - 1]
            X_prime = X_prime[1:]
            reg_2 = linear_model.LinearRegression(fit_intercept=True)
            reg_2.fit(X.reshape(-1, 1), X_prime.reshape(-1, 1))
            b = reg_2.coef_[0][0]
            self.hyper_parameters.loc['b', stock]  = b
            a = reg_2.intercept_[0]
            self.hyper_parameters.loc['a', stock] = a

            zeta = X_prime - a - b * X
            kappa = -np.log(b)*observations
            self.hyper_parameters.loc['Kappa', stock] = kappa
            mean = a/(1 - b)
            self.hyper_parameters.loc['Mean', stock] = mean
            # sigma = np.sqrt(np.var(zeta)*2*kappa/(1 - b**2))
            sigma_eq = np.sqrt(np.var(zeta)/(1-b**2))
            self.hyper_parameters.loc['Sigma_eq', stock] = sigma_eq

    def predict(self, return_vector):
        self.data = self.data.append(return_vector)
        self._update_hyper_parameters()
        signal_vector = -self.hyper_parameters.loc['Mean']/self.hyper_parameters.loc['Sigma_eq']
        signal_vector = signal_vector.rename(lambda old_column_name: old_column_name.replace('Return', 'Signal'))
        return signal_vector















