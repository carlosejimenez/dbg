import numpy as np
import os
import pandas
import requests
import statistics
from threading import Lock

from collections import OrderedDict
from datetime import datetime, timedelta

dax = {'WDI': 'DE0007472060', 'DPW': 'DE0005552004', 'DBK': 'DE0005140008', 'RWE': 'DE0007037129',
       'VNA': 'DE000A1ML7J1', 'LHA': 'DE0008232125', 'DB1': 'DE0005810055', 'TKA': 'DE0007500001',
       'EOAN': 'DE000ENAG999', 'BAS': 'DE000BASF111', 'MUV2': 'DE0008430026', 'IFX': 'DE0006231004',
       'BEI': 'DE0005200000', '1COV': 'DE0006062144', 'CON': 'DE0005439004', 'SAP': 'DE0007164600',
       'HEI': 'DE0006047004', 'SIE': 'DE0007236101', 'HEN3': 'DE0006048432', 'ALV': 'DE0008404005',
       'VOW3': 'DE0007664039', 'BAYN': 'DE000BAY0017', 'ADS': 'DE000A1EWWW0', 'FRE': 'DE0005785604',
       'DAI': 'DE0007100000', 'FME': 'DE0005785802', 'DTE': 'DE0005557508', 'BMW': 'DE0005190003',
       'MRK': 'DE0006599905'}

urls = {'xetra': 'https://api.developer.deutsche-boerse.com/prod/xetra-public-data-set/1.0.0/xetra',
        'eurex': 'https://api.developer.deutsche-boerse.com/prod/eurex-public-data-set/1.0.0/eurex'}

data_columns = {'xetra': ['Mnemonic', 'Date', 'Time', 'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'TradedVolume',
                          'NumberOfTrades'],
                'eurex': ['Isin', 'SecurityType', 'MaturityDate', 'StrikePrice', 'PutOrCall', 'Date', 'Time',
                          'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'NumberOfContracts', 'NumberOfTrades']}

holidays = ['2018-03-30', '2018-04-02', '2018-05-01', '2018-05-21', '2018-10-03', '2018-12-25', '2018-12-26',
            '2019-01-01',

            # '2017-06-19', '2017-06-20', '2017-06-21', '2017-06-22', '2017-06-23', '2017-06-26', '2017-06-27',
            # '2017-06-28', '2017-06-29', '2017-06-30', '2017-10-03', '2017-10-23', '2017-10-24', '2017-10-30',
            # '2017-10-31', '2017-12-25', '2017-12-26', '2018-01-01', '2018-07-13'
            ]
            #todo remove these

lock = Lock()


def build_x_y(return_df, window, alpha):
    """
    Given a returns dataframe, we construct the EMA, SMA feature vectors returned as X, and the associated labels,
    offset 1, returned as Y.
    :param return_df: returns dataframe
    :param window: window for EMA, SMA
    :param alpha: alpha for EMA
    :return: X, Y tuple
    """
    assert type(return_df) == pandas.DataFrame

    ema_df = make_ema_df(return_df, window, alpha)
    sma_df = make_sma_df(return_df, window)

    return_df = return_df.loc[window:]
    ema_df = ema_df.loc[:len(ema_df) - 2]  # offset 1
    sma_df = sma_df.loc[:len(sma_df) - 2]  # offset 1

    assert len(return_df) == len(ema_df) == len(sma_df)
    return_df = return_df.reset_index(drop=True)

    Y = list(return_df['Return'])
    X = list(zip(ema_df['EMA'], sma_df['SMA']))

    return X, Y


def build_ar_x_y(array, p, column=None):
    """
    Makes an AR(p) process dataset, for a single array, or dataframe column if it is specified.
    Essentially, this produces a shifted lag dataset.
    :param array: Array, or dataframe on which to construct the dataset.
    :param p: AR(p), the number of values contained in the dependent vars vector.
    :param column: optional, use for dataset parameter; which column should we construct X, Y for?
    :return: two lists, x and y, where x is a list of vectors, and y is the associated labels.
    """
    if column:
        array = array[column].tolist()
    else:
        array = array.tolist()
    assert p < len(array)
    x = []
    y = []
    for i in range(p, len(array)):
        y.append(array[i])
        x.append(array[i-p:i])
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


def make_ema_df(data, window, alpha):
    """Given a return dataframe, returns the ema for each point in the dataframe based on parameters.

    :param data: a return dataframe
    :param window: int, ema's moving window
    :param alpha: weight in (0, 1)
    :return: ema dataframe with original dataframe - window points
    """
    if len(data) < window:
        raise ValueError(f'Data input shorter than window, ema cannot be computed.')
    if not 0 <= alpha <= 1:
        raise ValueError(f'alpha: {alpha} not in bounds of (0, 1)')
    # splicing does NOT work the same in pandas as in Python. Pandas include spliced value when head splicing.
    ema = [statistics.mean(data['Return'].loc[:window-1])]
    # window-2: because we already added a value to ema.
    ema_df = data[['Mnemonic', 'Date', 'Time']].loc[window-1:]
    # Pandas excludes the spliced value whe tail splicing.
    returns = data['Return'].loc[window:]
    for value in returns:
        ema_point = alpha * value + (1-alpha) * ema[-1]
        ema.append(ema_point)

    # resets index for dataframe so we can iterate from 0
    ema_df = ema_df.reset_index(drop=True)
    ema_df['EMA'] = ema
    return ema_df


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


def make_sma_df(data, window):
    """Given a return dataframe, returns the sma for each point in the dataframe.
    :param data: return dataframe
    :param window: int, sma's moving window
    :return:
    """
    if len(data) < window:
        raise ValueError(f'Data input shorter than window, ema cannot be computed.')
    # window-2: because we initialize sma with a value already.
    sma_df = data[['Mnemonic', 'Date', 'Time']].loc[window - 1:]
    # splicing does NOT work the same in pandas as in Python. Pandas includes spliced value when head splicing.
    window_list = list(data['Return'].loc[:window - 1])
    sma = [statistics.mean(window_list)]
    # Pandas excludes the spliced value whe tail splicing.
    returns = data['Return'].loc[window:]
    for value in returns:
        window_list = window_list[1:] + window_list[:1]
        window_list[-1] = value
        sma.append(statistics.mean(window_list))

    # resets index for dataframe so we can iterate from 0
    sma_df = sma_df.reset_index(drop=True)
    sma_df['SMA'] = sma
    return sma_df


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


def split_data(X, Y, percentage_to_evaluate):
    # Slice out evaluation set.
    evaluation_set_count = int(len(X) * percentage_to_evaluate)
    evaluation_set_x = X[len(X) - evaluation_set_count:]
    x = X[:len(X) - evaluation_set_count]
    evaluation_set_y = Y[len(X) - evaluation_set_count:]
    y = Y[:len(X) - evaluation_set_count]
    return np.array(x), np.array(evaluation_set_x), np.array(y), np.array(evaluation_set_y)


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

    # Form new DataFrame with the correct format.
    index_df = dfs[0]
    index_df = index_df.rename(columns={'Price': f"Price_{index_df['Mnemonic'][0]}"})
    index_df = index_df.drop('Mnemonic', axis=1)
    for df in dfs[1:]:
        current_df = df
        current_df = current_df.rename(columns={'Price': f"Price_{current_df['Mnemonic'][0]}"})
        current_df = current_df.drop('Mnemonic', axis=1)
        index_df =  pandas.merge(index_df, current_df, how='outer', on=['Date', 'Time'])
    index_df = index_df.sort_values(by=['Date', 'Time'])
    index_df = index_df.fillna(method='ffill').fillna(method='bfill')

    # Add index prices to DataFrame.
    price_columns = index_df.columns[2:]
    index_prices = index_df[price_columns].apply(lambda row: sum(row)/len(row), axis=1)
    index_df['Price_Index'] = index_prices

    return index_df

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

class StatisticalArbitrage():
    def __init__(self, index):
        expected_columns = ['Date', 'Time', 'Price_Index']
        validate_df(index, expected_columns)









