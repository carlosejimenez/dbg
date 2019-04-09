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
       'MRK': 'DE0006599905' }



urls = {'xetra': 'https://api.developer.deutsche-boerse.com/prod/xetra-public-data-set/1.0.0/xetra',
        'eurex': 'https://api.developer.deutsche-boerse.com/prod/eurex-public-data-set/1.0.0/eurex'}

data_columns = {'xetra': ['Mnemonic', 'Date', 'Time', 'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'TradedVolume',
                          'NumberOfTrades'],
                'eurex': ['Isin', 'SecurityType', 'MaturityDate', 'StrikePrice', 'PutOrCall', 'Date', 'Time',
                          'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'NumberOfContracts', 'NumberOfTrades']}

holidays = ['2018-03-30', '2018-04-02', '2018-05-01', '2018-05-21', '2018-10-03', '2018-12-25', '2018-12-26',
            '2019-01-01',

            '2017-06-19', '2017-06-20', '2017-06-21', '2017-06-22', '2017-06-23', '2017-06-26', '2017-06-27',
            '2017-06-28', '2017-06-29', '2017-06-30', '2017-10-03', '2017-10-23', '2017-10-24', '2017-10-30',
            '2017-10-31', '2017-12-25', '2017-12-26', '2018-01-01', '2018-07-13']
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
    ema_df = ema_df.loc[:len(ema_df) - 2]
    sma_df = sma_df.loc[:len(sma_df) - 2]

    assert len(return_df) == len(ema_df) == len(sma_df)
    return_df = return_df.reset_index(drop=True)

    Y = list(return_df['Return'])
    X = list(zip(ema_df['EMA'], sma_df['SMA']))

    return X, Y


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


def make_return_df(stock, start, end=None, interval=30, dirpath='./xetra/', difference=False):
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
    data = pandas.DataFrame(columns=data_columns['xetra'])

    for date in trading_daterange(start, end):
        filename = dirpath + f'{stock}-{date}.feather'
        if not os.path.isfile(filename):
            # If data does not yet exist, we download it first.
            download(date, 'xetra', open('apikey', 'r').readline().strip(), dirpath, stock)
        if os.path.isfile(filename):
            df = pandas.read_feather(filename, columns=data_columns['xetra'])
            data = data.append(df)

    buckets = OrderedDict()

    for index, row in data.iterrows():
        date = row['Date']
        hour, minutes = row['Time'].split(':')
        if int(hour) < 7:  # Price adjustements are recorded as pre-open trades. We omit them from analysis.
            continue
        price = row['EndPrice']
        minute_val = interval * (int(minutes) // interval)
        minute_val = str(minute_val) if minute_val > 10 else '0' + str(minute_val)
        key = f'{date} {hour}:{minute_val}'

        # by using a dictionary we let the data structure do the bucketing for us.
        if key not in buckets:
            buckets[key] = (int(minutes), price)
        else:
            # Here we are only keeping the first data point in the window, in order to calculate the return later.
            continue

    return_df = pandas.DataFrame(columns=['Mnemonic', 'Date', 'Time', 'Return'])

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
        return_df.loc[len(return_df)] = [stock, date, time, ret]

    return return_df


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


