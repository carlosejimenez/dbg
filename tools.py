import os

import pandas
import requests

from collections import OrderedDict
from datetime import datetime, timedelta

dax = {'WDI': 'DE0007472060', 'DPW': 'DE0005552004', 'DBK': 'DE0005140008', 'RWE': 'DE0007037129',
       'VNA': 'DE000A1ML7J1', 'LHA': 'DE0008232125', 'DB1': 'DE0005810055', 'TKA': 'DE0007500001',
       'EOAN': 'DE000ENAG999', 'BAS': 'DE000BASF111', 'MUV2': 'DE0008430026', 'IFX': 'DE0006231004',
       'BEI': 'DE0005200000', '1COV': 'DE0006062144', 'CON': 'DE0005439004', 'SAP': 'DE0007164600',
       'HEI': 'DE0006047004', 'SIE': 'DE0007236101', 'HEN3': 'DE0006048432', 'ALV': 'DE0008404005',
       'VOW3': 'DE0007664039', 'BAYN': 'DE000BAY0017', 'ADS': 'DE000A1EWWW0', 'FRE': 'DE0005785604',
       'DAI': 'DE0007100000', 'FME': 'DE0005785802', 'DTE': 'DE0005557508', 'BMW': 'DE0005190003',
       'MRK': 'DE0006599905', 'LIN': 'IE00BZ12WP82'}

urls = {'xetra': 'https://api.developer.deutsche-boerse.com/prod/xetra-public-data-set/1.0.0/xetra',
        'eurex': 'https://api.developer.deutsche-boerse.com/prod/eurex-public-data-set/1.0.0/eurex'}

data_columns = {'xetra': ['Mnemonic', 'Date', 'Time', 'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'TradedVolume',
                          'NumberOfTrades'],
                'eurex': ['Isin', 'SecurityType', 'MaturityDate', 'StrikePrice', 'PutOrCall', 'Date', 'Time',
                          'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'NumberOfContracts', 'NumberOfTrades']}


def download(date, api, api_key, filepath, stock_query=None):
    """download feather archive files for all DAX stocks from Xetra for a particular date (YYYY-MM-DD).
    downloaded data schema is 'Mnemonic', 'Date', 'Time', 'StartPrice', 'MaxPrice',
     'MinPrice', 'EndPrice', 'TradedVolume', 'NumberOfTrades'.

     :param date: Date as YYYY-MM-DD
     :param api: api to query
     :param api_key: key to use for api
     :param filepath: Path to save price data.
     :param stock_query: Optional, can be str, e.g. 'BMW'; or a list of of mnemonics, e.g. ['BMW', 'SAP'], default is
     the DAX index.

    stdout is stocks that failed to write, most likely from an api error.
    """
    filepath = './' + filepath + api + '/'
    os.makedirs(filepath, exist_ok=True)
    url = urls[api]
    columns = data_columns[api]

    if stock_query:
        if type(stock_query) is str:
            stock_query = [stock_query]
    else:
        stock_query = dax

    for stock_name in stock_query:
        filename = filepath + f'{stock_name}-{date}.feather'
        if os.path.isfile(filename):
            continue

        headers = {
            'X-DBP-APIKEY': api_key,
        }

        params = (
            ('date', f'{date}'), ('limit', 1000), ('isin', f'{dax[stock_name]}')
        )

        try:
            response = requests.get(url, headers=headers, params=params)
            print(f'\rSaving file {stock_name} for {date}', end='')
            response_trimmed = [{i: minute[i] for i in columns} for minute in response.json()]
            df = pandas.DataFrame(response_trimmed, columns=columns)
            df.to_feather(filename)

        except:
            print(f'\r{api}-{stock_name} failed to write for date {date}.')


def make_return_df(stock, start, end=None, interval=30, filepath='./'):
    """
    Given a stock, a start date, end date optional, we return a returns dataframe with column headings
    ['Mnemonic', 'Date', 'Time', 'Return'].
    :param stock: Mnemonic string
    :param start: YYYY-MM-DD string
    :param end: YYYY-MM-DD string
    :param interval: integer divisor of 60
    :param filepath: string
    :return: pandas.Dataframe()
    """
    if 60 % interval != 0:
        raise ValueError(f'Interval of {interval} not a divisor of 60.')

    end = end if end else start
    filepath = filepath if filepath[-1] == '/' else filepath + '/'
    data = pandas.DataFrame(columns=data_columns['xetra'])

    for date in trading_daterange(start, end):
        filename = filepath + f'xetra/{stock}-{date}.feather'
        if not os.path.isfile(filename):
            # If data does not yet exist, we download it first.
            download(date, 'xetra', open('apikey', 'r').readline().strip(), filepath, stock)
        if os.path.isfile(filename):
            df = pandas.read_feather(filename, columns=data_columns['xetra'])
            data = data.append(df)

    buckets = OrderedDict()

    for index, row in data.iterrows():
        date = row['Date']
        hour, minutes = row['Time'].split(':')
        price = row['EndPrice']
        minute_val = interval * (int(minutes) // interval)
        minute_val = str(minute_val) if minute_val > 10 else '0' + str(minute_val)
        key = f'{date} {hour}:{minute_val}'

        # by using a dictionary we let the data structure do the bucketing for us.
        if key not in buckets:
            buckets[key] = (int(minutes), price)
        else:
            if int(minutes) > buckets[key][0]:
                buckets[key] = (int(minutes), price)

    return_df = pandas.DataFrame(columns=['Mnemonic', 'Date', 'Time', 'Return'])

    price_items = list(buckets.items())

    for index in range(1, len(buckets)):
        old_price = price_items[index-1][1]
        key, price = price_items[index]
        date, time = key.split(' ')
        ret = (price[1] - float(old_price[1])) / float(old_price[1])
        return_df.loc[len(return_df)] = [stock, date, time, ret]

    return return_df


holidays = ['2018-03-30', '2018-04-02', '2018-05-01', '2018-05-21', '2018-10-03', '2018-12-25', '2018-12-26',
            '2019-01-01']


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


