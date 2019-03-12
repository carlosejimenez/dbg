import sys
import re
import datetime
import requests
import pandas

dax = {'WDI': 'DE0007472060', 'DPW': 'DE0005552004', 'DBK': 'DE0005140008', 'RWE': 'DE0007037129',
       'VNA': 'DE000A1ML7J1', 'LHA': 'DE0008232125', 'DB1': 'DE0005810055', 'TKA': 'DE0007500001',
       'EOAN': 'DE000ENAG999', 'BAS': 'DE000BASF111', 'MUV2': 'DE0008430026', 'IFX': 'DE0006231004',
       'BEI': 'DE0005200000', '1COV': 'DE0006062144', 'CON': 'DE0005439004', 'SAP': 'DE0007164600',
       'HEI': 'DE0006047004', 'SIE': 'DE0007236101', 'HEN3': 'DE0006048432', 'ALV': 'DE0008404005',
       'VOW3': 'DE0007664039', 'BAYN': 'DE000BAY0017', 'ADS': 'DE000A1EWWW0', 'FRE': 'DE0005785604',
       'DAI': 'DE0007100000', 'FME': 'DE0005785802', 'DTE': 'DE0005557508', 'BMW': 'DE0005190003',
       'MRK': 'DE0006599905', 'LIN': 'IE00BZ12WP82'}



def download(date, api_key = 'e6e8d13f-2e66-476d-b375-c55b33eb7f8a'):
    """download feather archive files for all DAX stocks from Xetra for a particular date (YYYY-MM-DD).
    downloaded data schema is 'Mnemonic', 'Date', 'Time', 'StartPrice', 'MaxPrice',
     'MinPrice', 'EndPrice', 'TradedVolume', 'NumberOfTrades'.

    stdout is stocks that failed to write, most likely from an api error.
    """

    columns = ['Mnemonic', 'Date', 'Time', 'StartPrice', 'MaxPrice', 'MinPrice', 'EndPrice', 'TradedVolume',
               'NumberOfTrades']

    for key in dax:
        headers = {
            'X-DBP-APIKEY': api_key,
        }

        params = (
            ('date', f'{date}'), ('limit', 1000), ('isin', f'{dax[key]}')
        )

        url = 'https://api.developer.deutsche-boerse.com/prod/xetra-public-data-set/1.0.0/xetra'

        try:
            response = requests.get(url, headers=headers, params=params)

            response = response.json()
            response_trimmed = []

            for minute in response:
                entry = {i: minute[i] for i in columns}
                response_trimmed.append(entry)

            df = pandas.DataFrame(response_trimmed, columns=columns)

            df.to_feather(f'{key}-{date}.feather')

        except:
            print(f'{key} failed to write for date {date}.')


if __name__ == '__main__':

    # provide date to download all data for each company by command line argument YYYY-MM-DD
    date = sys.argv[1]

    download(date)
