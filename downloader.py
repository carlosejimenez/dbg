import argparse
from tools import download, trading_daterange

if __name__ == '__main__':
    api_key = open('apikey', 'r').readline().strip()

    parser = argparse.ArgumentParser()
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--filepath')
    args = parser.parse_args()

    # provide date to download all data for each company by command line argument YYYY-MM-DD
    start_date = args.start
    if start_date is None:
        raise ValueError('--start must be provided.')

    end_date = args.end if args.end else start_date
    filepath = args.filepath if args.filepath else './'
    filepath = filepath+'/' if not filepath[-1] == '/' else filepath

    for day in trading_daterange(start_date, end_date):
        download(date=day, api='xetra', api_key=api_key, filepath=filepath)
        # download(date=day, api='eurex', api_key=api_key, filepath=filepath)
