from tools import make_return_df

if __name__ == '__main__':

    returns = make_return_df(stock='SAP', start='2018-06-28', interval=14, end='2018-06-29', filepath='./data')
    print(returns.head())
