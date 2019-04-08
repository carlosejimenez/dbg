import tools
from matplotlib import pyplot as plt
import datetime
import pandas
# import statsmodels.api as sm

# df = pandas.read_feather('/Users/carlosejimenez/Documents/School/cs5140/dbg/xetra/BMW-2017-08-08.feather')
returns = tools.make_return_df(stock='BMW', start='2017-08-08', end='2017-08-09')

# my_df = tools.make_return_df('BMW', '2018-12-30', '2019-03-18')
#
# returns = my_df['Return']
# dates = [datetime.datetime.fromisoformat(x).date for x in my_df['Date']]
#
# plt.scatter(x=range(len(dates)), y=returns)
# plt.show()

# print(returns.head(20))
# print(len(returns))
# data = sm.tsa.ARMA(returns, (3, 3)).fit()
