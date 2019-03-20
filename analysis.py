import tools
import statsmodels.api as sm

my_df = tools.make_return_df('BMW', '2017-06-17', '2019-03-18')

returns = my_df['Return']
# print(returns.head(20))
# print(len(returns))
# data = sm.tsa.ARMA(returns, (3, 3)).fit()
