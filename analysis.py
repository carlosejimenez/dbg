import itertools
    
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import time
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn import linear_model, metrics
from sklearn.linear_model import Ridge, Lasso


import tools
first_day = '2017-06-17'
yesterday = date.today() - timedelta(1)

percentage_to_evaluate = 0.05

my_df = tools.make_return_df('BMW', first_day, yesterday.strftime('%Y-%m-%d'), interval=60)

x, y = tools.build_x_y(my_df, 10, 0.2)


# Slice out evaluation set.
evaluation_set_count = int(len(my_df)*percentage_to_evaluate)
evaluation_set_x = x[len(my_df) - evaluation_set_count:]
x = x[:len(my_df) - evaluation_set_count]
evaluation_set_y = y[len(my_df) - evaluation_set_count:]
y = y[:len(my_df) - evaluation_set_count]

# kf = KFold(n_splits=3)
# for train_index, test_index in kf.split(returns):
#     print(train_index, ' ', test_index)
alpha_evaluation = []
alphas = np.arange(0, 1, 0.01)
for alpha in alphas:
    clf = Lasso(alpha=alpha)  # fixme want to test alpha
    clf.fit(x, y)
    score = clf.score(evaluation_set_x, evaluation_set_y)
    alpha_evaluation.append((alpha, score))

plot_x = [x[0] for x in alpha_evaluation]
plot_y = [y[1] for y in alpha_evaluation]
plt.plot(plot_x, plot_y)
plt.show()
plt.savefig('interval_60_alpha_test_without_lasso.png')




# regr = linear_model.LinearRegression()
# regr.fit(training_set_x, training_set_y)

exit(-1)


# predictions_y = regr.predict(test_set_x)
# print(regr.coef_)
# print(metrics.mean_squared_error(test_set_y, predictions_y))
# print(metrics.r2_score(test_set_y, predictions_y))







# GRAPH BMW RETURNS
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
#
# fig, ax = plt.subplots()
#
# dates = [datetime.datetime.fromisoformat(x).date() for x in my_df['Date']]
#
# ax.plot(dates, my_df['Return'])
# ax.xaxis.set_minor_locator(months)
# ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))
# ax.xaxis.set_major_locator(years)
# years_formator = mdates.DateFormatter('%Y')
# ax.xaxis.set_major_formatter(years_formator)
# ax.grid(True)

# ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

# plt.show()



# KFOLD TESTING
# kf = KFold(n_splits=3)
# for train_index, test_index in kf.split(returns):
#     print(train_index, ' ', test_index)

# ARMA TESTING

# model = sm.tsa.ARMA(my_df['Return'], (10, 5))
# res= model.fit()
#
#
# things = res.plot_predict(start=len(my_df)- 5, end=len(my_df) + 2)
# plt.show()

