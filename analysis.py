import itertools
    
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import time
import numpy as np
import math
from threading import Thread
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn import linear_model, metrics
from sklearn.linear_model import LassoCV, RidgeCV, Lasso, Ridge


import tools
first_day = '2017-06-17'
yesterday = date.today() - timedelta(1)

percentage_to_evaluate = 0.05

interval_length = 2
window_length = 10
ema_hyper_parameter = 0.2
k_fold_hyperparameter = 6


my_df = tools.make_return_df('BMW', first_day, yesterday.strftime('%Y-%m-%d'), interval=interval_length, difference=True)

x, y = tools.build_x_y(my_df, window_length, ema_hyper_parameter)

# Slice out evaluation set.
evaluation_set_count = int(len(my_df)*percentage_to_evaluate)
evaluation_set_x = x[len(my_df) - evaluation_set_count:]
x = x[:len(my_df) - evaluation_set_count]
evaluation_set_y = y[len(my_df) - evaluation_set_count:]
y = y[:len(my_df) - evaluation_set_count]


alphas_ridge = np.arange(0, 1, 0.01)
alphas_lasso = np.arange(0, 100, 1)

def run_regression_return_score(regression_function, X, Y, alphas, cv=None):
    """
    Runs regression technique on given X, Y with regression_function and list of alphas.
    :param regression_function: one of the following functions Lasso, Ridge, LassoCV, RidgeCV
    :param X: The independent variables.
    :param Y: The dependent variables.
    :param alphas: A list of alpha hyperparameters to test.
    :param cv: If passing LassoCV or RidgeCV a positive integer cv must be given. See
    sklearn documentation on LassoCV or RidgeCV.
    :return: (score, alpha) that has the highest score.
    """
    if len(alphas) == 0:
        raise ValueError(f'alphas must be given {alphas}')

    if (regression_function == LassoCV or regression_function == RidgeCV) and cv is None:
        raise ValueError('Must pass in cv when passing in LassCV or RidgeCV.')

    if cv is not None:
        clf = regression_function(alphas=alphas, cv=cv)
        clf.fit(X, Y)
        return clf.score(evaluation_set_x, evaluation_set_y), clf.alpha_
    else:
        best_alpha = alphas[0]
        best_score = -math.inf
        for alpha in alphas:
            clf = regression_function(alpha)
            clf.fit(X, Y)
            current_score = clf.score(evaluation_set_x, evaluation_set_y)
            # print(f'current score is: {current_score}')
            if current_score > best_score:
                # print('changed best score')
                best_score = current_score
                best_alpha = alpha
        return best_score, best_alpha


clf = RidgeCV(np.arange(0.01, 1, 0.01), cv=k_fold_hyperparameter, fit_intercept=False)
clf.fit(x, y)
print(f'coefficient is {clf.coef_}')

for tick in evaluation_set_x:



print(clf.score(evaluation_set_x, evaluation_set_y))
# clf = LassoCV(np.arange(0.01, 1, 0.01), 6)
# clf.fit(x, y)
# print(clf.score(evaluation_set_x, evaluation_set_y))

# print(f'Ridge CV, {run_regression_return_score(RidgeCV, x, y, alphas_ridge, 6)}')
# print(f'Ridge standard, {run_regression_return_score(Ridge, x, y, alphas_ridge)}')
# print(f'Lasso CV, {run_regression_return_score(LassoCV, x, y, alphas_lasso, 6)}')
# print(f'Lasso standard, {run_regression_return_score(Lasso, x, y, alphas_lasso)}')





# alpha_evaluation.append((alpha, score))

# plot_x = [x[0] for x in alpha_evaluation]
# plot_y = [y[1] for y in alpha_evaluation]
# plt.plot(plot_x, plot_y)
# plt.show()
# plt.savefig('interval_60_alpha_test_without_lasso.png')


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

# dates = [datetime.datetime.fromisoformat(x).date() for x in my_df['Date']]

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

