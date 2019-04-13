import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import numpy as np
import datetime
import math
import statsmodels.api as sm
from sklearn import linear_model, metrics
from sklearn.linear_model import LassoCV, RidgeCV, Lasso, Ridge

import tools


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

def graph_returns(return_data_frame):
    print(return_data_frame.columns)
    if 'Date' not in return_data_frame.columns:
        raise ValueError(f'dataframe should have column: Data')
    if 'Return' not in return_data_frame.columns:
        raise ValueError(f'dataframe should have column: Return')

    years = mdates.YearLocator()
    months = mdates.MonthLocator()

    fig, ax = plt.subplots()

    dates = [datetime.datetime.fromisoformat(x).date() for x in return_data_frame['Date']]

    ax.scatter(dates, return_data_frame['Return'], marker='.')
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))
    ax.xaxis.set_major_locator(years)
    years_formator = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(years_formator)
    ax.grid(True)

    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

    plt.show()



if __name__ == '__main__':
    first_day = '2018-06-17'
    yesterday = date.today() - timedelta(1)

    percentage_to_evaluate = 0.05

    interval_length = 1
    window_length = 10
    ema_hyper_parameter = 0.2
    k_fold_hyperparameter = 6

    my_df = tools.make_return_df('BMW', first_day, yesterday.strftime('%Y-%m-%d'), interval=interval_length)
    graph_returns(my_df)

    x, y = tools.build_x_y(my_df, window_length, ema_hyper_parameter)

    x, evaluation_set_x, y, evaluation_set_y = tools.split_data(x, y, percentage_to_evaluate)

    alphas_ridge = np.arange(0, 1, 0.01)
    alphas_lasso = np.arange(0, 100, 1)

    clf = RidgeCV(np.arange(0.01, 1, 0.01), cv=k_fold_hyperparameter, fit_intercept=False)
    clf.fit(x, y)
    print(f'coefficient is {clf.coef_}')


def perform_arma():
    # ARMA TESTING
    model = sm.tsa.ARMA(my_df['Return'], (10, 5))
    res= model.fit()

    things = res.plot_predict(start=len(my_df)- 5, end=len(my_df) + 2)
    plt.show()

