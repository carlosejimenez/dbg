import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta, datetime
import pandas


def graph_prices(y_pred, y_test, title, dates, ticks='Year'):
    years = mdates.YearLocator()
    months = mdates.MonthLocator()

    yt = list(y_test)
    yp = [None]*(len(y_test) - len(y_pred)) + list(y_pred)
    fig, ax = plt.subplots()
    ax.plot_date(x=dates[:len(yt)], y=yt, fmt="r-", label=f'True y')
    ax.plot_date(x=dates[:len(yp)], y=yp, fmt="b-", label=f'Predicted y')
    if ticks=='Year':
        ax.xaxis.set_major_locator(years)
    else:
        ax.xaxis.set_major_locator(months)
    if ticks=='Year':
        formatter = mdates.DateFormatter('%Y')
    else:
        formatter = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(formatter)
    # plt.plot(x=dates, y=yt, color='navy', label=f'True y')
    # plt.plot(x=dates, y=yp[:-1], color='red', label=f'Predicted y')
    ax.set_title(title)
    # ax.xlabel = 'Time'
    # ax.ylabel = f'y'
    ax.legend()
    # plt.show()
    plt.savefig(title)
    plt.clf()


def graph_returns(return_data_frame, title):
    print(return_data_frame.columns)
    if 'Date' not in return_data_frame.columns:
        raise ValueError(f'dataframe should have column: Data')
    if 'Return' not in return_data_frame.columns:
        raise ValueError(f'dataframe should have column: Return')

    years = mdates.YearLocator()
    months = mdates.MonthLocator()

    fig, ax = plt.subplots()

    if type(return_data_frame['Date'][0]) is pandas.Timestamp:
        dates = [x.to_pydatetime().date() for x in return_data_frame['Date']]
    else:
        dates = [datetime.fromisoformat(x).date() for x in return_data_frame['Date']]

    ax.plot(dates, return_data_frame['Return'])
    ax.set_title(title)
    # ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))
    ax.xaxis.set_major_locator(months)
    years_formator = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(years_formator)
    ax.grid(True)

    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

    plt.show()

def return_array_to_df(returns):
    x_values = [x[0] for x in returns]
    y_values = [x[1] for x in returns]
    plt.plot(x_values, y_values)
    return_df = pandas.DataFrame(x_values, columns=['Date'])
    return_df['Return'] = y_values
    return return_df
