import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta, datetime
import pandas

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
