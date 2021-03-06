import numpy as np
import matplotlib.pylab as plt
from matplotlib.dates import YearLocator, DateFormatter
from datetime import datetime
import tensorflow.python.keras.backend as K
from math import log
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

import tools


class Lstm:
    def __init__(self, time_steps, input_size, output_size):
        """
        sets up local fields and objects.
        :param time_steps: time step p for AR process p
        :param input_size: number of stocks we are trying to predict. Ignore p. just stocks number
        :param output_size: same as input_size??. Potentially redundant field.
        """
        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.x_train = None
        self.y_train = None
        self.y_purge = None

        self.scale_y = MinMaxScaler(feature_range=(-1, 1))
        self.scale_x = MinMaxScaler(feature_range=(-1, 1))
        self.model = Sequential()

    def load_and_split_data(self, x, y, test_ratio, purge_ratio, units, epochs, batch_size):
        self.x_train, x_test, self.y_train, y_test = tools.split_data(x, y, test_ratio=test_ratio,
                                                                      purge_ratio=purge_ratio)
        self.y_purge = y[len(self.y_train):len(y) - len(y_test)]
        self.y_train = self.scale_y.fit_transform(np.array(self.y_train).reshape(-1, self.output_size))
        self.x_train = self.scale_x.fit_transform(self.x_train)
        # x_test = self.scale_x.transform(x_test)
        self.x_train = np.reshape(self.x_train, (-1, self.time_steps, self.input_size))
        # x_test = np.reshape(x_test, (-1, self.time_steps, self.input_size))

        self.model.add(LSTM(units=units, return_sequences=True, input_shape=(self.time_steps, self.input_size)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=units, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=units, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=units))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=self.output_size))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        hist = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

        return x_test, y_test


    def load_data(self, x, y, units, epochs, batch_size):
        self.x_train = np.array(x)
        self.y_train = np.array(y)
        # print(f'X_train.shape = {self.x_train.shape}')
        self.y_train = self.scale_y.fit_transform(np.array(y).reshape(-1, self.output_size))
        self.x_train = self.scale_x.fit_transform(self.x_train)
        self.x_train = np.reshape(self.x_train, (-1, self.time_steps, self.input_size))

        self.model.add(LSTM(units=units, return_sequences=True, input_shape=(self.time_steps, self.input_size)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=units, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=units, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=units))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=self.output_size))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x_test, y_test=None, graph=False, title=None, dates=None):
        x_test = np.array(x_test).reshape(-1, self.time_steps*self.output_size)
        # print(f'x_test.shape = {x_test.shape}')
        x_test = self.scale_x.transform(np.array(x_test))
        x_test = np.array(x_test).reshape((-1, self.time_steps, self.input_size))
        y_pred = self.model.predict(x_test)
        y_pred = self.scale_y.inverse_transform(y_pred)
        # for val in x_test:
        #     val_pred = self.model.predict(val.reshape((1, val.shape[0], val.shape[1])))
        #     val_pred = self.scaler.inverse_transform(val_pred)
        #     y_pred.append(val_pred[0][0])
        assert len(y_pred) == len(x_test), f'Prediction malformed. x_test dim {x_test.shape[0]} != ({len(y_pred)},)'
        if graph:

            assert y_test is not None, f'y_test parameter must be initialized to graph.'
            assert title is not None, f'title parameter must be initialized to graph.'
            # y = y_train + y_purge + y_test (clearly this is the case)
            # self.scaler must use inverse_transform to convert the training set back to its real values.
            y = list(self.scale_y.inverse_transform(self.y_train)) + list(self.y_purge) + list(y_test)
            y_p = [None]*len(list(self.y_train.flatten()) + list(self.y_purge))  # y_p to graph y_prediction
            y_p.extend(y_pred)
            self.y_purge = [None] * len(self.y_train) + self.y_purge + [None]*len(y_test)  # y_purge to graph y_purge
            fig, ax = plt.subplots()
            ax.plot_date(dates[:len(y)], y[:len(dates)], fmt="r-", label=f'True y')
            ax.plot_date(dates[:len(y_p)], y_p[:len(dates)], fmt='b-', label=f'Predicted y')
            ax.plot_date(dates[:len(self.y_purge)], self.y_purge[:len(dates)], fmt='g-', label=f'Purged values')
            ax.set_title(f'{title}')
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            # ax.xlabel('Time')
            # ax.ylabel(f'y')
            ax.legend()
            plt.savefig(title)
            plt.clf()

            return np.array(y_pred)
        else:
            return np.array(y_pred)
