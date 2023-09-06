import os.path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_merged_data(only_standard_refuelling=True, drop_na=True):
    base_path = os.path.abspath("")
    data_directory = os.path.join(base_path, "data")
    # read as pickle
    df = pd.read_pickle(os.path.join(data_directory, 'merged.pkl'))

    if only_standard_refuelling:
        df = df[df['refuellingChargingDegreeActual'] >= 14.5]

    if drop_na:
        df = df.dropna()

    return df


def split_xy(df, input_columns, sample_size=None):
    if sample_size is not None:
        df = df.sample(sample_size)

    x = df[input_columns]
    y = df['refuellingTimePointActual']
    return x, y


class FurnaceModelTrainer:
    def __init__(self, df, input_columns, slice_step=1, cross_validation_splitter=TimeSeriesSplit(n_splits=5),
                 cross_validation_params=None, skip_first_cv_splits=0, model_builder=lambda: LinearRegression(),
                 fit_params=None, generate_eval_set=False):
        self.df = df
        self.slice_step = slice_step
        self.input_columns = input_columns
        self.cross_validation_splitter = cross_validation_splitter
        self.cross_validation_params = cross_validation_params
        if self.cross_validation_params is None:
            self.cross_validation_params = {}
        self.skip_first_cv_splits = skip_first_cv_splits
        self.model_builder = model_builder
        self.fit_params = fit_params
        if self.fit_params is None:
            self.fit_params = {}
        self.generate_eval_set = generate_eval_set

        self.X, self.y = split_xy(self.df, self.input_columns)
        self.model = None
        self.cross_validation_models = None

    def cross_validation(self):
        mape_scores = []
        mse_scores = []
        self.cross_validation_models = []

        splits = self.cross_validation_splitter.split(self.X, **self.cross_validation_params)

        for i in range(self.skip_first_cv_splits):
            next(splits)

        for train_index, test_index in splits:
            train_index = train_index[::self.slice_step]
            test_index = test_index[::self.slice_step]
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            data_train, data_test = self.df.iloc[train_index], self.df.iloc[test_index]

            model = self.model_builder()

            if self.generate_eval_set:
                X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
                model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], **self.fit_params)
            else:
                model.fit(X_train, y_train, **self.fit_params)

            y_pred = model.predict(X_test)

            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            mape_scores.append(mape)
            mse_scores.append(mse)
            self.cross_validation_models.append((model, X_train, X_test, y_train, y_test, data_train, data_test))

            print(f"Statistics for this fold: {mape}, {mse}, {np.sqrt(mse)}")

        avg_mape = np.mean(mape_scores)
        avg_mse = np.mean(mse_scores)

        print(f"Average MAPE: {avg_mape}")
        print(f"Average MSE: {avg_mse}")
        print(f"Average RSME: {np.sqrt(avg_mse)}")

        return avg_mape

    def train(self):
        self.model = self.model_builder()
        X = self.X[::self.slice_step]
        y = self.y[::self.slice_step]
        self.model.fit(X, y, **self.fit_params)

        return self.model

    def split_data(self, n=-1):
        model, X_train, X_test, y_train, y_test, data_train, data_test = self.cross_validation_models[n]

        y_pred = model.predict(X_test)

        return model, X_train, X_test, y_train, y_test, data_train, data_test, y_pred

    def visualize_split_scatter(self, n=-1):
        model, X_train, X_test, y_train, y_test, data_train, data_test, y_pred = self.split_data(n)
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([0, 100], [0, 100], color='red', lw=3)

    def visualize_split_example(self, n=-1, days=5):
        model, X_train, X_test, y_train, y_test, data_train, data_test, y_pred = self.split_data(n)

        start_date = data_test['date'].iloc[0]
        end_date = start_date + pd.Timedelta(days=days)

        # data for plot
        data_display = data_test.loc[(data_test['date'] >= start_date) & (data_test['date'] <= end_date)]
        y_pred_display = y_pred[:len(data_display)]

        sns.lineplot(x=data_display['date'], y=data_display['refuellingTimePointActual'], label='actual')
        sns.scatterplot(x=data_display['date'], y=y_pred_display, label='predicted', color='red')
        sns.scatterplot(x=data_display['date'], y=data_display['refuellingTimePoint'], label='Ofen Vorhersage',
                        color='blue')

        # set figure size
        plt.gcf().set_size_inches(16, 8)

    def visualize_split(self, n=-1, days=5):
        plt.subplot(2, 1, 1)
        self.visualize_split_scatter(n)
        plt.subplot(2, 1, 2)
        self.visualize_split_example(n, days)