import os
import datetime

import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns


"""
This WindowGenerator class is based on the one with the same name from the Tensorflow time series examples, which can be found in: https://www.tensorflow.org/tutorials/structured_data/time_series. However, this has been extensively modified to suit application specific needs.
"""
class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_X, val_X, test_X, test_dates, label_columns = None):
        #Store the raw data
        self.train_X = train_X
        self.val_X = val_X
        self.test_X = test_X

        self.test_dates = test_dates

        # work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_X.columns)}

        # work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width+shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis = -1)

        # slicing does not preserve static shape info. set the shapes manually.
        # Then tf.data.Datasets become easier to inspect
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='cases', max_subplots=3):
#         inputs, labels = self.example
        inputs, labels = next(iter(self.test))
        plt.figure(figsize = (20,10))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()
        plt.xlabel('Time [weeks]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length = self.total_window_size,
            sequence_stride = 1,
            shuffle = True,
            batch_size = 32,
        )
        ds = ds.map(self.split_window)
        return ds

    def split_time(self, time_data):
        input_times = time_data[:,self.input_slice]
        label_times = time_data[:,self.labels_slice]
        return input_times, label_times

    def make_predictions(self, model):
        # From test_X, test_dates
        sample_size = self.test_X.shape[0]

        n_batches = int(sample_size/self.label_width)

        data_array = []
        time_array = []

        for n in np.arange(0,n_batches):
            i1 = n*self.label_width
            i2 = n*self.label_width+self.total_window_size
            if i2 >= sample_size:
                break

            time_array.append(np.arange(i1,i2))
            data_array.append(np.array(self.test_X[i1:i2]))

        data_windows = tf.stack(data_array)
        time_windows = tf.stack(time_array)

        inputs, labels = self.split_window(data_windows)
        input_times, label_times = self.split_time(time_windows)

        label_times = tf.reshape(label_times, [-1]).numpy()

        time_axis = self.test_dates.values[label_times]
        label_array = tf.reshape(labels, [-1]).numpy()

        predictions = model(inputs)

        prediction_array = tf.reshape(predictions, [-1]).numpy()

        return time_axis, label_array, prediction_array

    def make_shifted_predictions(self, model, time_shift):
        # From test_X, test_dates
        sample_size = self.test_X.shape[0]

        n_batches = int(sample_size/self.label_width)

        data_array = []
        time_array = []

        for n in np.arange(0,n_batches):
            i1 = n*self.label_width + time_shift
            i2 = n*self.label_width+self.total_window_size + time_shift
            if i2 >= sample_size:
                break

            time_array.append(np.arange(i1,i2))
            data_array.append(np.array(self.test_X[i1:i2]))

        data_windows = tf.stack(data_array)
        time_windows = tf.stack(time_array)

        inputs, labels = self.split_window(data_windows)
        input_times, label_times = self.split_time(time_windows)

        label_times = tf.reshape(label_times, [-1]).numpy()

        time_axis = self.test_dates.values[label_times]
        label_array = tf.reshape(labels, [-1]).numpy()

        predictions = model(inputs)

        prediction_array = tf.reshape(predictions, [-1]).numpy()

        return time_axis, label_array, prediction_array

    def combine_shifted_predictions(self,model):
        time_axis, label_array, prediction_array = self.make_shifted_predictions(model,0)
        pred_df = pd.DataFrame()
        pred_df['date'] = time_axis
        pred_df['pred'] = prediction_array
        pred_df['act'] = label_array
        pred_df.set_index('date',inplace=True)

        for time_shift in np.arange(1,self.label_width):
            time_axis, label_array, prediction_array = self.make_shifted_predictions(model,time_shift)
            new_df = pd.DataFrame()
            new_df['date'] = time_axis
            new_df['pred'] = prediction_array
            new_df['act'] = label_array
            new_df.set_index('date',inplace=True)
            pred_df = pd.concat((pred_df,new_df))

        pred_df = pred_df.groupby(pred_df.index).mean()
        time_axis = pred_df.index
        prediction_array = pred_df['pred'].values
        label_array = pred_df['act'].values

        return time_axis, label_array, prediction_array


    @property
    def train(self):
        return self.make_dataset(self.train_X)

    @property
    def val(self):
        return self.make_dataset(self.val_X)

    @property
    def test(self):
        return self.make_dataset(self.test_X)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the .train dataset
            result = next(iter(self.test))
            # and cache it for next time
            self._example = result
        return result

def prepare_data(ic_id, wr_df, inc_df, pics, split_frac, violin = False):

    X = pd.DataFrame()
    X['date'] = inc_df['date']
    X = pd.merge(X, inc_df[[str(ic_id), 'date']],how='outer', on='date')
    X.rename({str(ic_id): 'cases'}, axis='columns', inplace=True)
    X = pd.merge(X, inc_df[pics.append(pd.Index(['date']))], how='outer', on='date')
    X = pd.merge(X, wr_df, how='outer', on='date')

    X = X.loc[:, (X != 0).any(axis=0)]


    timeAxis = X.pop('date')

    n = len(X)
    train_X = X[0:int(n*split_frac[0])]
    val_X = X[int(n*split_frac[0]):int(n*(split_frac[0]+split_frac[1]))]
    test_X = X[int(n*(split_frac[0]+split_frac[1])):]
    test_dates = timeAxis[int(n*(split_frac[0]+split_frac[1])):]

#     print(f'Data points: {n}')
    print(f'Shape of dataset: {X.shape}')
#     print(f'Shape of train dataset: {train_X.shape}')
#     print(f'Shape of validation dataset: {val_X.shape}')
#     print(f'Shape of test dataset: {test_X.shape}')

    train_mean = train_X.mean()
    train_std = train_X.std()

    train_X = (train_X - train_mean) / train_std
    val_X = (val_X - train_mean) / train_std
    test_X = (test_X - train_mean) / train_std

    if violin:
        X_std = (X - train_mean)/train_std
        X_std = X_std.melt(var_name = 'Column', value_name = 'Normalized')
        plt.figure(figsize=(18,10))
        ax = sns.violinplot(x='Column', y='Normalized', data=X_std)
        _ = ax.set_xticklabels(X.keys(), rotation = 90)

    return train_X, val_X, test_X, test_dates, train_mean, train_std, X.shape[1]

def initialize_models(n_features, out_steps):
    linear_model = tf.keras.Sequential([
        # Take the last time-step
        # shape [batch, time, features] => [batch, 1, featuress]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(out_steps*n_features, kernel_initializer = tf.initializers.zeros),
        # shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, n_features])
    ])

    lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.  # single output: True, Multi-output: False
        tf.keras.layers.LSTM(32, return_sequences=False),   # True: Training on multiple timesteps. False: Final timestep
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(out_steps*n_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, n_features])
    ])

    gru_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        tf.keras.layers.GRU(32, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(out_steps*n_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, n_features])
    ])

    return linear_model, lstm_model, gru_model

def compile_and_fit(model, window, verbosity, patience=10, num_epochs = 20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])

    history = model.fit(window.train, epochs = num_epochs, verbose=verbosity, validation_data = window.val, callbacks = [early_stopping])

    return history

def fit_and_predict(linear_model, lstm_model, gru_model, window_var, iterations, verbosity):
    test_per = {}
    # Fit and predict linear model
    history = compile_and_fit(linear_model, window_var, verbosity, num_epochs = iterations)

    test_per['Linear'] = linear_model.evaluate(window_var.test, verbose=verbosity)

    # Fit and predict lstm model
    history = compile_and_fit(lstm_model, window_var, verbosity, num_epochs = iterations)

    test_per['LSTM'] = lstm_model.evaluate(window_var.test, verbose=verbosity)

    # Fit and predict gru model
    history = compile_and_fit(gru_model, window_var, verbosity, num_epochs = iterations)

    test_per['GRU'] = gru_model.evaluate(window_var.test, verbose=verbosity)

    return test_per

def run_ml_predictions(target_loc, wr_df, inc_df, pic_list, nFeatCounts, splits, in_steps, out_steps, iters, verbosity):
    test_errors = {}
    forecast_windows = {}
    linear_models = {}
    lstm_models = {}
    gru_models = {}
    for nPIC in nFeatCounts:
        train_data, val_data, test_data, test_dates, train_mean, train_std, num_features = prepare_data(target_loc, wr_df, inc_df, pic_list[0:nPIC], splits, False)

        forecast_window = WindowGenerator(input_width=in_steps, label_width=out_steps, shift=out_steps, train_X = train_data, val_X = val_data, test_X = test_data, test_dates = test_dates, label_columns = ['cases'])

        linear_model, lstm_model, gru_model = initialize_models(1, out_steps)
                                                                                                    # window, epochs, verbose
        print(f'Fitting and predicting with {nPIC} additional features')
        test_error = fit_and_predict(linear_model, lstm_model, gru_model, forecast_window, iters, verbosity)

        test_errors[nPIC] = test_error
        forecast_windows[nPIC] = forecast_window
        linear_models[nPIC] = linear_model
        lstm_models[nPIC] = lstm_model
        gru_models[nPIC] = gru_model
#     print('Error Metric Index: ' + str(lstm_model.metrics_names.index('mean_absolute_error')))
    return test_errors, forecast_windows, linear_models, lstm_models, gru_models

def find_best_performers(feature_counts, test_errors, metric_index):
    linear_errors = []
    lstm_errors = []
    gru_errors = []
    
    error_df = pd.DataFrame()

    for nPIC in feature_counts:
        linear_errors.append(test_errors[nPIC]['Linear'][metric_index])
        lstm_errors.append(test_errors[nPIC]['LSTM'][metric_index])
        gru_errors.append(test_errors[nPIC]['GRU'][metric_index])

    error_df['nFeat'] = feature_counts
    error_df['linear_error'] = linear_errors
    error_df['lstm_error'] = lstm_errors
    error_df['gru_error'] = gru_errors
    error_df = error_df.set_index('nFeat')
    min_errors = error_df.idxmin()
    return min_errors
