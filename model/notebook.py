# %% [markdown]
# # Introduction
# 
# This notebook is purely for development. When we want to create a new model, we just use this notebook, pull in data from somewhere (ideally generated from our `simulator`) and train a model. 
# 
# When we finish, we will convert this notebook to a script, that we execute (reading data from the same file, training the model, etc) as soon as the docker container starts. The file dumps a model that our server.py loads and uses to perform predictions against data coming in via POST requests from our simulator.
# 
# 
# ### Overview on the anomaly prediction system 
# 
# This model is trained on signals that are in a healthy state. It's the ground truth of our system. When we pass in a new waveform, our model will predict the expected waveform and do a validation against the prediction vs what we passed in. If we encounter a deviation from our input to what the model has predicted the waveform to behave like, we flag the input as anomalous, scaled in criticality based on a threshold. Ie, 10% error dictates a high level of risk for the business.

# %%
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import csv
import json

DEBUG = False

# %% [markdown]
# ## Loading our data

# %%
def load_data(filename): 
    """
    Load our .csv file
    returns two arrays, one for the time-steps
    and the second for the actual readings at those steps.
    """
    time_steps = []
    amplitudes = []
    with open(filename, mode='r') as file: 
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)
        for row in csv_reader: 
            raw_list = json.loads(row[1])
            for i, value in enumerate(raw_list): 
                amplitudes.append(value)
                time_steps.append(i)
            # we're only reading one row so we break here. 
            break 
    time = np.array(time_steps)
    amps = np.array(amplitudes)
    return time, amps
time, series = load_data("readings.csv")

# %%
def plot_values(debug=False, series=None): 
    if debug == True: 
        plt.plot(series, label="time values")
        plt.show()

def plot_series(x, y, format="-", start=0, end=None, title=None, xlabel=None, ylabel=None, legend=None): 
    if DEBUG == True: 
        plt.figure(figsize=(10, 6))
        if type(y) is tuple: 
            for y_curr in y: 
                plt.plot(x[start:end], y_curr[start:end], format)
        else: 
            plt.plot(x[start:end], y[start:end], format)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend: 
            plt.legend(legend)
        plt.title(title)
        plt.grid(True)
        plt.show()

plot_series(time, series)

# %% [markdown]
# ## Splitting our dataset
# 
# Here we start to split our dataset into train and validation sets. We'll follow through with building out windows of these datasets to pass into a tuning model, so as to find our optimal learning rate.

# %%
split_time = 3000 # out of a 4k long list of values.
time_train = time[:split_time]
x_train = series[:split_time]

time_valid = time[split_time:]
x_valid = series[split_time:]

print(f"time validation set shape: {time_valid.shape}")
print(f"time train set shape: {time_train.shape}")


# %% [markdown]
# #### Buffering our window sizes. 

# %%

from helpers import window
""" 
def window(series, window_size, batch_size, shuffle_buffer): 
    dataset = tf.data.Dataset.from_tensor_slices(series) # create the dataset frame
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True) # build out windows of a set size and shift by one for each 
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1)) # batches each window into concrete arrays
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(1)
    return dataset
"""

# %%
window_size = 30 
batch_size = 32 
shuffle = 1000
train_set = window(x_train, window_size, batch_size=batch_size, shuffle_buffer=shuffle)

# %% [markdown]
# ### Building our tuning model 

# %%
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,)), 
    tf.keras.layers.Dense(30, activation="relu"), 
    tf.keras.layers.Dense(30, activation="relu"), 
    tf.keras.layers.Dense(1)
])
model.summary()

# %%
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20 ))
optimizer = tf.keras.optimizers.Adam()
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

# %%
if DEBUG == True: 
    lrs = 1e-8 * (10 ** (np.arange(100) / 20))
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.semilogx(lrs, history.history['loss'])
    plt.tick_params('both', length=10, width=1, which='both')
    plt.axis([1e-8, 1e-3, 0, 100])

# %% [markdown]
# #### Train a new model 

# %%
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,)), 
    tf.keras.layers.Dense(30, activation='relu'), 
    tf.keras.layers.Dense(30, activation='relu'), 
    tf.keras.layers.Dense(1)
])
model.summary()

# %%
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
history = model.fit(train_set, epochs=30)

# %% [markdown]
# ### Build out the model predictions

# %%
from helpers import model_forecast
""" 
def model_forecast(model, series, window_size, batch_size): 
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset, verbose=0)
    return forecast
"""

# %%
forecast_series = series[split_time - window_size: - 1] # get the forecast series for the "validation set"
forecast = model_forecast(
    model, forecast_series, window_size, 
    batch_size) 
results = forecast.squeeze() 
plot_series(time_valid, (x_valid, results))

# %%
print("==== Final Results =====")
print(tf.keras.metrics.mse(x_valid, results).numpy())
print(tf.keras.metrics.mae(x_valid, results).numpy())
print("========================")

# %%
if DEBUG: 
    loss = history.history['loss']
    loss = loss[15:]
    plt.plot(loss)
    plt.show()

# %% [markdown]
# ### Dumping the model 
# 
# If our model meets certain criteria, then we dump it to a file we load from the server.

# %%
mae = tf.keras.metrics.mae(x_valid, results).numpy()
mse = tf.keras.metrics.mse(x_valid, results).numpy()
if mae <= 0.1 and mse <= 0.1: 
    print("saving model. all scores passed...")
    model.save('final.h5')

# %%

"""

import numpy as np
import tensorflow as tf

INPUT_VALUE = payload.json()  # make sure this is a NumPy array or convertible

# load the model
model = tf.keras.models.load_model('final.h5')

# convert input to np array and reshape if needed
input_array = np.array(INPUT_VALUE)

# predict
predictions = model.predict(input_array).flatten()

# calculate errors
errors = np.abs(input_array.flatten() - predictions)

# define threshold
threshold = np.mean(errors) + 3 * np.std(errors)

# flag anomalies
anomalies = errors > threshold

# calculate anomaly percentage
anomaly_percent = np.mean(anomalies) * 100

if anomaly_percent > 5:  # example threshold, adjust as needed
    print(f"Anomaly detected! {anomaly_percent:.2f}% of points are anomalous.")
else:
    print(f"No significant anomalies detected. Only {anomaly_percent:.2f}% anomalous.")


"""

# %%



