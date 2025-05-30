import tensorflow as tf 

# Preprocessing Functions

def window(series, window_size=30, batch_size=32, shuffle_buffer=1000): 
    """This function takes in a series and batches it into windows"""
    dataset = tf.data.Dataset.from_tensor_slices(series) # create the dataset frame
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True) # build out windows of a set size and shift by one for each 
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1)) # batches each window into concrete arrays
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(1)
    return dataset
    


def model_forecast(model, series, window_size, batch_size): 
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset, verbose=0)
    return forecast