import tensorflow as tf
def make_dataset(batch_sz, x, y=None, shuffle=False, shuffle_buffer_size=1000):
    """Create a slice Dataset from a pandas DataFrame and labels"""

    def input_fn():
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dict(x))
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_sz).repeat()
        else:
            dataset = dataset.batch(batch_sz)
        return dataset.make_one_shot_iterator().get_next()

    return input_fn