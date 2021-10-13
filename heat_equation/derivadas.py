import tensorflow as tf

def diff(f, x, h, order=1):

    if order == 1:

        return (-f(x + 2 * h) + 8 * (f(x + h) - f(x - h)) + f(x - 2 * h)) / (12 * tf.norm(h))

    if order == 2:

        return (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2 * h)) / (12 * tf.square(tf.norm(h)))
