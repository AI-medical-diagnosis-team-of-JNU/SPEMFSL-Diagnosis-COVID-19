

import numpy as np
import pandas as pd
import tensorflow as tf

def acc_(y_label,y_predict):
    epoch_acc=tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_label,axis=-1),tf.argmax(y_predict,axis=-1)),"float"))
    return epoch_acc