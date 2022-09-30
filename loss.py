
import numpy as np
import pandas as pd
import tensorflow as tf

def loss_(labels,pred):
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    result=loss(labels,pred)
    return result