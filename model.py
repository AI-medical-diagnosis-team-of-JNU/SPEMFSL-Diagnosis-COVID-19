import numpy as np
import pandas as pd
import tensorflow as tf


num_words=2000
maxlen=128

#InceptionV3
covn_base=tf.keras.applications.InceptionV3(weights='imagenet',include_top=False)

for layer in covn_base.layers:
    layer.trainable=False


def create_model_images(input_1_shape):
    # inputs=tf.keras.Input(input_1_shape)
    layer = covn_base(input_1_shape)
    fla = tf.keras.layers.GlobalAveragePooling2D()(layer)
    den1 = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                 bias_regularizer=tf.keras.regularizers.l1(0.01))(fla)
    den2 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                 bias_regularizer=tf.keras.regularizers.l1(0.01))(den1)
    model = tf.keras.models.Model(input_1_shape, den2)
    return model

def create_model_radiology_report(input_2_shape):
    #inputs=tf.keras.Input(input_2_shape)
    emd=tf.keras.layers.Embedding(num_words,256,input_length=maxlen)(input_2_shape)
    lstm=tf.keras.layers.LSTM(units=128)(emd)
    den1=tf.keras.layers.Dense(64,activation='relu')(lstm)
    model=tf.keras.models.Model(input_2_shape,den1)
    return model


def merge_model(input_1_shape, input_2_shape):
    input1 = tf.keras.Input(input_1_shape)
    input2 = tf.keras.Input(input_2_shape)
    model1 = create_model_images(input1)
    model2 = create_model_radiology_report(input2)
    r1 = model1.output
    r2 = model2.output

    basal_1 = tf.keras.layers.Dense(128, activation='relu')(r2)
    basal_2 = tf.keras.layers.Dense(64, activation='sigmoid')(basal_1)

    x = tf.keras.layers.multiply([r1, basal_2])

    x1 = tf.keras.layers.Dense(32, activation='relu')(x)
    x2 = tf.keras.layers.Dense(16, activation='relu')(x1)
    x3 = tf.keras.layers.Dense(5)(x2)
    model = tf.keras.models.Model([input1, input2], x3)
    return model



