import tensorflow as tf
from sklearn.metrics import f1_score,precision_score,recall_score


def acc_(y_label,y_predict):
    epoch_acc=tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_label,axis=-1),tf.argmax(y_predict,axis=-1)),"float"))
    return epoch_acc

def precision_(y_true, y_pred):
    return precision_score(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1),average='macro')

def recall_(y_true, y_pred):
    return recall_score(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1),average='macro')

def f1_score_(y_true, y_pred):
    return f1_score(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1),average='macro')
