import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score,precision_score,recall_score

inner_optimizer=tf.keras.optimizers.Adam(learning_rate_inner)
outer_optimizer=tf.keras.optimizers.Adam(learning_rate_outer)

inner_train_step=5
update_test_step=3


def train_step(support_set_images, query_set_images, support_set_radiology_report, query_set_radiology_report,
               support_labels_onehot, query_labels_onehot):
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    meta_weights = model.get_weights()
    with tf.GradientTape() as query_t:
        for index in range(batch):
            model.set_weights(meta_weights)
            support_data = support_set_images[index]
            query_data = query_set_images[index]
            support_radiology_report = support_set_radiology_report[index]
            query_radiology_report = query_set_radiology_report[index]
            support_la = support_labels_onehot[index]
            query_la = query_labels_onehot[index]
            for inner_step in range(inner_train_step):
                with tf.GradientTape() as support_t:
                    support_logits = model([support_data, support_radiology_report])
                    support_loss = loss_(support_la, support_logits)
                inner_grads = support_t.gradient(support_loss, model.trainable_variables)
                inner_optimizer.apply_gradients(zip(inner_grads, model.trainable_variables))
            query_logits = model([query_data, query_radiology_report])
            query_pred = tf.nn.softmax(query_logits)
            query_loss = loss_(query_la, query_logits)

            epoch_acc = acc_(query_la, query_pred)
            epoch_precision = precision_score(tf.argmax(query_la, axis=-1), tf.argmax(query_pred, axis=-1),
                                              average='macro')
            epoch_recall = recall_score(tf.argmax(query_la, axis=-1), tf.argmax(query_pred, axis=-1), average='macro')
            epoch_f1 = f1_score(tf.argmax(query_la, axis=-1), tf.argmax(query_pred, axis=-1), average='macro')

            train_loss.append(query_loss)
            train_acc.append(epoch_acc)
            train_precision.append(epoch_precision)
            train_recall.append(epoch_recall)
            train_f1.append(epoch_f1)

        meta_batch_loss = tf.reduce_mean(tf.stack(train_loss))
        model.set_weights(meta_weights)
        outer_grads = query_t.gradient(meta_batch_loss, model.trainable_variables)
        outer_optimizer.apply_gradients(zip(outer_grads, model.trainable_variables))
    every_loss = [loss.numpy() for loss in train_loss]
    every_acc = [acc.numpy() for acc in train_acc]
    every_precision = [precision for precision in train_precision]
    every_recall = [recall for recall in train_recall]
    every_f1 = [f1 for f1 in train_f1]
    return meta_batch_loss, every_loss, every_acc, every_precision, every_recall, every_f1

