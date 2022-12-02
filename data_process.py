import numpy as np
import pandas as pd

all_num=5

#number of classes
num_way=5

#number of examples per class for support set
num_shot=1

#number of query points
num_query=5

#task batch size
batch=4

img_height=100
img_width=100
channels=3

maxlen=128


# data process
def few_shot(dataset_images, dataset_radiology_report, number):
    support_set_images = np.zeros([batch, num_way * num_shot, img_height, img_width, channels],
                                  dtype=np.float32)
    query_set_images = np.zeros([batch, num_way * num_query, img_height, img_width, channels],
                                dtype=np.float32)

    support_set_radiology_report = np.zeros([batch, num_way * num_shot, maxlen], dtype=np.int32)
    query_set_radiology_report = np.zeros([batch, num_way * num_query, maxlen], dtype=np.int32)

    support_labels = np.zeros([batch, num_way * num_shot], dtype=np.int32)
    query_labels = np.zeros([batch, num_way * num_query], dtype=np.int32)

    for i in range(batch):
        episodic_classes = np.random.permutation(all_num)[:num_way]
        # support_set
        support_images = np.zeros([num_way, num_shot, img_height, img_width, channels],
                                  dtype=np.float32)
        # support_set
        support_radiology_report = np.zeros([num_way, num_shot, maxlen], dtype=np.int32)

        # query_set
        query_images = np.zeros([num_way, num_query, img_height, img_width, channels],
                                dtype=np.float32)
        # query_set
        query_radiology_report = np.zeros([num_way, num_query, maxlen], dtype=np.int32)

        for index, class_ in enumerate(episodic_classes):
            selected = np.random.permutation(number)[:num_shot + num_query]
            selected_0 = selected[:num_shot]
            selected_1 = selected[num_shot:]

            # image
            support_images[index] = dataset_images[class_, selected_0]
            # radiology_report
            support_radiology_report[index] = dataset_radiology_report[class_, selected_0]

            # image
            query_images[index] = dataset_images[class_, selected_1]
            # radiology_report
            query_radiology_report[index] = dataset_radiology_report[class_, selected_1]

        support_images = np.expand_dims(support_images, axis=-1).reshape(num_way * num_shot, img_width, img_height,
                                                                         channels)
        query_images = np.expand_dims(query_images, axis=-1).reshape(num_way * num_query, img_width, img_height,
                                                                     channels)

        support_radiology_report = np.expand_dims(support_radiology_report, axis=-1).reshape(num_way * num_shot, maxlen)
        query_radiology_report = np.expand_dims(query_radiology_report, axis=-1).reshape(num_way * num_query, maxlen)

        support_set_images[i] = support_images
        query_set_images[i] = query_images

        support_set_radiology_report[i] = support_radiology_report
        query_set_radiology_report[i] = query_radiology_report

        s_labels = []
        q_labels = []
        for j in range(num_way):
            s_labels = s_labels + [episodic_classes[j]] * num_shot
            q_labels = q_labels + [episodic_classes[j]] * num_query
        support_labels[i] = np.array(s_labels)
        query_labels[i] = np.array(q_labels)
    return support_set_images, query_set_images, support_set_radiology_report, query_set_radiology_report, support_labels, query_labels
