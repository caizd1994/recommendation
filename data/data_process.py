import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from data.data_io import PosShifts, LineParser,DataParser


pkl_file = open("./raw_data.pkl", 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

writer = tf.python_io.TFRecordWriter("./train3.tfrecords")

# with tf.Session() as sess:
#     with open("./final_track2_train.txt") as f:
#         for lines in f:
#             temp = DataParser.data_parser(lines,7)
#             # print(temp[0]["feature_idx"])
#             cnt = 0
#             for elem in temp:
#                 if cnt == 0:
#                     print(list(elem))
#                     cnt+=1
#                 else:
#                     print(elem)
#

cnt = 0
for index, row in data.iterrows():

    # user_id, user_city, item_id，author_id，item_city
    if(cnt < 10000):
        # uid = min(max(int(row["uid"]),0),1)
        # city = min(max(int(row["city"]),0), 1)
        # item_id = min(max(int(row["item_id"]),0), 1)
        # author_id = min(max(int(row["author_id"]),0), 1)
        # item_city = min(max(int(row["item_city"]),0), 1)
        example = tf.train.Example(
            features= tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row['finish'])]) ),
                    "uid": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row['city']),int(row['item_id']),int(row['author_id']),int(row["item_city"])])),
                    # "city": tf.train.Feature(int64_list=tf.train.Int64List(value=[city])),
                    # "item_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[item_id])),
                    # "author_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[author_id])),
                    # "item_city": tf.train.Feature(int64_list=tf.train.Int64List(value=[item_city])),
                }
            )
        )
        cnt+=1
        writer.write(example.SerializeToString())
    else:
        break
writer.close()


# writer = tf.python_io.TFRecordWriter("../data/test.tfrecords")
# test = data[data['target'] == -1]
# for index, row in train.iterrows():
#     # print(row['target'])
#     temp = [0 for i in range(0,208)]
#     for index,elem in enumerate(row['question_text'].strip("[]").split(",")):
#         temp[index] = int(elem)
#     example = tf.train.Example(
#         features=tf.train.Features(
#             feature={
#                 'word': tf.train.Feature(int64_list=tf.train.Int64List(value=temp))
#             }
#         )
#     )
#     writer.write(example.SerializeToString())
# writer.close()