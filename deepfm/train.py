import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from data.data_io import PosShifts,LineParser
from deepfm.FM_Model import FMModel
import os
def prepare_data_fn(batch_size,num_epochs,task="finish",data_mode='train'):
    """ prepare train, val fn"""
    if data_mode == 'train':
        dataset = tf.data.TextLineDataset("./final_track2_train.txt")
    else:
        raise Exception("unknown data_mode", data_mode)

    if task == "finish":
        dataset = dataset.map(LineParser.parse_finish_line)
    elif task == "like":
        dataset = dataset.map(LineParser.parse_like_line)
    else:
        raise Exception("unknown task", task)

    dataset = dataset.shuffle(buffer_size=30)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    data_iterator = dataset.make_one_shot_iterator()
    idx, features, labels = data_iterator.get_next()
    feature_infos = {}
    feature_infos['feature_idx'] = idx
    feature_infos['feature_values'] = features
    tf.logging.info(labels)

    return feature_infos, labels









if __name__ == '__main__':
    # train_next = prepare_data_fn(128,1,"finish")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'CPU':1}),
        log_step_count_steps=1)
    PosShifts(track=2)
    feature_size = PosShifts.get_features_num()
    print(feature_size)
    params={
        'feature_size': feature_size,
        'embedding_size': 128 ,
        'learning_rate': 0.001,
        'field_size': 5,
        'batch_size': 512,
        'optimizer': "adam"}

    model = tf.estimator.Estimator(
        model_fn=FMModel.fm_model_fn,
        model_dir="./",
        params=params,
        config=config)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: prepare_data_fn(batch_size=512,num_epochs=1,data_mode='train'))
    val_spec = tf.estimator.EvalSpec(input_fn=lambda: prepare_data_fn(batch_size=512,num_epochs=1,data_mode='val'))
    tf.estimator.train_and_evaluate(model, train_spec, val_spec)