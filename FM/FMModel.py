import tensorflow as tf
import numpy as np
import pandas as pd
from common_utils.utils import get_optimizer

class FM_model(object):
    @staticmethod
    def train_op(features,labels,params):
        bias = tf.get_variable("bias", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        labels = tf.cast(tf.reshape(labels,[-1,1]),dtype = tf.float32)
        #first_order
        first_order = tf.feature_column.input_layer(features=features, feature_columns= params['first_feature_columns'])  #[batch_size, fea_size]
        first_order = tf.reduce_sum(first_order,axis = 1 ,keepdims=True)  #[batch_size ,1]

        #second_order
        second_order = tf.feature_column.input_layer(features=features, feature_columns=params['second_feature_columns'])
        second_order = tf.reshape(second_order,shape=[-1,params['fea_size'],params['embedding_size']])

        #square-sum
        square_sum = tf.square(second_order)
        square_sum = tf.reduce_sum(square_sum,axis= 1)

        #sum_square
        sum_square = tf.reduce_sum(second_order,axis=1)
        sum_square = tf.square(sum_square)

        second_order = tf.reduce_sum(0.5 * (sum_square - square_sum),-1,keep_dims=True)

        logits = second_order + first_order + bias
        predicts = tf.nn.sigmoid(logits)

        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
        sigmoid_loss = tf.reduce_mean(sigmoid_loss)

        optimizer = get_optimizer(params['optimizer'],learning_rate=params['learning_rate'])

        train = optimizer.minimize(sigmoid_loss,global_step=tf.train.get_global_step())


        eval_metric_ops = {
            "auc": tf.metrics.auc(labels,predicts)
        }

        return tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.TRAIN,
            predictions=predicts,
            loss=sigmoid_loss,
            eval_metric_ops=eval_metric_ops,
            train_op=train
        )