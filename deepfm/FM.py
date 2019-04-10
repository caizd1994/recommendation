import tensorflow as tf
import numpy as np
import pandas as pd
from common_utils.utils import get_optimizer

class FM(object):
    def __init__(self):
        self.bias = tf.get_variable("bias",shape= [1] ,dtype=tf.float32,initializer = tf.constant_initializer(0.0))


    def train_op(self,train_data,labels,line_col,second_col,params):
        labels = tf.cast(tf.reshape(labels,[-1,1]),dtype = tf.float32)
        #first_order
        first_order = tf.feature_column.input_layer(features=train_data, feature_columns= line_col)  #[batch_size, fea_size]
        first_order = tf.reduce_sum(first_order,axis = 1 ,keepdims=True)  #[batch_size ,1]

        #second_order
        second_order = tf.feature_column.input_layer(features=train_data, feature_columns=second_col)

        #square-sum
        square_sum = tf.square(second_order)
        square_sum = tf.reduce_sum(square_sum,axis= 1 , keepdims=True)

        #sum_square
        sum_square = tf.reduce_sum(second_order,axis=1,keepdims=True)
        sum_square = tf.square(sum_square)

        second_order = 0.5 * (sum_square - square_sum)

        logits = second_order + first_order + self.bias
        predicts = tf.nn.sigmoid(logits)

        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
        sigmoid_loss = tf.reduce_mean(sigmoid_loss)

        optimizer = get_optimizer(params['get_optimizer'],learning_rate=params["learn_rate"],global_step=tf.train.get_global_step)

        train = optimizer.minimize(sigmoid_loss,global_step=tf.train.get_global_step)

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