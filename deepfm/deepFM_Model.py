import tensorflow as tf
import numpy as np
import pandas as pd
from common_utils.utils import get_optimizer

class deepFM(object):
    def __init__(self):
        pass

    def build_FM_layer(self,features,parms):
        #line_part
        bias = tf.get_variable("fm_bias", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        line_emb = tf.feature_column.input_layer(features,parms['first_feature_columns'])
        line_logits = tf.reduce_sum(line_emb,axis=1,keepdims=True)

        fm_emb = tf.feature_column.input_layer(features,parms['second_feature_columns'])
        fm_emb = tf.reshape(fm_emb,shape=[-1,parms['fea_size'],parms['embedding_size']])
        #sum_squre
        sum_squre = tf.reduce_sum(fm_emb,axis=1,keepdims=True)
        sum_squre = tf.square(sum_squre)
        #squre_sum
        squre_sum = tf.square(fm_emb)
        squre_sum = tf.reduce_sum(squre_sum,axis=1,keepdims=True)

        second_order = tf.reduce_sum(0.5 * (sum_squre - squre_sum), -1, keep_dims=True)
        fm_logits = second_order + line_logits + bias
        return fm_logits

    def build_deep_layer(self,features,parms):
        units = parms['hidden_units']
        dim = units[-1]
        last_w = tf.get_variable("final_w", shape=[dim,1],initializer=tf.truncated_normal_initializer(stddev=0.01))
        last_b = tf.get_variable("bias_last", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        deep_net = tf.feature_column.input_layer(features,parms['second_feature_columns'])

        for layer in units:
            deep_net = tf.layers.dense(deep_net,layer,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer())
            if parms["is_bn"] == "true":
                deep_net = tf.layers.batch_normalization(deep_net)
            if parms["is_dropout"] == "true":
                deep_net = tf.layers.dropout(deep_net,rate=parms["dropout_rate"])
        deep_net = tf.matmul(deep_net , last_w) + last_b
        return deep_net

    def train_op(self,features,labels,params):
        labels = tf.cast(tf.reshape(labels, [-1, 1]), dtype=tf.float32)

        FM_logits = self.build_FM_layer(features,params)
        deep_logits = self.build_deep_layer(features,params)

        logits = FM_logits+deep_logits
        predicts = tf.nn.sigmoid(logits)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels)
        loss = tf.reduce_mean(loss)

        if(params['is_decy'] == "true"):
            learn_rate = tf.train.exponential_decay(params['learning_rate'],tf.train.get_global_step(),decay_steps=params['decay_steps'],
                                                    decay_rate=params['decay_rate'])
        else:
            learn_rate = params['decay_steps']

        optimize = get_optimizer(params["optimizer"],learning_rate=learn_rate,global_step=tf.train.get_global_step())\
            .minimize(loss, global_step=tf.train.get_global_step())

        eval_metric_ops = {
            "auc":tf.metrics.auc(labels=labels,predictions=predicts)
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            predictions=predicts,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            train_op=optimize
        )



