import tensorflow as tf
import numpy as np
import pandas as pd
from common_utils.utils import get_optimizer


def cross_layer(input_x,x_l,dim,name):
    with tf.variable_scope(name):
        w = tf.get_variable("weight",[dim],initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("bias",[dim],initializer=tf.truncated_normal_initializer(stddev=0.01))
        x_next = tf.tensordot(tf.reshape(x_l,[-1,1,dim]),w,1)
    return input_x * x_next + b + x_l

def build_cross_layers(x0, params):
    num_layers = params['num_cross_layers']
    x = x0
    dim = x.get_shape().as_list()[1]
    for i in range(num_layers):
        x = cross_layer(input_x=x0, x_l=x,dim=dim, name='cross_{}'.format(i))
    return x

def build_deep_layers(input_x,parms):
    deep_emb = input_x
    for elem in parms['hidden_units']:
        deep_emb = tf.layers.dense(deep_emb, units=elem, activation=tf.nn.relu)
    return deep_emb


def train_op(features,labels,params):
    labels = tf.cast(tf.reshape(labels, [-1, 1]), dtype=tf.float32)
    in_put_x = tf.feature_column.input_layer(features=features,feature_columns=params['second_feature_columns'])
    deep_emb = build_deep_layers(in_put_x,params)
    cross_emb = build_cross_layers(in_put_x,params)
    concat_emb = tf.concat([cross_emb, deep_emb], axis=1)
    shape_w = concat_emb.get_shape().as_list()[1]
    print(shape_w)
    # last_w = tf.get_variable("final_w", shape=[shape_w,1],initializer=tf.truncated_normal_initializer(stddev=0.01))
    # last_b = tf.get_variable("bias_last", [1], initializer=tf.constant([0]))
    # logits = concat_emb * last_w + last_b
    logits = tf.layers.dense(concat_emb,1,activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                             ,use_bias=True)
    predicts = tf.nn.sigmoid(logits)

    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    sigmoid_loss = tf.reduce_mean(sigmoid_loss)


    if params["is_decy"] == "true":
        learn_rate = tf.train.exponential_decay(params['learning_rate'],tf.train.get_global_step(),decay_steps=2000,decay_rate=0.9)
    else:
        learn_rate = params['learning_rate']
    optimizer = get_optimizer(params['optimizer'], learning_rate=learn_rate)
    train = optimizer.minimize(sigmoid_loss, global_step=tf.train.get_global_step())

    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, predicts)
    }
    auc, update_op = tf.metrics.auc(labels, predicts)
    tf.summary.scalar('auc', update_op)

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        predictions=predicts,
        loss=sigmoid_loss,
        eval_metric_ops=eval_metric_ops,
        train_op=train
    )




