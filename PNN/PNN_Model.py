import tensorflow as tf
import numpy as np
import pandas as pd
from common_utils.utils import get_optimizer


def embedding_prepare(features,params):
    embed_size = params["embedding_size"]
    mode = params["mode"]
    if mode == "inner":
        fea_size = len(params['second_feature_columns'])
        num_pairs = int(fea_size * (fea_size - 1) / 2)
        raw_emb = tf.feature_column.input_layer(features=features,feature_columns=params['second_feature_columns']) #B,N*E
        emb = tf.reshape(raw_emb, [-1, fea_size, embed_size]) #B,N,E
        row = []
        col = []
        for i in range(fea_size):
            for j in range(i+1,fea_size):
                row.append(i)
                col.append(j)
        p = tf.transpose(
            tf.gather(tf.transpose(emb,[1,0,2]),indices=row),
            [1,0,2]
        )
        q = tf.transpose(
            tf.gather(tf.transpose(emb,[1,0,2]),indices=col),
            [1,0,2]
        )
        p = tf.reshape(p, [-1, num_pairs, embed_size])
        q = tf.reshape(q, [-1, num_pairs, embed_size])
        inner_product = tf.reduce_sum(p * q,axis=-1)
        inner_product = tf.reshape(inner_product,shape=[-1,num_pairs])
        result = tf.concat([raw_emb,inner_product],axis=1)
        return result
        # batch * n * 1 * k, batch * 1 * n * k
        # ip = tf.reshape(
        #     tf.reduce_sum(
        #         tf.expand_dims(xw3d, 2) *
        #         tf.expand_dims(xw3d, 1),
        #         3),
        #     [-1, num_inputs**2])

    elif mode == "out":
        fea_size = len(params['second_feature_columns'])
        num_pairs = int(fea_size * (fea_size - 1) / 2)
        kernel = None
        if(params["kernel"] == "mat"):
            kernel = tf.get_variable("kernel_product",shape=[embed_size,fea_size,embed_size])
        raw_emb = tf.feature_column.input_layer(features=features,feature_columns=params['second_feature_columns']) #B,N*E
        emb = tf.reshape(raw_emb, [-1, fea_size, embed_size]) #B,N,E
        row = []
        col = []
        for i in range(fea_size):
            for j in range(i+1,fea_size):
                row.append(i)
                col.append(j)
        p = tf.transpose(
            tf.gather(tf.transpose(emb,[1,0,2]),indices=row),
            [1,0,2]
        )
        q = tf.transpose(
            tf.gather(tf.transpose(emb,[1,0,2]),indices=col),
            [1,0,2]
        )
        p = tf.reshape(p, [-1, num_pairs, embed_size])
        q = tf.reshape(q, [-1, num_pairs, embed_size])
        result = None
        kp = None
        if params["kernel"] == "mat":
            # batch * 1 * pair * k
            p = tf.expand_dims(p, 1)
            # batch * pair
            kp = tf.reduce_sum(
                # batch * pair * k
                tf.multiply(
                    # batch * pair * k
                    tf.transpose(
                        # batch * k * pair
                        tf.reduce_sum(
                            # batch * k * pair * k
                            tf.multiply(
                                p, kernel),
                            -1),
                        [0, 2, 1]),
                    q),
                -1)
        else:
            # 1 * pair * (k or 1)
            k = tf.expand_dims(kernel, 0)
            # batch * pair
            kp = tf.reduce_sum(p * q * k, -1)
        result = tf.concat([raw_emb, kp], axis=1)
        # inner_product = tf.reduce_sum(p*q,axis=-1)
        # inner_product = tf.reshape(inner_product,shape=[-1,num_pairs])
        # result = tf.concat([raw_emb,inner_product],axis=1)
        return result


def deep_part(deep_net,params):
    units = params['hidden_units']
    dim = units[-1]
    last_w = tf.get_variable("final_w", shape=[dim, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
    last_b = tf.get_variable("bias_last", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    for elem in units:
        deep_net = tf.layers.dense(deep_net,elem,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer())
        if params["is_bn"] == "true":
            deep_net = tf.layers.batch_normalization(deep_net)
        if params["is_dropout"] == "true":
            deep_net = tf.layers.dropout(deep_net, rate=params["dropout_rate"])
    deep_net = tf.matmul(deep_net, last_w) + last_b
    return deep_net


def train_op(features,labels,params):
    labels = tf.cast(tf.reshape(labels, [-1, 1]), dtype=tf.float32)

    input_emb = embedding_prepare(features,params)
    deep_logits = deep_part(input_emb, params)

    predicts = tf.nn.sigmoid(deep_logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=deep_logits,labels=labels)
    loss = tf.reduce_mean(loss)

    if (params['is_decy'] == "true"):
        learn_rate = tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(),
                                                decay_steps=params['decay_steps']
                                                ,decay_rate=params['decay_rate'])
    else:
        learn_rate = params['decay_steps']

    optimize = get_optimizer(params["optimizer"], learning_rate=learn_rate, global_step=tf.train.get_global_step()) \
        .minimize(loss, global_step=tf.train.get_global_step())

    eval_metric_ops = {
        "auc": tf.metrics.auc(labels=labels, predictions=predicts)
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        predictions=predicts,
        loss=loss,
        eval_metric_ops=eval_metric_ops,
        train_op=optimize
    )