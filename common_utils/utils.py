import numpy as np
import tensorflow as tf
import pandas as pd


def get_optimizer(optimizer, learning_rate,global_step=0.9):
    if optimizer == "sgd":
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == "adadelta":
        return tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                          rho=0.95,
                                          epsilon=1e-08)
    elif optimizer == "adagrad":
        return tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif optimizer == "adam":
        return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=0.95,
                                      beta2=0.999,
                                      epsilon=1e-09)
    elif optimizer == "momentum":
        return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                          momentum=0.9)
    elif optimizer == "ftrl":
        return tf.train.FtrlOptimizer(learning_rate)
    elif optimizer == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                         decay=0.9,
                                         momentum=0.0,
                                         epsilon=1e-10)
    elif optimizer == "adagradDA":
        global_step = tf.cast(global_step,tf.int64)
        return tf.train.AdagradDAOptimizer(learning_rate=learning_rate,global_step=global_step)
    else:
        exit(1)