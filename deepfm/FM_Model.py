import tensorflow as tf
import pandas as pd
import numpy as np
from common_utils.utils import get_optimizer

class FM(object):
    def __init__(self,dim_v,dim_fea,fea_size,optimizer_type):
        self.v = tf.random_normal([dim_fea,dim_v],stddev=0.01,mean=0.0) #交叉项权值矩阵
        self.w = tf.random_normal([dim_v])  #线性部分权值
        self.x = tf.placeholder(tf.float32,[None,fea_size],name="input_x") #样本
        self.label = tf.placeholder(tf.int32,[None,1],name="label")
        self.b = tf.random_normal([fea_size])
        self.optimizer_type = optimizer_type  #优化器
    def train_op(self):
        #get FM result
        sum_squre = tf.pow(tf.matmul(tf.transpose(self.v),self.x),2)
        squre_sum = tf.matmul(tf.pow(self.x,2),tf.pow(self.v,2))
        single_result = tf.subtract(sum_squre - squre_sum)

        interactions_part = 0.5 * tf.reduce_sum(single_result,axis=1,keep_dims=True)

        line_part = tf.add(self.b + tf.reduce_sum(tf.multiply(self.w,self.x),axis=1,keep_dims=True))

        logits = tf.add(interactions_part,line_part)

        loss = tf.reduce_mean(tf.square(self.label - logits))

        get_optimizer(self.optimizer_type,learning_rate=0.001).minimize(loss)
        return loss

