import numpy as np
def shift():
    """ position shifts for different field features """
    shifts = [73974, 396, 4122689, 850308, 461, 5]
    shifts = [0] + shifts

    sum = 0
    for index, shift in enumerate(shifts):
        sum += shift
        shifts[index] = sum
    return shifts
if __name__ == '__main__':
    import tensorflow as tf

    # 两个矩阵的对应元素各自相乘！！
    x = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y = tf.constant([[0.0, 0.0,0.0]])
    # 注意这里这里x,y要有相同的数据类型，不然就会因为数据类型不匹配而出错
    z = tf.multiply(y,x)

    with tf.Session() as sess:
        print(sess.run(z))
    # label_index = 6
    # line = "57384	32	43192	142828	0	0	0	0	4513	34178	53085993699	39"
    # # idx = [0 if feature < 0 else feature for feature in features]
    # content = line.split('\t')
    # print(content)
    # label = np.float32(content[label_index].strip())
    #
    # feature_num = 5
    # features = content[:feature_num+1]
    # print(features)
    # features = list(map(lambda feature: np.float32(feature), features))
    # print(list(features))
    # idx = [0 if feature < 0 else feature  for feature in features]
    #
    # features = [np.float32(0) if feature < 0 else np.float32(1) for feature in features]
    # print(features)
    # features = features[:feature_num]
    # print(features)
    # idx = idx[:feature_num]
    # print(idx)
    # shifts = shift()
    # idx = [idx[i] + shifts[i] for i in range(len(idx))]
    # # for i in range(len(idx)):
    # #     print(idx[i] + shifts[i])
    # print(idx)
    # idx= map(lambda one_id: np.int32(one_id), idx)
    # print(list(idx))