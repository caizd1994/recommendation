import tensorflow as tf

a = tf.constant([[[1,2],[3,4],[5,6]]
                    ,[[7,8],[9,10],[11,12]]
                 ,[[13,14],[15,16],[17,18]]
                 ]
                )

k = tf.constant([[[1,2],[3,4],[5,6]]
                    ,[[7,8],[9,10],[11,12]]
                 ]
                )
row = []
col = []
for i in range(3):
    for j in range(i + 1, 3):
        row.append(i)
        col.append(j)
print(len(row))
print(row)
print(col)
print(len(col))

p = tf.transpose(
    tf.gather(
        a,row
    ),
    [1,0,2]
)

q = tf.transpose(
    tf.gather(
        a,col
    ),
    [1,0,2]
)
p = tf.reshape(p, [-1, 3, 2])
p = tf.expand_dims(p, 1)
temp =tf.reduce_sum(tf.multiply(p, k),-1)
# p = tf.expand_dims(p, 1)
kp = tf.reduce_sum(
    # batch * pair * k
    tf.multiply(
        # batch * pair * k
        tf.transpose(
            # batch * k * pair
            tf.reduce_sum(
                # batch * k * pair * k
                tf.multiply(
                    p, k),
                -1),
            [0, 2, 1]),
        q),
    -1)

with tf.Session() as sess:
    print(k)
    print(p)
    print(sess.run(p))
    print("-" * 50)
    print(sess.run(temp))
    print(temp)












    # print(sess.run(tf.expand_dims(a,1)))
    # print("*" * 50)
    # print(sess.run(a))
    # print(sess.run(tf.transpose(a,[2,0,1])))
    # print(sess.run(tmp1))
    # print("*" * 50)
    # print(sess.run(tmp2))


