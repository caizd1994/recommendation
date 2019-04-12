import tensorflow as tf

a = tf.constant([[[1,2],[3,4],[5,6]]
                    ,[[7,8],[9,10],[11,12]]
                 ,[[13,14],[15,16],[17,18]]
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

tmp1 = tf.transpose(
    tf.gather(
        a,row
    ),
    [1,0,2]
)

tmp2 = tf.transpose(
    tf.gather(
        a,col
    ),
    [1,0,2]
)

with tf.Session() as sess:
    print(sess.run(tf.expand_dims(a,1)))
    print("*" * 50)
    print(sess.run(a))

    # print(sess.run(tf.transpose(a,[2,0,1])))
    # print(sess.run(tmp1))
    # print("*" * 50)
    # print(sess.run(tmp2))


