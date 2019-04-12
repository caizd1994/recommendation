import tensorflow as tf

# a = []
# with open("C:\\Users\\Administrator\\Downloads\\idex_1553689939867.csv") as f:
#     for lines in f:
#         field = lines.strip("\n").split(",")
#         if(field[0] == "20190326"):
#             a.extend(field[2].split(";"))
# print(a)
# b = []
# with open("C:\\Users\\Administrator\\Downloads\\idex_1553690777516.csv") as f:
#     for lines in f:
#         field = lines.strip("\n").split(",")
#         if(field[0] == "20190326"):
#             b.extend(field[2].split(";"))
# a = set(a)
# print(b)
# b = set(b)
# print(len(a))
# print(len(b))
# print(a.difference(b))

a = "你和我一样也算错过吗？一起来寻找答桉吧！|你和我一样也算错过吗？一起来寻找答桉吧！|null|1545877413973514|11168|https://h5.qzone.qq.com/weishi/feed/74Y2p2HSa1H6PuzQQ|http://pic200.weishi.qq.com/066cdff93b1e4bf2b0442a7d2c0bcover.jpg|e5b883e4b881e788b1e5ada6e4b9a0|1553143187|https://xp.qpic.cn/oscar_pic/0/1047_63646665653531322d313732392pict/0|-1.0|60|0.5357142857142857|-1.0|1|1|1|0|1499888.0|-187.0|NaN|20|5|场景_家庭,剧情_秀智商,剧情_脑筋急转弯|知识,纸上答题|15632".encode("hex")
print(len(a.split("|")))


# with tf.Session() as sess:
#     # print(sess.run(tf.random_normal(shape=[1, 2, 2])))
#     x = tf.constant(1,shape=[1,2,2])
#     print(sess.run(x))
#     sum = tf.reduce_sum(x,axis=2,keepdims=True)
#     print(sess.run(sum))