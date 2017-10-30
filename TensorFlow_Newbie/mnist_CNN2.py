#*- utf8
#@author：andywu1018@126.com
import  tensorflow as tf
import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
# mnist.train 训练样本 # *.images [55000,784]  *.labels  [55000,10] 标签 == 0到1
# mnist.test  测试样本 # *.images [10000,784]  *.labels  [10000,10]

#定义模型 ：softmax 回归   定义：softmax(x)i =  exp(xi)/Σ[exp(xj)]

# y = softmax(Wx+b)
x = tf.placeholder('float',[None,784]) #定义占位符入参x作为图片像素
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# y = softmax(Wx+b)
y = tf.nn.softmax(tf.matmul(x,W)+b)

# trainning
# 评价标准 ： 交叉熵
# H y'(y)  =  -Σy'ilog(yi)     *y是预测分布  y'是实际分布，需要我们手动输入
y_ = tf.placeholder('float',[None,10])
cross_entropy =  -tf.reduce_sum(y_*tf.log(y))

# 收敛方法 : 梯度下降法 下降速率0.01  损失函数:交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化所有变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#尽量使用with ,即停即闭

#开始训练模型
for  i in range(1000):
    batch_xs ,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

#评估模型
correct_prediction  = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

#正确率
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

#  ******NOTES
# 1.tf.matmul(W,x)  != tf.matmul(x,W)?    这个想当然了，根据矩阵乘法，换一下位置就会出错
# 2.我在初始化变量的时候用了 with 语句 ,导致在后面每一步都使用了 with 语句来打开执行sess，会报错
#   报错原因是变量未初始化,改成初始化之后不关闭sess之后不出现错误.
# 3.整个流程：先确定数据类型，数据大小，规则等；确定模型，入参使用placeholder，需要训练的参数使用Variable；
#   确定损失函数，本次使用交叉熵；确定优化算法（收敛），常用的有梯度下降算法、最小二乘法；随机训练；评估模型
