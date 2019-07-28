import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%matplotlib inline #如果你使用的是jupyter notebook取消注释
np.random.seed(1)
def ComputeCost(Z3,Y):
    """
    计算成本

    参数：
        Z3 - 前向传播的结果
        Y - 标签，一个占位符，和Z3的维度相同

    返回：
        cost - 成本值


    """
    logits = tf.transpose(Z3) #转置
    labels = tf.transpose(Y)  #转置

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    return cost

def InitParameter(layer_size): # include input layer
    deep = len(layer_size)
    p = {}
    for i in range(1, deep):
        w_shape = [layer_size[i], layer_size[i-1]]
        b_shape = [layer_size[i], 1]
        w_name = "W" + str(i)
        b_name = "b" + str(i)
        temp_w = tf.compat.v1.get_variable(w_name, w_shape, initializer=tf.contrib.layers.xavier_initializer(seed=1))
        temp_b = tf.compat.v1.get_variable(b_name, b_shape, initializer=tf.zeros_initializer())
        p[w_name] = temp_w
        p[b_name] = temp_b
    return p

def ForwardPropagation(X, parameters):
    deep = len(parameters) // 2 # don't include input layer
    a = X
    for i in range(1, deep + 1):
        w = parameters["W" + str(i)]
        b = parameters["b" + str(i)]
        z = tf.add(tf.matmul(w, a), b)
        if i < deep:
            a = tf.nn.relu(z)
    return z

def GestureRecognition():
    learning_rate = 0.0001
    num_epochs = 1500
    minibatch_size = 32
    print_cost = True
    is_plot = True
    costs = []

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T #每一列就是一个样本
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

    m = X_train_flatten.shape[1]

    #归一化数据
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255

    #转换为独热矩阵
    Y_train = tf_utils.convert_to_one_hot(Y_train_orig,6)
    Y_test = tf_utils.convert_to_one_hot(Y_test_orig,6)

    tf.set_random_seed(1)
    seed = 3
    tf.compat.v1.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形。

    X = tf.compat.v1.placeholder(tf.float32, [X_train.shape[0], None], name="X")
    Y = tf.compat.v1.placeholder(tf.float32, [Y_train.shape[0], None], name="Y")

    parameters = InitParameter([12288, 25, 12, 6])

    z = ForwardPropagation(X, parameters)
    cost = ComputeCost(z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0  # 每代的成本
            num_minibatches = int(m / minibatch_size)  # minibatch的总数量
            seed = seed + 1
            minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # 选择一个minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # 数据已经准备好了，开始运行session
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches

            # 记录并打印成本
            ## 记录成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                # 是否打印：
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")

        # 计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(z), tf.argmax(Y))

        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

GestureRecognition()