from __future__ import print_function

import tensorflow as tf
import numpy as np
from  urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
import lexstring

from sklearn.model_selection import train_test_split

class readfile:
    def pre_file(file_in, file_out=None):
        with open(file_in, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        res = []
        for i in range(len(lines)):
            line = lines[i].strip()
            # 提取 GET类型的数据
            if line.startswith("GET"):
                res.append(line.split(" ")[1])
            # 提取 POST类型的数据
            elif line.startswith("POST") or line.startswith("PUT"):
                method = line.split(' ')[0]
                url = line.split(' ')[1]
                j = 1
                # 提取 POST包中的数据
                while True:
                    # 定位消息正文的位置
                    if lines[i + j].startswith("Content-Length"):
                        break
                    j += 1
                j += 2
                data = lines[i + j].strip()
                url += '?' + data
                res.append(url)
        return res

    def pre_file2(normal_file_raw):
        result = readfile.pre_file(normal_file_raw)
        rels = []
        for url in result:
            url_re = urlparse(url)
            if len(url_re.query) == 0:
                rel = url_re.path
            else:
                rel = url_re.path + "?" + url_re.query
            rels.append(rel)
        return rels


    # 将样本数据合并
    def dataconbine(normal, abnormal):
        alldata = normal + abnormal
        return alldata

    # 将标签数据合并
    def label(normal,abnormal):
        yBad = [[1,0] for i in range(0, len(abnormal))]  # labels, 1 for malicious and 0 for clean
        yGood = [[0,1] for i in range(0, len(normal))]
        ylabel = yGood+yBad
        return ylabel


class feature:
    def feature_process_word(data):
        vectorizer = TfidfVectorizer(min_df=0.0, analyzer="word", lowercase=False, sublinear_tf=True,
                                     token_pattern=r"(?u)\b\w+\b|[^a-zA-Z0-9]|",
                                     ngram_range=(1, 1))  # token_pattern=r"(?u)\b\w+\b|[^a-zA-Z0-9]|" 保存全部字符
        vectorizer.fit(data)
        X_vec = vectorizer.transform(data)
        return X_vec

class divide_data:
    def divicde(data1,label):
        train_indices = np.random.choice(data1.shape[0], round(0.8 * data1.shape[0]),
                                         replace=False)
        test_indices = np.array(list(set(range(data1.shape[0])) - set(train_indices)))
        texts_train1 = data1[train_indices]
        texts_test1 = data1[test_indices]
        #texts_train2 = np.array([y for iy, y in enumerate(data2) if iy in train_indices])
        #texts_test2 = np.array([y for iy, y in enumerate(data2) if iy in test_indices])
        target_train = np.array([y for iy, y in enumerate(label) if iy in train_indices])
        target_test = np.array([y for iy, y in enumerate(label) if iy in test_indices])
        return texts_train1,texts_test1,target_train,target_test


class LSTM_model:

    def main(self,texts_train1,text_test1,target_train,target_test):
        # 设置用到的参数
        lr0 = 0.001
        global_step = tf.Variable(0)
        lr_decay = 0.99
        lr_step = 500
        # 在训练和测试的时候 想使用不同的batch_size 所以采用占位符的方式
        batch_size = tf.placeholder(tf.int32, [])
        # 输入数据是28维 一行 有28个像素
        input_size = 45
        # 时序持续时长为28  每做一次预测，需要先输入28行
        timestep_size = 1
        # 每个隐含层的节点数
        hidden_size = 20
        # LSTM的层数
        layer_num = 2
        # 最后输出的分类类别数量，如果是回归预测的呼声应该是1
        class_num = 2
        _X = tf.placeholder(tf.float32, [None,1, 45])
        y = tf.placeholder(tf.float32, [None, class_num])
        keep_prob = tf.placeholder(tf.float32)
        lr = tf.train.exponential_decay(
            lr0,
            global_step,
            decay_steps=lr_step,
            decay_rate=lr_decay,
            staircase=True)
        # 定义一个LSTM结构， 把784个点的字符信息还原成28*28的图片
        X = tf.reshape(_X, [-1, 1, 45])

        def unit_lstm():
            # 定义一层LSTM_CELL hiddensize 会自动匹配输入的X的维度
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
            # 添加dropout layer， 一般只设置output_keep_prob
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            return lstm_cell

        # 调用MultiRNNCell来实现多层 LSTM
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([unit_lstm() for i in range(2)], state_is_tuple=True)

        # 使用全零来初始化state
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state,
                                           time_major=False)
        h_state = outputs[:, -1, :]

        # 设置loss function 和优化器
        W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
        y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
        # 损失和评估函数
        cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
        train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # 开始训练
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            _batch_size = 10
            start = (i * _batch_size) % texts_train1.shape[0]  #
            end = start + _batch_size
            batch_len = texts_train1[start:end].shape[0]
            batch_x = texts_train1[start:end].reshape((batch_len, 1, 45))
            batch_y = target_train[start:end].reshape((batch_len, 2))
            sess.run([train_op], feed_dict={_X: batch_x, y: batch_y,keep_prob: 1.0,batch_size: _batch_size})
            if (i + 1) % 200 == 0:
                loss,train_accuracy = sess.run([cross_entropy,accuracy], feed_dict={
                    _X: batch_x, y: batch_y, keep_prob: 1.0, batch_size: _batch_size
                })
                print("step %d,loss %s, training accuracy %g" % ((i + 1), loss,train_accuracy))

        len1=len(texts_test1)
        print("test accuracy %g" % sess.run(accuracy, feed_dict={_X: texts_test1[:].reshape((len1,1,45)), y:target_test.reshape((len1, 2)), keep_prob: 1.0,
                                                                 batch_size: len1}))




if __name__ == '__main__':
    data1=readfile.pre_file2('C:\\Users\\asus\\Desktop\\waf_test\\data\\cisc_normalTraffic_train.txt')
    data2=readfile.pre_file2('C:\\Users\\asus\\Desktop\\waf_test\\data\\cisc_anomalousTraffic_test.txt')
    #x_data1=np.array(feature.vector_five(data1))
    #x_data2=np.array(feature.vector_five(data2))
    #all_data1 = np.concatenate([x_data1, x_data2])
    all_data = readfile.dataconbine(data1,data2)
    all_data = lexstring.tokenstring(all_data)
    all_data=feature.feature_process_word(all_data).toarray()
    #y_normal = np.zeros(shape=(x_data1.shape[0]), dtype='int')
    #y_anomalous = np.ones(shape=(x_data2.shape[0]), dtype='int')
    #ylabel= np.concatenate([y_normal, y_anomalous])
    y_label=np.array(readfile.label(data1,data2))
    #texts_train1, texts_test1, target_train, target_test=divide_data.divicde(all_data,y_label)
    texts_train1, texts_test1, target_train, target_test = train_test_split(all_data,y_label, test_size=0.3)

    LSTM_model().main(texts_train1, texts_test1, target_train, target_test)