from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from  urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import lexstring
from numpy import hstack

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


    def replacement(data):
        result = []
        if data == None:
            return 0
        else:
            for i in data:
                str1 = re.sub('[0-9]', 'S', i)  # 将数字转换为S
                str2=re.sub('[a-zA-Z]','M',str1)
                str3 = re.sub('[^a-zA-Z0-9&?=./]', 'G', str2)
                result.append(str3)
            return result


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
                                     ngram_range=(1, 1))  # token_pattern=r"(?u)\b\w+\b|[^a-zA-Z0-9]|" 保存全部字符
        vectorizer.fit(data)
        X_vec = vectorizer.transform(data)
        return X_vec

    def vector_five(data):
        feature = []
        for i in range(len(data)):
            s = data[i].split("?", 1)
            url_len = len(data[i])  # url长度
            path = s[0]
            path_str = re.search('(.*)://(.*?):(.*?)[^a-zA-Z0-9](.*)', path).group(4)
            path_len = len(path_str)
            if len(s) == 1:
                parameter_len = 0
                parameter_num = 0
                parameter_spe_num = 0
            if len(s) != 1:
                query = s[1]
                parameter_len = len(query)  # 参数长度
                parameters = query.split("&")
                parameter_num = len(parameters)  # 参数数目
                parameter_number_num = 0
                parameter_str_num = 0
                parameter_spe_num = 0
                par_val_sum = 0
                for parameter in parameters:
                    try:
                        # 采用 split("=", 1)是为了处理形如 open=123=234&file=op的参数
                        [par_name, par_val] = parameter.split("=", 1)
                    except ValueError as err:
                        # 处理形如 ?open 这样的参数
                        # print(err)
                        # print(data[i])
                        break
                    par_val_sum += len(par_val)
                    parameter_number_num += len(re.findall("\d", par_val))
                    parameter_str_num += len(re.findall(r"[a-zA-Z]", par_val))
                    parameter_spe_num += len(par_val) - len(re.findall("\d", par_val)) - len(
                        re.findall(r"[a-zA-Z]", par_val))
            feature.append([url_len, parameter_len, parameter_num, path_len, parameter_spe_num])
        return feature

class divide_data:
    def divicde(data1,data2,label):
        train_indices = np.random.choice(data1.shape[0], round(0.8 * data1.shape[0]),
                                         replace=False)
        test_indices = np.array(list(set(range(data1.shape[0])) - set(train_indices)))
        texts_train1 = data1[train_indices]
        texts_test1 = data1[test_indices]
        texts_train2 = np.array([y for iy, y in enumerate(data2) if iy in train_indices])
        texts_test2 = np.array([y for iy, y in enumerate(data2) if iy in test_indices])
        target_train = np.array([y for iy, y in enumerate(label) if iy in train_indices])
        target_test = np.array([y for iy, y in enumerate(label) if iy in test_indices])
        return texts_train1,texts_test1,texts_train2,texts_test2,target_train,target_test


class Bi_LSTM_model:
    def __init__(self):
        self.steps = 1
        self.learning_rate = 0.001
        self.batch_size = 10
        self.display_step = 50
        self.num_input = 42
        self.num_hidden = 20
        self.num_classes = 2
    def BiRNN(self,x, weights, biases):
        x = tf.unstack(x, self.steps, 1)
        lstm_fw_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


    def main(self,texts_train1,texts_test1,texts_train2,texts_test2,target_train,target_test):
        texts_train=hstack((texts_train1,texts_train2))
        texts_test=hstack((texts_test1,texts_test2))
        X = tf.placeholder("float", [None, self.steps,self.num_input])
        Y = tf.placeholder("float", [None, self.num_classes])
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([2 * self.num_hidden, self.num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        logits = Bi_LSTM_model().BiRNN(X, weights, biases)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        training_steps=texts_train.shape[0]//self.batch_size
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            for i in range(training_steps):
                start = (i * self.batch_size) % texts_train.shape[0]  #
                end = start + self.batch_size
                batch_len = texts_train[start:end].shape[0]
                batch_x = texts_train[start:end].reshape((batch_len, self.steps, self.num_input))
                batch_y = target_train[start:end].reshape((batch_len, self.num_classes))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if i % self.display_step == 0 or i == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X:  batch_x,
                                                                         Y: batch_y})
                    print("Step " + str(i) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
                i = i + 1

            print("Optimization Finished!")
            len1 = len(texts_test)
            test_data = texts_test[:len1]
            test_y = target_test[:len1]
            print("test accuracy %g" % sess.run(accuracy, feed_dict={X: test_data.reshape((len1, self.steps, self.num_input)), Y: test_y.reshape((len1, self.num_classes)),}))



if __name__ == '__main__':
    data1=readfile.pre_file('C:\\Users\\asus\\Desktop\\waf_test\\data\\cisc_normalTraffic_train.txt')
    data2=readfile.pre_file('C:\\Users\\asus\\Desktop\\waf_test\\data\\cisc_anomalousTraffic_test.txt')
    x_data1=np.array(feature.vector_five(data1))
    x_data2=np.array(feature.vector_five(data2))
    all_data1 = np.concatenate([x_data1, x_data2])
    all_data=readfile.dataconbine(data1,data2)
    all_data = lexstring.tokenstring(all_data)
    all_data=feature.feature_process_word(all_data).toarray()
    ylabel = np.array(readfile.label(data1, data2))
    texts_train1, texts_test1,texts_train2,texts_test2,\
    target_train, target_test=divide_data.divicde(all_data1,all_data,ylabel)


    Bi_LSTM_model().main(texts_train1,texts_test1,texts_train2,texts_test2,target_train,target_test)