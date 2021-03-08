#coding=utf-8
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from jieba import cut

# 数据
positive_texts = [
    "我今天很高兴",
    "我很开心",
    "他很高兴",
    "他很开心"
]
negative_texts = [
    "我不高兴",
    "我不开心",
    "他今天不高兴",
    "他不开心"
]

label_name_dict = {
    0: "正面情感",
    1: "负面情感"
}

def chinese_tokenizer(docs):
    for doc in docs:
        yield list(cut(doc))

embedding_size=60
num_classes=2

all_texts=positive_texts+negative_texts
labels=[0]*len(positive_texts)+[1]*len(negative_texts)

max_document_length=4
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,0,tokenizer_fn=chinese_tokenizer)
datas=np.array(list(vocab_processor.fit_transform(all_texts)))
print(datas)
vocab_size = len(vocab_processor.vocabulary_)

# 容器，存放输入输出
datas_placeholder = tf.placeholder(tf.int32, [None, max_document_length])
labels_placeholder = tf.placeholder(tf.int32, [None])

# 词向量表
embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size], initializer=tf.truncated_normal_initializer)

# 将词索引号转换为词向量[None, max_document_length] => [None, max_document_length, embedding_size]
embedded = tf.nn.embedding_lookup(embeddings, datas_placeholder)

# 转换为LSTM的输入格式，要求是数组，数组的每个元素代表某个时间戳一个Batch的数据
rnn_input = tf.unstack(embedded, max_document_length, axis=1)

# 定义LSTM
lstm_cell = BasicLSTMCell(20, forget_bias=1.0)
rnn_outputs, rnn_states = static_rnn(lstm_cell, rnn_input, dtype=tf.float32)

#利用LSTM最后的输出进行预测
logits = tf.layers.dense(rnn_outputs[-1], num_classes)

predicted_labels = tf.argmax(logits, axis=1)

# 定义损失和优化器
losses= tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)

mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(mean_loss)

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 定义要填充的数据
    feed_dict = {
        datas_placeholder: datas,
        labels_placeholder: labels
    }

    print("开始训练")
    for step in range(100):
        _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=feed_dict)

        if step % 10 == 0:
            print("step = {}\tmean loss = {}".format(step, mean_loss_val))
    print("训练结束，进行预测")
    predicted_labels_val = sess.run(predicted_labels, feed_dict=feed_dict)
    for i, text in enumerate(all_texts):
        label = predicted_labels_val[i]
        label_name = label_name_dict[label]
        print("{} => {}".format(text, label_name))
