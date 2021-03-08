
import os
import lenet_5
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARAZTION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

MODEL_SAVE_PATH="D:\\ML\\TensorFlow_test\\model\\"
MODEL_NAME="model.ckpt"

def train(mnist):
    x=tf.placeholder(tf.float32,[None,lenet_5.IMAGE_SIZE,lenet_5.IMAGE_SIZE,lenet_5.NUM_CHANNELS],name='x-input')
    y_=tf.placeholder(tf.float32,[None,lenet_5.OUTPUT_NODE],name='y-input')
    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y=lenet_5.inference(x,None,regularizer)
    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,     #基础学习率
        global_step,            #迭代轮数
        mnist.train.num_examples / BATCH_SIZE,  #过完所有训练数据需要的迭代次数
        LEARNING_RATE_DECAY,    #学习率衰减速率
        staircase=True)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                print("After %d training step(s),loss ontraining" "batch is %g." %(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist=input_data.read_data_sets("D:\ML\TensorFlow_test\data\MNIST_data",one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()