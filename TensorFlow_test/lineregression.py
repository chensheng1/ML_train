import tensorflow as tf
from numpy.random import RandomState


batch_size=8
x=tf.placeholder(tf.float32,shape=[None,2],name="x-input")
y_=tf.placeholder(tf.float32,shape=[None,1],name="y-input")

w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w1)

loss_less=10
loss_more=1
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*10,(y_-y)*1))
train_step=tf.train.AdamOptimizer(0.01).minimize(loss)

ram=RandomState(1)
data_size=128
X=ram.rand(data_size,2)
Y=[[x1+x2+ram.rand()/10.0-0.05] for (x1,x2) in X]
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    step=5000
    for i in range(step):
        start=(i*batch_size)%data_size
        end=min(start+batch_size,data_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
    print(sess.run(w1))