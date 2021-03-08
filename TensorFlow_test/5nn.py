import tensorflow as tf
from numpy.random import RandomState


def get_weight(shape,lamdba):
    var=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamdba)(var))
    return var

x=tf.placeholder(tf.float32,shape=[None,2],name="x-input")
y_=tf.placeholder(tf.float32,shape=[None,1],name="y-input")

batch_size=8
layer_dimension=[2,10,10,10,1]
n_layer=len(layer_dimension)

cur_layer=x
in_dimension=layer_dimension[0]

for i in range(1,n_layer):
    out_dimension=layer_dimension[i]
    weight=get_weight([in_dimension,out_dimension],0.001)
    bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))

    cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)

    in_dimension=layer_dimension[i]

mse_loss=tf.reduce_mean(tf.square(y_-cur_layer))
tf.add_to_collection('losses',mse_loss)

loss=tf.add_n(tf.get_collection('losses'))

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
        print(sess.run(loss,feed_dict={x:X[start:end],y_:Y[start:end]}))
