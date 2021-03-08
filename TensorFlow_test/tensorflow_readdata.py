import tensorflow as tf
import os

directory = str(os.getcwd())
conf_path1 = os.path.abspath(os.path.join(directory, os.path.pardir))
conf_path = os.path.join(conf_path1, 'TensorFlow_test\data')
filename=os.listdir(conf_path)
label=[]
for name in filename:
    if name.find("letters_source")>=0 :
        abnormal=[os.path.join(conf_path,name)]
        label.append(0)
    if name.find("letters_target2")>=0:
        normal=[os.path.join(conf_path,name)]
        label.append(1)
filename_queue=tf.train.string_input_producer(normal,shuffle=False)
reader=tf.TextLineReader()
key=reader.read(filename_queue)
example=tf.decode_csv(key,record_defaults=[['null']])
example_batch = tf.train.batch([example], batch_size=1, capacity=200, num_threads=2)
with tf.Session() as sess:
    coord=tf.train .Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    for i in range(1000):
        e_val=sess.run(example)
        print(e_val)
    coord.request_stop()
    coord.join(threads)