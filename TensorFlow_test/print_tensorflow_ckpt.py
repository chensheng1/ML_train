#-*- coding: UTF-8 -*-
# Author:chensheng
#create:2020-06-19
#describe：打印模型结构参数

import tensorflow as tf


ckpt_path="D:\\ML\\TensorFlow_test\\model\\model.ckpt-25001"
saver = tf.train.import_meta_graph(ckpt_path+'.meta',clear_devices=True)
graph = tf.get_default_graph()
with tf.Session( graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,ckpt_path)


def read_graph_from_ckpt(ckpt_path, input_names, output_name):
    saver = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_path)
        output_tf = graph.get_tensor_by_name(output_name)
        pb_graph = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [output_tf.op.name])

    with tf.Graph().as_default() as g:
        tf.import_graph_def(pb_graph, name='')
    with tf.Session(graph=g) as sess:
        OPS = get_ops_from_pb(g, input_names, output_name)
    return OPS

def get_ops_from_pb(graph,input_names,output_name,save_ori_network=True):
    if save_ori_network:
        with open('ori_network.txt','w+') as w:
            OPS=graph.get_operations()
            for op in OPS:
                txt = str([v.name for v in op.inputs])+'---->'+op.type+'--->'+str([v.name for v in op.outputs])
                w.write(txt+'\n')
    inputs_tf = [graph.get_tensor_by_name(input_name) for input_name in input_names]
    output_tf =graph.get_tensor_by_name(output_name)
    OPS =get_ops_from_inputs_outputs(graph, inputs_tf,[output_tf] )
    with open('network.txt','w+') as w:
        for op in OPS:
            txt = str([v.name for v in op.inputs])+'---->'+op.type+'--->'+str([v.name for v in op.outputs])
            w.write(txt+'\n')
    OPS = sort_ops(OPS)
    OPS = merge_layers(OPS)
    return OPS