"""Run Inspect First Layer Weights of dilation/residual/DHS"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys

# from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

# import deepHaemWindow

# Basic model parameters as external flags -------------------------------------
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('model', '', 'Checkpoint of model file to be tested.')
# flags.DEFINE_string('graph', '', 'Defined graph of model.')
# # SAVE LOCATION
# flags.DEFINE_string('out_dir', 'save_data_out', 'Directory to store the weights.')
# in_model = FLAGS.model
# in_graph = FLAGS.graph


# for running interactive
exec(open("/home/ron/fusessh/scripts/machine_learning/epigenome_nets/deepHaem/deepHaemWindow.py").read())
in_model = '/home/ron/fusessh/machine_learning/deepHaem/training_runs_archive/runs_dhw_mouse_enc/training_dhw_mouse_enc_deepC_parameters_as_human_15052019/best_checkpoint-217584'
in_graph = '/home/ron/fusessh/machine_learning/deepHaem/training_runs_archive/runs_dhw_mouse_enc/training_dhw_mouse_enc_deepC_parameters_as_human_15052019/best_checkpoint-217584.meta'

outfile = '/home/ron/fusessh/machine_learning/deepHaem/training_runs_archive/runs_dhw_mouse_enc/training_dhw_mouse_enc_deepC_parameters_as_human_15052019/saved_conv_weights_mouse_deepc_arch.npy'
# outfile2 = '/home/ron/fusessh/machine_learning/deepHaem/training_runs_archive/training_dhw_mouse_enc_24102018/saved_conv_weights_mouse_enc_first_layer.npy'

# GLOBAL Options ---------------------------------------------------------------
BP_CONTEXT = 1000
NUM_CLASSES = 1022

# LOAD MODEL -------------------------------------------------------------------
# with tf.Session() as sess:
sess = tf.InteractiveSession()

# load meta graph and restore weights
saver = tf.train.import_meta_graph(in_graph)
saver.restore(sess, in_model)
graph = tf.get_default_graph()

# Inspect ----------------------------------------------------------------------
# See all trainable parameters
weights = tf.trainable_variables()
weights

# Get weights and Biases
weights_hidden1 = sess.run('Convolutional_stack/conv_layer1/weights:0')
biases_hidden1 = sess.run('Convolutional_stack/conv_layer1/Variable:0')
weights_hidden2 = sess.run('Convolutional_stack/conv_layer2/weights:0')
biases_hidden2 = sess.run('Convolutional_stack/conv_layer2/Variable:0')
weights_hidden3 = sess.run('Convolutional_stack/conv_layer3/weights:0')
biases_hidden3 = sess.run('Convolutional_stack/conv_layer3/Variable:0')
weights_hidden4 = sess.run('Convolutional_stack/conv_layer4/weights:0')
biases_hidden4 = sess.run('Convolutional_stack/conv_layer4/Variable:0')
weights_hidden5 = sess.run('Convolutional_stack/conv_layer5/weights:0')
biases_hidden5 = sess.run('Convolutional_stack/conv_layer5/Variable:0')

# save weigths as numpy arrays
np.savez(outfile,
    weights_hidden1, biases_hidden1,
    weights_hidden2, biases_hidden2,
    weights_hidden3, biases_hidden3,
    weights_hidden4, biases_hidden4,
    weights_hidden5, biases_hidden5)

npzfile = np.load(outfile + ".npz")
npzfile.files

type(npzfile['arr_0'])

a = npzfile['arr_0']
a.shape

# np.save(outfile2, weights_hidden1)

# print as txt
# print('Fetching and saving Kernel/Filter Weights ...')
# for w in range(weights_shape[2]):
#     # get and save weights
#     # weights_shaped = weights[:,:,w].reshape(FLAGS.kernel1_width,4)
#     kernel = weights[:,:,w]
#     np.savetxt(FLAGS.out_dir + '/weights_filter' + str(w) + '.txt', kernel, delimiter="\t")
#
# print('Done!')
