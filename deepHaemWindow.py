"""Tensorflow implementation to reproduce Basset with same published ...
hyperparameters and architechture:
3 Convolutional layers with batch norm and max pooling:
    300 x 19 > BN > ReLU > MaxP 3w
    200 x 11 > BN > ReLU > MaxP 4w
    200 x 7 > BN > ReLU > MaxP 4w
2 Linear/fully connected with 1000 units > RELU and 0.3p Dropout
Final fully connected with 164 units into sigmoid
------------------------------------------------------------------

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import tensorflow as tf

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def preload_variable(name, data):
    '''Create variable from numpy data.'''
    variable = tf.Variable(data, name=name)
    return variable

def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    # variable = tf.Variable(tf.truncated_normal(shape), name=name)
    return variable

def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)

# Define Convolutional Layer
def convolutional_layer(name,
                  input_batch,
                  units,
                  kernel_width,
                  pool_width,
                  keep_prob,
                  to_seed,
                  seed_weights,
                  seed_biases,
                  to_batch_norm=False):
    '''Create a convolutional layer:
        INPUT:
        name: must be unique for graph purpose
        input_batch: 3D input tensor batch, length, channels/width
        units: hidden units (# kernels)
        kernel_width: width of the convolutional kernels/filters
        pool_width: (max) pool width
        keep_prob: dropout keep probability
        to_seed: True / False if to pre seed weights and biases in this layer
        seed_weights: numpy array of seed weights
        seed_biases: numpy array of seed biases
        to_batch_norm: True/False select of to perform batch norm at every layer
        RETURNS:
        3D tensor batch, length/pool_width, channels width'''
    with tf.name_scope(name):
        # get shapes
        channels = input_batch.get_shape().as_list()[2]
        # create variables
        if to_seed:
            weights = preload_variable("weights", seed_weights)
            biases = preload_variable("biases", seed_biases)
        else:
            weights = create_variable('weights', [kernel_width, channels, units])
            biases = create_bias_variable('biases', [units])
        # define convolutional steps
        conv = tf.add(tf.nn.conv1d(input_batch, weights, stride=1, padding='SAME'), biases)
        conv = tf.nn.relu(conv)
        # make summary histograms of weights
        tf.summary.histogram(name + '_conv_weights', weights)
        tf.summary.histogram(name + '_conv_biases', biases)
        # activation summary
        _activation_summary(conv)
        # Max Pool
        conv = tf.layers.max_pooling1d(conv, pool_width, strides=pool_width, padding='same', name=str(name+'max_pool'))
        # Dropout
        out = tf.nn.dropout(conv, keep_prob)
        # # batch norm
        if to_batch_norm == True:
            out = tf.layers.batch_normalization(out)
        return out

# INFERENCE
def inference(seqs,
              conv_layers,
              hidden_units_scheme,
              kernel_width_scheme,
              max_pool_scheme,
              upstream_connected,
              upstream_connections,
              num_classes,
              batch_size,
              keep_prob_inner,
              keep_prob_outer,
              seed_weights,
              seed_scheme,
              seed_weights_list
              ):
    """INFERENCE

    Args:
    seqs: Sequence placeholder.
    hidden1_units: Size of the first hidden layer.
    kernel1_width: width of the kernel (*4 rows to get the kernel)
    max_pool1_width: width to apply for max pooling layer 1

    Returns:
    softmax_linear: Output tensor with the computed logits.
    """
    print('seqs shape')
    print(seqs.get_shape().as_list())

    current_layer = tf.cast(seqs, tf.float32)

    # Convolutional Layer Stack ================================================
    # run an inital dilated layer with dilation 1 to map to the dilational unit output
    with tf.name_scope('Convolutional_stack'):
        for i in range(conv_layers):
            j = i + 1
            k = i * 2
            if seed_weights and seed_scheme[i] == 1:
                weights_load_string = 'arr_' + str(k)
                biases_load_string = 'arr_' + str(k+1)
                print('Pre-seeding Layer: ' + str(j))
                current_layer = convolutional_layer(
                    'conv_layer{}'.format(j),
                    current_layer,
                    hidden_units_scheme[i],
                    kernel_width_scheme[i],
                    max_pool_scheme[i],
                    keep_prob_inner,
                    True,
                    seed_weights_list[weights_load_string],
                    seed_weights_list[biases_load_string],
                    to_batch_norm=False)
            else:
                current_layer = convolutional_layer(
                    'conv_layer{}'.format(j),
                    current_layer,
                    hidden_units_scheme[i],
                    kernel_width_scheme[i],
                    max_pool_scheme[i],
                    keep_prob_inner,
                    False,
                    "dummy",
                    "dummy",
                    to_batch_norm=False)
            print('Conv %s shape' % j)
            print(current_layer.get_shape().as_list())

    # Reshape ==================================================================
    with tf.name_scope('reshape'):
        fully_connected_width = (current_layer.get_shape().as_list()[1]) * hidden_units_scheme[len(hidden_units_scheme)-1]
        print("fully_connected_width")
        print(fully_connected_width)
        current_layer = tf.reshape(current_layer, [-1, fully_connected_width])
        print('fully connection reshaped')
        print(current_layer.get_shape().as_list())

    # Optional Upstream single fully connected (2D) ============================
    if upstream_connected == True:
        with tf.name_scope('upstream_fully_connected'):
            weights = create_variable('weights', [fully_connected_width, upstream_connections])
            biases = create_bias_variable('biases', [upstream_connections])
            current_layer = tf.nn.relu(tf.add(tf.matmul(current_layer, weights), biases))
            _activation_summary(current_layer)
            tf.summary.histogram('upstream_linear_weights', weights)
            current_layer = tf.nn.dropout(current_layer, keep_prob_outer)
            print('Upstream FC Layer shape')
            print(current_layer.get_shape().as_list())

    # Final full connection(s) into logits =====================================
    with tf.name_scope('signmoid_linear'):
        if upstream_connected:
            weights = create_variable('weights', [upstream_connections, num_classes])
        else:
            weights = create_variable('weights', [fully_connected_width, num_classes])
        biases = create_bias_variable('biases', [num_classes])
        logits = tf.add(tf.matmul(current_layer, weights), biases)
        print('Logits shape')
        print(logits.get_shape().as_list())
        _activation_summary(logits)
        tf.summary.histogram('sigmoid_weights', weights)

    return logits

def loss(logits, labels, l2_regularization_strength, batch_size):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size, NUM_CLASSES].

  Returns:
    loss: Loss tensor of type float.
  """
  with tf.name_scope('Loss'):
      labels = tf.to_float(labels)
      # modified cross entropy to explicit mathematical formula of sigmoid cross entropy loss
      sigmoid = tf.nn.sigmoid(logits)
      cross_entropy = -tf.reduce_sum(((labels*tf.log(sigmoid + 1e-9))+((1-labels)*tf.log(1-sigmoid+1e-9))), name='cross_entropy')
      # mean over batch size but sum over all classes
      cross_entropy_loss = tf.reduce_sum(cross_entropy, name='cross_entropy_sum_mean_of_batches')/batch_size
      # add summarizers if training case (loss for test is reported per epoch)
      tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

      # add regularizer:
      if l2_regularization_strength == 0:
          return cross_entropy_loss
      else:
          # L2 regularization for all trainable parameters
          l2_loss = tf.add_n([tf.nn.l2_loss(v)
                              for v in tf.trainable_variables()
                              if not('bias' in v.name)])
          # Add the regularization term to the loss
          total_loss = (cross_entropy_loss + l2_regularization_strength * l2_loss)
          # add summarizers
          tf.summary.scalar('l2_loss', l2_loss)
          tf.summary.scalar('total_loss', total_loss)

          return total_loss

def loss_test(logits, labels, batch_size):
  """Calculates the loss from the logits and the labels for ttraining case without summaries.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size, NUM_CLASSES].
  Returns:
    loss: Loss tensor of type float.
  """
  with tf.name_scope('Test_Loss'):
      labels = tf.to_float(labels)
      sigmoid = tf.nn.sigmoid(logits)
      cross_entropy = -tf.reduce_sum(((labels*tf.log(sigmoid + 1e-9))+((1-labels)*tf.log(1-sigmoid+1e-9))), name='cross_entropy')
      test_loss = tf.reduce_sum(cross_entropy, name='test_cross_entropy_mean_over_batch_size')/batch_size
      return test_loss

def training(loss, learning_rate, beta_1, beta_2, epsilon, global_step):
  """Sets up the training Operations.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # with Learning Rate decay
  # learning_rate = tf.train.exponential_decay(learning_rate, global_step, learning_rate_decay_steps, 0.96)
  optimizer = tf.train.AdamOptimizer(
    learning_rate = learning_rate,
    beta1 = beta_1,
    beta2 = beta_2,
    epsilon = epsilon)
  trainables = tf.trainable_variables()
  train_op = optimizer.minimize(loss, var_list=trainables, global_step=global_step)

  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  # correct = tf.nn.in_top_k(logits, tf.argmax(labels, 1), 1)
  # return tf.reduce_sum(tf.cast(correct, tf.int32))
  labels = tf.to_float(labels)
  correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
  mean_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return mean_correct
