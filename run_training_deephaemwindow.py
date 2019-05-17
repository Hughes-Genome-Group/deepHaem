"""Trains and Evaluates deepHaemWindow playground network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys

# from six.moves import xrange  # pylint: disable=redefined-builtin

import h5py
import numpy as np
import tensorflow as tf

import deepHaemWindow

# Basic model parameters as external flags -------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_file', '', 'Iput Training and Test Sequences and Labels in hdf5 Format.\
Expects "training_seqs", "training_labels", "test_seqs", "test_labels" labeled data.')
# TRAINIGS SETTINGS
flags.DEFINE_integer('max_epoch', 2, 'Number of epoch through train data to run trainer.')
flags.DEFINE_float('keep_prob_inner', 0.8, 'Keep probability for dropout')
flags.DEFINE_float('keep_prob_outer', 0.8, 'Keep probability for dropout')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_float('l2_strength', 0.0001, 'L2 regularization strength.')
flags.DEFINE_boolean('shuffle', True, 'If to shuffle the trainset at the start of each epoch.')
# CONVOLUTIONAL STACK OPTIONS
flags.DEFINE_integer('conv_layers', 3, 'Number of convolutional layers.')
flags.DEFINE_string('hidden_units_scheme', '300,600,900', 'Comma seperated hidden units scheme. Must have length of number of conv layers specified!')
flags.DEFINE_string('kernel_width_scheme', '20,8,8', 'Comma seperated kernel width scheme. Must have length of number of conv layers specified!')
flags.DEFINE_string('max_pool_scheme', '5,5,5', 'Comma seperated max pool scheme. Must have length of number of conv layers specified!')
# FULLY CONNECTED Options
flags.DEFINE_boolean('upstream_connected', False, 'Add an upstream fully connecte layer [True/False].')
flags.DEFINE_integer('upstream_connections', 100, 'Connections for the upstream fully connected layer.')
# OPTIMIZER (ADAM) options
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('beta1', 0.9, 'ADAM: beta1.')
flags.DEFINE_float('beta2', 0.999, 'ADAM: beta2.')
flags.DEFINE_float('epsilon', 1e-08, 'ADAM: epsilon.')
# TRAIN LOCATION
flags.DEFINE_string('train_dir', 'training_run_data', 'Directory to put the training data.')
flags.DEFINE_boolean('reload_model', False, 'If to reload a checkpoint/model file.')
flags.DEFINE_string('model', None, 'Path to checkpoint/model file.')
# Options for preseeding with pretreined weights
flags.DEFINE_boolean('seed_weights', False, 'Select if to pre seed weights with numpy array stored weights.')
flags.DEFINE_string('seed_scheme', '0,0,0', 'Specify which layers are preseeded with the weights provided. [format: 1,1,0]')
flags.DEFINE_string('seed_file', None, 'Path to saved numpy file with saved weights. Weight and bias dimensions must match with the ones specified as hyper params for this run!')
# Machine options
flags.DEFINE_integer('gpu', 0, 'Select a single available GPU and mask the rest. Default 0.')
# Log Options
flags.DEFINE_integer('report_every', 100, 'Set interval of batch steps o when to report raining loss and log progress, losses and weights etc.')
# flag if to train with boolean values stored (labels and sequence)
flags.DEFINE_string('stored_dtype', "int32", 'Indicate what data format sequence and labels where stored in. ["bool", "int8", "int32"]')
flags.DEFINE_integer('seed', 1234, 'Random seed for tensorflow (graph level).')
# GLOBAL Options ---------------------------------------------------------------
flags.DEFINE_integer('bp_context', 1000, 'Number of classes to classify. Default 600.')
flags.DEFINE_integer('num_classes', 936, 'Number of classes to classify. Default 182.')
BP_CONTEXT = FLAGS.bp_context
NUM_CLASSES = FLAGS.num_classes

# SET RANDOM SEED --------------------------------------------------------------
tf.set_random_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)  # use same seed for numpy --> for shuffeling

# PREP CONVOLUTIONAL SCHEME ----------------------------------------------------
hidden_units_scheme = [x.strip() for x in FLAGS.hidden_units_scheme.split(',')]
hidden_units_scheme = list(map(int, hidden_units_scheme))
kernel_width_scheme = [x.strip() for x in FLAGS.kernel_width_scheme.split(',')]
kernel_width_scheme = list(map(int, kernel_width_scheme))
max_pool_scheme = [x.strip() for x in FLAGS.max_pool_scheme.split(',')]
max_pool_scheme = list(map(int, max_pool_scheme))
seed_scheme = [x.strip() for x in FLAGS.seed_scheme.split(',')]
seed_scheme = list(map(int, seed_scheme))

# Assert length of schemes all match specified number of conv layers
if len(hidden_units_scheme) != FLAGS.conv_layers:
    print("Hidden Units Scheme does not have the number of entries expected from 'conv_layers' ...")
    sys.exit()
if len(kernel_width_scheme) != FLAGS.conv_layers:
    print("Hidden Width Scheme does not have the number of entries expected from 'conv_layers' ...")
    sys.exit()
if len(max_pool_scheme) != FLAGS.conv_layers:
    print("Max Pool Scheme does not have the number of entries expected from 'conv_layers' ...")
    sys.exit()
if FLAGS.seed_weights and len(seed_scheme) != FLAGS.conv_layers:
    print("Seed Scheme does not have the number of entries expected from 'conv_layers' ...")
    sys.exit()


# HELPER FUNCTIONS -------------------------------------------------------------
def placeholder_inputs(batch_size, dtype):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    seqs_placeholder: Sequences (hot coded) placeholder.
    labels_placeholder: Labels placeholder.
  """
  if dtype == "bool":
      seqs_placeholder = tf.placeholder(tf.bool, [None, BP_CONTEXT, 4], name='seqs')
      labels_placeholder = tf.placeholder(tf.bool, shape=[None, NUM_CLASSES], name='labels')
  elif dtype == "uint8":
      seqs_placeholder = tf.placeholder(tf.uint8, [None, BP_CONTEXT, 4], name='seqs')
      labels_placeholder = tf.placeholder(tf.uint8, shape=[None, NUM_CLASSES], name='labels')
  else:
      seqs_placeholder = tf.placeholder(tf.int32, [None, BP_CONTEXT, 4], name='seqs')
      labels_placeholder = tf.placeholder(tf.int32, shape=[None, NUM_CLASSES], name='labels')

  # Note that the shapes of the placeholders match the shapes
  return seqs_placeholder, labels_placeholder

def do_eval(sess,
            eval_correct,
            eval_loss,
            seqs_placeholder,
            labels_placeholder,
            seqs_test,
            labels_test,
            keep_prob_inner_placeholder,
            keep_prob_outer_placeholder
            ):
  """Runs one evaluation against the full epoch of test data.
  Return test accuracy and mean test loss per batch.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    seqs_placeholder: The sequences placeholder.
    labels_placeholder: The labels placeholder.
    keep_prob_pl: placeholder for the keep probability
    lines: Opend lines object of training data file
    cases: number of lines/cases in input file
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  test_loss = 0
  cases = labels_test.shape[0]
  steps_per_epoch = cases // FLAGS.batch_size

  # for each step
  for step in range(steps_per_epoch):
    test_batch_start = step*FLAGS.batch_size
    test_batch_end = step*FLAGS.batch_size+FLAGS.batch_size
    test_batch_range=range(test_batch_start, test_batch_end)
    feed_dict = {
          seqs_placeholder: seqs_test[test_batch_range],
          labels_placeholder: labels_test[test_batch_range],
          keep_prob_inner_placeholder: 1.0,
          keep_prob_outer_placeholder: 1.0
          }
    tmp_true_count, tmp_test_loss = sess.run([eval_correct, eval_loss], feed_dict=feed_dict)
    true_count += tmp_true_count
    test_loss += tmp_test_loss

  accuracy = true_count / steps_per_epoch
  test_loss = test_loss / steps_per_epoch

  print('Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f Test Loss: %0.04f' %
    (cases, true_count, accuracy, test_loss))

  return accuracy, test_loss  # return precision

def run_training():

  """Train endpoolBasset for a number of steps."""

  # Get the sets of seqs and labels from hdf5 formated file
  h5f = h5py.File(FLAGS.train_file, 'r')
  training_seqs = h5f['training_seqs']
  training_labels = h5f['training_labels']
  test_seqs = h5f['test_seqs']
  test_labels = h5f['test_labels']

  print('training_data')
  print(np.shape(training_seqs))
  training_cases = np.shape(training_seqs)[0]

  print('test_data')
  print(test_seqs.shape)
  test_cases = np.shape(training_seqs)[0]

  # make indices for shuffeling the data
  training_index = np.asarray(range(training_cases))

  # Write configuration/parameters to specified train dir
  # WRITE HYPER PARAMETERS and DATA PROPERTIES TO RUN LOG FILE
  current_time = time.localtime()
  timestamp = str(current_time[0]) + str(current_time[1]) + str(current_time[2]) + str(current_time[3]) + str(current_time[4])

  if not os.path.exists(FLAGS.train_dir):  # make train dir
      os.makedirs(FLAGS.train_dir)

  param_log_file = open(FLAGS.train_dir + '/hyperparameters_' + timestamp + '.log', 'w')
  param_log_file.write("# Hyperparameters for dilationDHS run at: " + str(current_time))
  param_log_file.write("\n\nInput: " + str(BP_CONTEXT) + " bp")
  param_log_file.write("\n\nArchitechture:")
  for i in range(FLAGS.conv_layers):
    j = i + 1
    param_log_file.write("\nHidden Units Layer %s: %s" % (j, hidden_units_scheme[i]))
    param_log_file.write("\nKernel Width Layer %s: %s" % (j, kernel_width_scheme[i]))
    param_log_file.write("\nMax Pool Layer %s: %s" % (j, max_pool_scheme[i]))
  if FLAGS.upstream_connected:
    param_log_file.write("\nUpstream FC Connections: %s" % FLAGS.upstream_connections)
  if FLAGS.seed_weights:
      param_log_file.write("\nPre-seeding with saved weights: " + FLAGS.seed_file)
  param_log_file.write("\n\nTraining Parameters:")
  param_log_file.write("\nBatch size: " + str(FLAGS.batch_size))
  param_log_file.write("\nDropout Keep Probability Inner: " + str(FLAGS.keep_prob_inner))
  param_log_file.write("\nDropout Keep Probability Outer: " + str(FLAGS.keep_prob_outer))
  param_log_file.write("\nLearning Rate (Intital): " + str(FLAGS.learning_rate))
  param_log_file.write("\n(ADAM) Beta 1: " + str(FLAGS.beta1))
  param_log_file.write("\n(ADAM) Beta 2: " + str(FLAGS.beta2))
  param_log_file.write("\n(ADAM) Epsilon: " + str(FLAGS.epsilon))
  param_log_file.write("\nMaximum Epoch Number: " + str(FLAGS.max_epoch) + "\n")
  param_log_file.write("\nL2 Regularizer Strength: " + str(FLAGS.l2_strength) + "\n")
  param_log_file.write("\nShuffle Training Set after every Epoch: " + str(FLAGS.shuffle) + "\n\n")

  total_trainable = 0
  for i in range(FLAGS.conv_layers):
    tmp_units = 4
    total_trainable += hidden_units_scheme[i] * kernel_width_scheme[i] * tmp_units
    tmp_units = hidden_units_scheme[i]
  if FLAGS.upstream_connected:
      total_trainable += hidden_units_scheme[i] * FLAGS.upstream_connections
      total_trainable += FLAGS.upstream_connections * NUM_CLASSES
  else:
      total_trainable += hidden_units_scheme[i] * NUM_CLASSES

  param_log_file.write("Total Trainable parameters:\t %s\n" % total_trainable)
  param_log_file.write("____________________________________________________\n")
  param_log_file.close()


  # load seed weights if specified
  if FLAGS.seed_weights:
      print("Loading saved weights ...")
      seed_weights_list = np.load(FLAGS.seed_file)
  else:
      seed_weights_list = ""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
  # with tf.device("/cpu:0"):

    # Generate placeholders for the seqs and labels (and dropout prob).
    seqs_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size, FLAGS.stored_dtype)
    keep_prob_inner_placeholder = tf.placeholder(tf.float32, name='keep_prob_inner')
    keep_prob_outer_placeholder = tf.placeholder(tf.float32, name='keep_prob_outer')

    # # input shape test -------------------------------------------------------
    # print('Shapes:')
    # l = training_labels[0]
    # s = training_seqs[0]
    # # # print(l)
    # # # print(s)
    # # test a feed dict
    # fd = {
    #       seqs_placeholder: training_labels[0:(FLAGS.batch_size - 1)],
    #       labels_placeholder: training_seqs[0:(FLAGS.batch_size - 1)]
    #   }
    # print('Test feed Dict:')
    # print(fd)
    # sys.exit()

    # Building the Graph -------------------------------------------------------
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Ops to calc logits
    logits = deepHaemWindow.inference(
        seqs_placeholder,
        FLAGS.conv_layers,
        hidden_units_scheme,
        kernel_width_scheme,
        max_pool_scheme,
        FLAGS.upstream_connected,
        FLAGS.upstream_connections,
        NUM_CLASSES,
        FLAGS.batch_size,
        keep_prob_inner_placeholder,
        keep_prob_outer_placeholder,
        FLAGS.seed_weights,
        seed_scheme,
        seed_weights_list
        )
    tf.add_to_collection("logits", logits)

    # Add to the Graph the Ops for loss calculation.
    loss = deepHaemWindow.loss(logits, labels_placeholder, FLAGS.l2_strength, FLAGS.batch_size)
    loss_test = deepHaemWindow.loss_test(logits, labels_placeholder, FLAGS.batch_size)
    tf.add_to_collection("loss_test", loss_test)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = deepHaemWindow.training(
        loss,
        FLAGS.learning_rate,
        FLAGS.beta1,
        FLAGS.beta2,
        FLAGS.epsilon,
        global_step)
    tf.add_to_collection("train_op", train_op)

    # Add the Ops to compare the logits to the labels during evaluation.
    eval_op = deepHaemWindow.evaluation(logits, labels_placeholder)
    tf.add_to_collection("eval_op", eval_op)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=5)

    # init op
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    config = tf.ConfigProto();
    config.gpu_options.visible_device_list = str(FLAGS.gpu)
    config.allow_soft_placement = True
    sess = tf.Session(config = config)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

    # reload model if specified or initialize ----------------------------------
    if FLAGS.reload_model == True:
        print("Restoring previous model checkpoint ....")
        saver.restore(sess, FLAGS.model)
        # reload global step
        total_step = tf.train.global_step(sess, global_step)
    else:
        sess.run(init)
        total_step = 0

    # Start the TRAININGloop ===================================================
    epoch = 0
    step = 0
    best_loss = 500.0

    if FLAGS.shuffle:
        print("Shuffeling training set ...")
        # get a randomized test index to learn with independently shuffled sets for comparing the loss !!!!
        np.random.shuffle(training_index)

    # for step in xrange(FLAGS.max_steps):
    while epoch < FLAGS.max_epoch:
        step += 1
        total_step += 1
        start_time = time.time()
        # Fill a feed dictionary with the respective set of seqs and labels
        batch_start = (step-1)*FLAGS.batch_size
        batch_end = (step-1)*FLAGS.batch_size+FLAGS.batch_size
        batch_range=range(batch_start, batch_end)
        # convert to shuffled traiing index
        batch_range = training_index[batch_range]
        batch_range = np.sort(batch_range)
        batch_range = batch_range.tolist()
        #feed
        feed_dict = {
              seqs_placeholder: training_seqs[batch_range],
              labels_placeholder: training_labels[batch_range],
              keep_prob_inner_placeholder: FLAGS.keep_prob_inner,
              keep_prob_outer_placeholder: FLAGS.keep_prob_outer
              }

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op. To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

        # recognize end of epoch, reset step, Get Test Accuracy and Save =======
        if (step + 1) * FLAGS.batch_size > training_cases:
            epoch += 1
            step = 0  # reset step
            print("Epoch %s done:" % epoch)
            print('Test Data Accuracy Eval:')
            test_accuracy, test_loss = do_eval(sess,
                    eval_op,
                    loss_test,
                    seqs_placeholder,
                    labels_placeholder,
                    test_seqs,
                    test_labels,
                    keep_prob_inner_placeholder,
                    keep_prob_outer_placeholder
                    )
            test_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=test_loss)])
            test_accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag="test_accuracy", simple_value=test_accuracy)])
            summary_writer.add_summary(test_loss_summary, total_step)
            summary_writer.add_summary(test_accuracy_summary, total_step)
            # Save Checkpoint if test loss is smaller then the previous best ===
            if best_loss > test_loss:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'best_checkpoint')
                saver.save(sess, checkpoint_file, global_step=total_step)
                best_loss = test_loss
            if FLAGS.shuffle == True:
                print("Shuffeling training set ...")
                np.random.shuffle(training_index)

        # Write the summaries and print and overview every X steps =============
        if step % FLAGS.report_every == 0:
            duration = time.time() - start_time # get step time
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, total_step)
            summary_writer.flush()

  h5f.close()

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
