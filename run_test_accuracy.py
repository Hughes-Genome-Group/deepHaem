"""Run Test prediction to calculate accuracy, AUC and ROC curve of the the
supplied model vs. base line from class frequencies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin

import h5py
import numpy as np
import tensorflow as tf
# import pandas as pd

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import interp

# import endpoolDeepHaemElement

# Basic model parameters as external flags -------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dlmodel', 'deepHaemWindow', 'Specifcy the DL model file to use e.g. <endpoolDeepHaemElement>.py')
flags.DEFINE_string('test_on', 'test', 'Either \"test\" or \"valid\" -> select if to test accuracy on the test or validation set')
flags.DEFINE_string('test_file', '', 'Input Training and Test Sequences and Labels in hdf5 Format.\
"test_seqs", "test_labels" labeled data. Or the validation file labels validation_seqs etc. ...')
flags.DEFINE_string('model', '', 'Checkpoint of model file to be tested.')
flags.DEFINE_string('graph', '', 'Defined graph of model.')
# RUN SETTINGS
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('test_dir', 'test_data_out', 'Directory to store the test data output')
flags.DEFINE_string('name_tag', 'eval', 'Nametag to add to filenames')
flags.DEFINE_string('slize', 'all', 'Comma separated list of start and end position of columns to slice out (0) indexed. Will use all if unspecified.')
flags.DEFINE_integer('only', 0, 'Set number of first lines to only use for testing (if 0 will do all).')
# WHAT TO DO
flags.DEFINE_string('savetxt', 'False', 'Select if to store score and labels as txt files for parsing.')
flags.DEFINE_string('roc', 'False', 'Calculate and Plot ROcurves per classifier.')
flags.DEFINE_string('prc', 'False', 'Calculate and Plot PRcurves per classifier.')
# Dataset Options
flags.DEFINE_integer('bp_context', 1000, 'Basepairs per feature.')
flags.DEFINE_integer('num_classes', 919, 'Number of classes.')
# machine options
flags.DEFINE_string('run_on', 'gpu', 'Select where to run on (cpu or gpu)')
flags.DEFINE_integer('gpu', 0, 'Select a single available GPU and mask the rest. Default 0.')
# add option to print prc/roc aucs
flags.DEFINE_string('roc_auc', 'False', 'Define if to print ROC AUC values (False/True)')
flags.DEFINE_string('prc_auc', 'False', 'Define if to print PRC AUC values (False/True)')

# some arg parsing and execution
# import flexible dl model arch
dlmodel = __import__(FLAGS.dlmodel)

# prepare for column slizes if specified
if FLAGS.slize != 'all':
    slize_scheme = [x.strip() for x in FLAGS.slize.split(',')]
    slize_scheme = list(map(int, slize_scheme))

# HELPER FUNCTIONS -------------------------------------------------------------
def do_eval(sess,
            eval_correct,
            eval_loss,
            get_sigmoid,
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
  print(cases)
  steps_per_epoch = cases // FLAGS.batch_size
  # make an empty array for the sigmoid test scores
  test_scores = np.zeros((steps_per_epoch*FLAGS.batch_size,FLAGS.num_classes))

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
    tmp_true_count, tmp_test_loss, tmp_sigmoid = sess.run([eval_correct, eval_loss, get_sigmoid], feed_dict=feed_dict)
    true_count += tmp_true_count
    test_loss += tmp_test_loss
    # arragne sigmoids as test scores and concatinate
    tmp_sigmoid = np.asarray(tmp_sigmoid)
    tmp_sigmoid = np.squeeze(tmp_sigmoid)
    # add to the empty test scores matrix/array
    test_scores[step*FLAGS.batch_size:step*FLAGS.batch_size+FLAGS.batch_size,] = tmp_sigmoid

  accuracy = true_count / steps_per_epoch
  test_loss = test_loss / steps_per_epoch

  print('Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f Test Loss: %0.04f' %
    (cases, true_count, accuracy, test_loss))

  return accuracy, test_loss, test_scores # return precision

''' START '''
# Get the Test set of seqs and labels from hdf5 formated file -----------------
h5f = h5py.File(FLAGS.test_file, 'r')
if FLAGS.test_on == 'test':
    test_seqs = h5f['test_seqs'][()]
    test_labels = h5f['test_labels'][()]
elif FLAGS.test_on == 'valid':
    test_seqs = h5f['validation_seqs'][()]
    test_labels = h5f['validation_labels'][()]
else:
    raise ValueError("FLAGS.test_on must by be either \"test\" or \"valid\"!")

if FLAGS.only > 0:
    test_seqs = test_seqs[0:FLAGS.only,:,:]
    test_labels = test_labels[0:FLAGS.only,:]

print('Test data:')
print(test_seqs.shape)
test_cases = np.shape(test_seqs)[0]

# make test dir
if not os.path.exists(FLAGS.test_dir):
    os.makedirs(FLAGS.test_dir)

# Load Model -------------------------------------------------------------------
# Create a session for running Ops on the Graph.
config = tf.ConfigProto();
if FLAGS.run_on == 'gpu':
    config.gpu_options.visible_device_list = str(FLAGS.gpu)
config.allow_soft_placement = True

with tf.Session(config = config) as sess:
    # load meta graph and restore weights
    saver = tf.train.import_meta_graph(FLAGS.graph)
    saver.restore(sess, FLAGS.model)
    # get place holders and ops ------------------------------------------------
    graph = tf.get_default_graph()
    seqs_placeholder = graph.get_tensor_by_name("seqs:0")
    labels_placeholder = graph.get_tensor_by_name("labels:0")
    keep_prob_inner_placeholder = graph.get_tensor_by_name("keep_prob_inner:0")
    keep_prob_outer_placeholder = graph.get_tensor_by_name("keep_prob_outer:0")
    logits = tf.get_collection("logits")[0]
    logits = tf.get_collection("logits")[0]
    # Define loss test new to adjust batch size (TODO find a more flexible way later?!)
    loss_test = dlmodel.loss_test(logits, labels_placeholder, FLAGS.batch_size)
    eval_op = tf.get_collection("eval_op")[0]
    sigmoid_op = tf.sigmoid(logits)

    # make test predictions ----------------------------------------------------
    print('Test Data Accuracy Eval:')
    test_accuracy, test_loss, test_scores = do_eval(sess,
            eval_op,
            loss_test,
            sigmoid_op,
            seqs_placeholder,
            labels_placeholder,
            test_seqs,
            test_labels,
            keep_prob_inner_placeholder,
            keep_prob_outer_placeholder
            )
    print('\"Exact\" Accuracy %s  Loss  %s' % (test_accuracy, test_loss))

    # slice specific columns if required
    if FLAGS.slize != 'all':
        print('Slizing columns: %s to %s' % (slize_scheme[0], slize_scheme[1]))
        test_scores = test_scores[:, slize_scheme[0]:(slize_scheme[1]+1)]
        test_labels = test_labels[:, slize_scheme[0]:(slize_scheme[1]+1)]

    # Save Txt of Scores and Labels --------------------------------------------
    if FLAGS.savetxt == 'True':
        np.savetxt(FLAGS.test_dir + '/test_scores_' + FLAGS.name_tag + '_save.txt',test_scores, fmt='%1.5f', delimiter='\t')
        np.savetxt(FLAGS.test_dir + '/test_labels_' + FLAGS.name_tag + '_save.txt', test_labels, fmt='%1i', delimiter='\t')

    # Calc and Plot ROC Curves -------------------------------------------------
    if FLAGS.prc == 'True':
        import matplotlib.pyplot as plt
        print('Calculating PR curves ...')

        # Calculate cases // batch to set batches to ensure end handeling !TODO make this robust!
        cases = test_labels.shape[0]
        steps_per_epoch = cases // FLAGS.batch_size
        test_range = range(steps_per_epoch*FLAGS.batch_size)

        print(test_labels.shape)
        print(test_scores.shape)

        # convert
        test_labels = test_labels.astype(int)

        # Compute ROC curve and ROC area for each class ----------------------------
        precision = dict()
        recall = dict()
        pr_auc = dict()
        class_range_to_iter = range(test_scores.shape[1])
        for i in class_range_to_iter:
            precision[i], recall[i], _ = precision_recall_curve(test_labels[test_range, i], test_scores[test_range, i])
            pr_auc[i] = auc(recall[i], precision[i])

        # Print ROC AUC values if specified --------------------------------------
        if FLAGS.roc_auc == 'True':
            fprc = open(FLAGS.test_dir + '/test_prc_aucs_' + FLAGS.name_tag + '_save.txt', "w")
            fprc.write("ID\tPRC_AUC\n")
            for i in class_range_to_iter:
                j = i + slize_scheme[0]
                fprc.write('%s\t%s\n' % (j, pr_auc[i]))
            fprc.close()

        # MICRO averaging ----------------------------------------------------------
        if FLAGS.slize == 'all':
            precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels[test_range].ravel(), test_scores[test_range].ravel())
        else:
            # specifc micro avg for slized columns
            #precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels[test_range, slize_scheme[0]:slize_scheme[1]].ravel(), test_scores[test_range, slize_scheme[0]:slize_scheme[1]].ravel())
            precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels[test_range,:].ravel(), test_scores[test_range,:].ravel())  # test labels and scores  arealready sliced

        pr_auc["micro"] = auc(recall["micro"], precision["micro"])

#        # MACRO averaging ----------------------------------------------------------
#        # aggregate all false positive rates
#        all_precision = np.unique(np.concatenate([precision[i] for i in class_range_to_iter]))
#        # Then interpolate all ROC curves at this points
#        mean_recall = np.zeros_like(all_precision)
#        for i in class_range_to_iter:
#            mean_recall += interp(all_precision, precision[i], recall[i])
#        # Finally average it and compute AUC
#        if FLAGS.slize == 'all':
#            mean_recall /= test_scores.shape[1]
#        else:
#            mean_recall /= slize_scheme[1] - slize_scheme[0] + 1
#
#        precision["macro"] = all_precision
#        recall["macro"] = mean_recall
#        pr_auc["macro"] = auc(recall["macro"], precision["macro"])

        # Print Results ------------------------------------------------------------
#        print('PR AUC - Micro Avg. %s  Macro Avg. %s' % (pr_auc['micro'], pr_auc['macro']))
        print('PR AUC - Micro Avg. %s' % (pr_auc['micro']))

        # Plot ---------------------------------------------------------------------
        plt.figure()
        for i in class_range_to_iter:
            plt.plot(recall[i], precision[i], lw=.5, color='grey')
        plt.plot(recall["micro"], precision["micro"],
                 label='micro-average PR curve (area = {0:0.2f})'
                       ''.format(pr_auc["micro"]),
                 color='orange', linewidth=2)
#        plt.plot(recall["macro"], precision["macro"],
#            label='macro-average PR curve (area = {0:0.2f})'
#                ''.format(pr_auc["macro"]),
#            color='blue', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(FLAGS.test_dir + '/plot_' + FLAGS.name_tag + '_' + FLAGS.test_on + '_pr_curves.png')

        # Assemble DataFrame


    # Calc and Plot ROC Curves -------------------------------------------------
    if FLAGS.roc == 'True':
        import matplotlib.pyplot as plt
        print('Calculating ROC curves ...')

        # Calculate cases // batch to set batches to ensure end handeling !TODO make this robust!
        cases = test_labels.shape[0]
        steps_per_epoch = cases // FLAGS.batch_size
        test_range = range(steps_per_epoch*FLAGS.batch_size)

        # Compute ROC curve and ROC area for each class ----------------------------
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        class_range_to_iter = range(test_scores.shape[1])
        for i in class_range_to_iter:
            fpr[i], tpr[i], _ = roc_curve(test_labels[test_range, i], test_scores[test_range, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Print ROC AUC values if specified --------------------------------------
        if FLAGS.roc_auc == 'True':
            froc = open(FLAGS.test_dir + '/test_roc_aucs_' + FLAGS.name_tag + '_save.txt', "w")   
            froc.write("ID\tROC_AUC\n")
            for i in class_range_to_iter:
                j = i + slize_scheme[0]
                froc.write('%s\t%s\n' % (j, roc_auc[i]))
            froc.close()

        # MICRO averaging ----------------------------------------------------------
        if FLAGS.slize == 'all':
            fpr["micro"], tpr["micro"], _ = roc_curve(test_labels[test_range].ravel(), test_scores[test_range].ravel())
        else:
            # specifc micro avg for slized columns
            #fpr["micro"], tpr["micro"], _ = roc_curve(test_labels[test_range, slize_scheme[0]:slize_scheme[1]].ravel(), test_scores[test_range, slize_scheme[0]:slize_scheme[1]].ravel())
            fpr["micro"], tpr["micro"], _ = roc_curve(test_labels[test_range, :].ravel(), test_scores[test_range, :].ravel())  # test labels and scores are already sliced
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # MACRO averaging ----------------------------------------------------------
        # aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in class_range_to_iter]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in class_range_to_iter:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        if FLAGS.slize == 'all':
            mean_tpr /= test_scores.shape[1]
        else:
            mean_tpr /= slize_scheme[1] - slize_scheme[0] + 1

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Print Results ------------------------------------------------------------
        print('ROC AUC - Micro Avg. %s  Macro Avg. %s' % (roc_auc['micro'], roc_auc['macro']))

        # Plot ---------------------------------------------------------------------
        plt.figure()
        for i in class_range_to_iter:
            plt.plot(fpr[i], tpr[i], lw=.5, color='grey')
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='orange', linewidth=2)
        plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='blue', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(FLAGS.test_dir + '/plot_' + FLAGS.name_tag + '_' + FLAGS.test_on + '_roc_curves.png')
