'''DESCRIPTION:
Run script to get saliency scores for class probability predictions over bed regions.

# FORMAT
BED:
    Regions smaller then bp_context will be extended to bp_context.
    Regions larger then bp_context will be centered on the central bp_context bp.
    Reads interprets: CHR START END
        reads and keeps a 4th NAME column
    Reports:
        Saliency scores for every base pair in the region either:
        * multiplied with the one-hot-encoded sequence and reduced to one
          output per position (DEFAULT)
        * or 4 saliency values per one for each base present or not (TO IMPLEMENT)

# Notes
    Requires bedtools to be loaded (module load bedtools) or accessible in PATH
    Always uses the positive strand.

# TODO
'''

from __future__ import absolute_import, division, print_function
import os.path
import time
import sys
import re
import h5py
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from math import log
from itertools import islice, cycle
import pybedtools
from pybedtools import BedTool
# from Bio.Seq import Seq
pybedtools.helpers.set_tempdir('.')

# Basic model parameters as external flags -------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dlmodel', 'deepHaemWindow', 'Specifcy the DL model file to use e.g. <endpoolDeepHaemElement>.py')
# RUN SETTINGS
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_string('out_dir', 'predictions_dir', 'Directory to store the predicted results')
flags.DEFINE_string('name_tag', 'pred', 'Nametag to add to filenames')
# WHAT TO DO
flags.DEFINE_integer('select', 0, 'Select a single classifier relative to which the saliency should be calculated. (default: 0)')
flags.DEFINE_string('gradient_input', 'sigmoid', 'Select \"sigmoid\" or \"logit\" (score before sigmoid transformation) relative to which to calculate the saliency score. default: sigmoid')
flags.DEFINE_integer('rounddecimals', 10, 'Select the number of decimal places to round saliency to. [Default 10]')
# flags.DEFINE_string('saliency_report_bases', 'present', 'Only report the saliency value for the \"present\" base (multiple saliencies with one-hot-encoded sequence). or for \"all\" bases.')
# network needs to be reconstructed so specifcy conv scheme and such
flags.DEFINE_integer('conv_layers', 5, 'Number of convolutional layers.')
flags.DEFINE_string('hidden_units_scheme', '1000,1000,1000,1000,1000', 'Comma seperated hidden units scheme. Must have length of number of conv layers specified!')
flags.DEFINE_string('kernel_width_scheme', '20,10,8,4,8', 'Comma seperated kernel width scheme. Must have length of number of conv layers specified!')
flags.DEFINE_string('max_pool_scheme', '3,4,5,10,4', 'Comma seperated max pool scheme. Must have length of number of conv layers specified!')
flags.DEFINE_boolean('upstream_connected', True, 'Add an upstream fully connecte layer [True/False].')
flags.DEFINE_integer('upstream_connections', 100, 'Connections for the upstream fully connected layer.')
# EXTERNAL files
flags.DEFINE_string('input', '', 'Must be a BED like file')
flags.DEFINE_string('model', './model', 'Checkpoint of model file to be tested. (Full path to model without suffix!)')
# flags.DEFINE_string('graph', './model.meta', 'Defined graph of model. (Full path to model File)')
# flags.DEFINE_string('labels', '', 'File conmtaining Class labels according to the model (one per line)')
flags.DEFINE_string('genome', 'hg19.fasta', 'Full path to fasta reference genome of interest to extract the sequence from.')
# Data Options
flags.DEFINE_integer('bp_context', 1000, 'Basepairs per feature.')
flags.DEFINE_integer('num_classes', 936, 'Number of classes.')
# machine options
flags.DEFINE_string('run_on', 'gpu', 'Select where to run on (cpu or gpu)')
flags.DEFINE_integer('gpu', 0, 'Select a single available GPU and mask the rest. Default 0.')
flags.DEFINE_string('stored_dtype', "float32", 'Indicate what data format sequence and labels where stored in. ["bool", "int8", "int32"]')



# PREPARATION ------------------------------------------------------------------
# import dl model architechture selected
dlmodel = __import__(FLAGS.dlmodel)

# GLOBAL OPTIONS ---------------------------------------------------------------
max_indel_size = 20

indel_offset = int(max_indel_size//2)
chunk_size = 100  # chunk size to read and process lines from input file

# PREP CONVOLUTIONAL SCHEME ----------------------------------------------------
hidden_units_scheme = [x.strip() for x in FLAGS.hidden_units_scheme.split(',')]
hidden_units_scheme = list(map(int, hidden_units_scheme))
kernel_width_scheme = [x.strip() for x in FLAGS.kernel_width_scheme.split(',')]
kernel_width_scheme = list(map(int, kernel_width_scheme))
max_pool_scheme = [x.strip() for x in FLAGS.max_pool_scheme.split(',')]
max_pool_scheme = list(map(int, max_pool_scheme))

# HELPER FUNCTIONS -------------------------------------------------------------
# Helper get hotcoded sequence
def get_hot_coded_seq(sequence):
    """Convert a 4 base letter sequence to 4-row x-cols hot coded sequence"""
    # initialise empty
    hotsequence = np.zeros((len(sequence),4))
    sequence = sequence.upper()
    # set hot code 1 according to gathered sequence
    for i in range(len(sequence)):
        if sequence[i] == 'A':
            hotsequence[i,0] = 1
        elif sequence[i] == 'C':
            hotsequence[i,1] = 1
        elif sequence[i] == 'G':
            hotsequence[i,2] = 1
        elif sequence[i] == 'T':
            hotsequence[i,3] = 1
    # return the numpy array
    return hotsequence

def calc_saliency(sess,
    get_saliency,
    seqs_placeholder,
    seqs,
    keep_prob_inner_placeholder,
    keep_prob_outer_placeholder
    ):
    """Make predictions --> get sigmoid output of net per sequence and class"""

    cases = seqs.shape[0]
    batches_to_run = cases // FLAGS.batch_size
    # cover cases where remainder cases are left
    remaining = cases - FLAGS.batch_size * batches_to_run
    predictions = np.zeros((cases, FLAGS.bp_context, 4))  # init empty predictions array
    for step in range(batches_to_run):
        test_batch_start = step * FLAGS.batch_size
        test_batch_end = step * FLAGS.batch_size + FLAGS.batch_size
        test_batch_range=range(test_batch_start, test_batch_end)
        feed_dict = {
              seqs_placeholder: seqs[test_batch_range],
              keep_prob_inner_placeholder: 1.0,
              keep_prob_outer_placeholder: 1.0
              }
        tmp_saliency = sess.run(saliency_op, feed_dict=feed_dict)
        tmp_saliency = np.asarray(tmp_saliency)
        tmp_saliency = np.squeeze(tmp_saliency)

        # add to the empty prediction scores array
        predictions[step*FLAGS.batch_size:step*FLAGS.batch_size+FLAGS.batch_size,] = tmp_saliency

    # handle remaining cases
    if remaining > 0:
        test_batch_range=range(cases-remaining, cases)
        # workaround for single value prediction
        if remaining == 1:
            test_batch_range=range(cases-remaining-1, cases)
        feed_dict = {
              seqs_placeholder: seqs[test_batch_range],
              keep_prob_inner_placeholder: 1.0,
              keep_prob_outer_placeholder: 1.0
              }
        tmp_saliency = sess.run(saliency_op, feed_dict=feed_dict)
        tmp_saliency = np.asarray(tmp_saliency)
        tmp_saliency = np.squeeze(tmp_saliency)
        # workaround for single value prediction (only use last remaining corresponding predicitons)
        predictions[-remaining:,] = tmp_saliency[-remaining:]

    return predictions

def make_bed_chunk_bed(lines):
    ''' Module Function: Make Bed out of Chunk of Bed Regions
    Args:
        lines: chunk of lines in bed-like format
    Returns:
        pybedtools Bed object of bed regions fitted to the bp_context of the
        network to employ. (extended or recentered if necessary)
    '''
    line_bed = []  # init empty string to store bed coords

    for line in lines:
        if re.match('^#', line):  # skip comment and header lines
            continue
        line_split = line.split()
        # create name tag if no name provided
        if len(line_split) < 4:
            tmp_tag = line_split[0] + ':' + str(line_split[1]) + "-" + str(line_split[2])
        elif line_split[3] == '':
            tmp_tag = line_split[0] + ':' + str(line_split[1]) + "-" + str(line_split[2])
        else:
            tmp_tag = line_split[3]
        tmp_chr = line_split[0]
        tmp_start = int(line_split[1])
        tmp_end = int(line_split[2])
        # recenter or extand coordinates if necessary
        # extend if smaller then bp_context/ recneter if bigger
        if tmp_end - tmp_start != FLAGS.bp_context :
            tmp_hlfwy = round((tmp_end - tmp_start) // 2) + tmp_start
            tmp_start = int(tmp_hlfwy - FLAGS.bp_context // 2)
            tmp_end = int(tmp_hlfwy + FLAGS.bp_context // 2)
        # append to storage for pybedtools
        line_bed.append((tmp_chr, tmp_start, tmp_end, tmp_tag))

    # create BedTool object as bed file
    bed = BedTool(line_bed)

    return bed

''' START '''
# check if existent --> else create out_dir and Init Output File ---------------
if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

out_file = FLAGS.out_dir + '/class_score_saliency_' + FLAGS.name_tag + '.bed'

# check if batch size 1 selected (need to fix that)
if FLAGS.batch_size <= 1:
    print("Please select a batch size greater then 1!")
    sys.exit()

# Set Genome -------------------------------------------------------------------
fasta = pybedtools.BedTool(FLAGS.genome)

# REBUILD NETWORK in default graph and load weigths to map saliency -----------
# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():

    # Generate placeholders for the seqs and labels (and dropout prob).
    seqs_placeholder = tf.placeholder(tf.float32, [None, FLAGS.bp_context, 4], name='seqs')
    labels_placeholder = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name='labels')
    keep_prob_inner_placeholder = tf.placeholder(tf.float32, name='keep_prob_inner')
    keep_prob_outer_placeholder = tf.placeholder(tf.float32, name='keep_prob_outer')

    # Building the Graph -------------------------------------------------------
    # Ops to calc logits
    logits = dlmodel.inference(
        seqs_placeholder,
        FLAGS.conv_layers,
        hidden_units_scheme,
        kernel_width_scheme,
        max_pool_scheme,
        FLAGS.upstream_connected,
        FLAGS.upstream_connections,
        FLAGS.num_classes,
        FLAGS.batch_size,
        keep_prob_inner_placeholder,
        keep_prob_outer_placeholder,
        False,
        [],
        [],
        False
        )

    # saliency ops
    sigmoid_op = tf.sigmoid(logits)
    # print("Sigmoid OP shape:")
    # print(tf.shape(sigmoid_op))
    if FLAGS.gradient_input == 'sigmoid':
        saliency_op = tf.gradients(sigmoid_op[:,FLAGS.select], seqs_placeholder)
    elif FLAGS.gradient_input == 'logits':
        saliency_op = tf.gradients(logits[:,FLAGS.select], seqs_placeholder)
    else:
        print("Select sigmoid or logit relative to which to calculate the gradient to")
        sys.exit()
    # print("Saliency OP shape:")
    # print(tf.shape(saliency_op))

    # SET SAVER ---------------------------------------------------
    saver = tf.train.Saver()

    # init op
    init = tf.global_variables_initializer()

    # Load Model -------------------------------------------------------------------
    # Create a session
    config = tf.ConfigProto();
    if FLAGS.run_on == 'gpu':
        config.gpu_options.visible_device_list = str(FLAGS.gpu)
    config.allow_soft_placement = True

    # Launch Session and retrieve stored OPs and Variables
    with tf.Session(config = config) as sess:
        # load meta graph and restore weights
        # saver = tf.train.import_meta_graph(FLAGS.model + '.meta')
        sess.run(init)
        saver.restore(sess, FLAGS.model)

        # Start working through input file in chunks -------------------------------
        done_count = 0
        with open(out_file, "w") as fw:
            with open(FLAGS.input, "r") as infile:
                # Read File in chunks of chunk_size lines
                while True:
                    lines = list(islice(infile, chunk_size))
                    if not lines:
                        break
                    # Create Bed Chunk to extract sequences from input Chunk -------
                    chunk_bed = make_bed_chunk_bed(lines)

                    # Get Sequences of Regions -------------------------------------
                    chunk_bed.sequence(fi=fasta)  # extract sequences
                    # clean and append sequences
                    chunk_seqs = []
                    with open(chunk_bed.seqfn, "r") as seqin:
                        for seq in iter(seqin):
                            if re.match('^>', seq):  # fasta locations
                                continue
                            seq = seq.strip()
                            seq = seq.upper()  # clean up sequence
                            chunk_seqs.append(seq)  # append

                            # Recenter all sequences to <bp_context> centered bases (indel handling)
                            for s in range(len(chunk_seqs)):
                                # get center, start and end
                                tmp_center = int(len(chunk_seqs[s])//2)
                                tmp_start = tmp_center - int(FLAGS.bp_context//2)  # -1 for 0-coord
                                tmp_end = tmp_center + int(FLAGS.bp_context//2)
                                # chop both sites equally
                                chunk_seqs[s] = chunk_seqs[s][tmp_start:tmp_end]


                    # make hot encoded numpy array ---------------------------------
                    chunk_hotseqs = []
                    for seq in chunk_seqs:
                        # print(seq[0:10])
                        seq = get_hot_coded_seq(seq)  # hot encode
                        chunk_hotseqs.append(seq)
                    chunk_hotseqs = np.asarray(chunk_hotseqs)

                    # Make Predictions with Sequences ------------------------------
                    chunk_predictions = calc_saliency(
                        sess,
                        saliency_op,
                        seqs_placeholder,
                        chunk_hotseqs,
                        keep_prob_inner_placeholder,
                        keep_prob_outer_placeholder)


                    # multiply with sequence ---------------------------------------
                    chunk_predictions = chunk_predictions * chunk_hotseqs
                    # summarise per base pair position
                    chunk_predictions = np.sum(chunk_predictions, axis=2)
                    # # normalize for number of classifiers that are combined
                    # print('Dividing by %s classifiers whos saliency was combined ...' % len(slize_scheme))
                    # chunk_predictions = chunk_predictions / len(slize_scheme)
                    # round
                    chunk_predictions = np.round(chunk_predictions, decimals = FLAGS.rounddecimals)

                    # Aggregate and Print Bed Regions with predicitons -------------
                    for i in range(chunk_predictions.shape[0]):
                        tmp_string = str(chunk_bed[i])
                        tmp_string = tmp_string.strip() + '\t' + '\t'.join(map(str,chunk_predictions[i,:])) + '\n'
                        fw.write(tmp_string)

                    # Report Progress
                    done_count += 1
                    print('Processed lines: %s' % (chunk_size * done_count))

                    # clean temp bed files every 100 steps
                    if done_count % 100 == 0:
                        pybedtools.helpers.cleanup()

print("Finished ...")

# close up
fw.close()
