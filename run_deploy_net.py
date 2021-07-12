'''DESCRIPTION:
Run predictions using a trained model.
Either give a sequence window in bed-like format and report its score or give a
vcf file with variants --> extract ref and var sequence and calculate the
damage score per class.

# FORMAT
BED:
    Regions smaller then bp_context will be extended to bp_context.
    Regions larger then bp_context will be centered on the central bp_context bp.
    Reads interprets: CHR START END
        reads and keeps a 4th NAME column
    Reports:
        P(class) per region per class in model
VCF:
    Reads, Required, only interprets: CHR POS NAME REF ALT (split at ,) ignores rest
    INDELS will be handled up to <max_indel_size> bp.
    Multiple Variants must be comma separated
    Will extract <bp_context> + 2 * 50 bp more to allow for indels up to 100bp
    and center on the cenral <bp_context> bp.
    Reports:
        total damage score: P(ref) - P(alt)
        logfold damage score: log(P(ref)/(1-P(ref))) log(P(alt)/(1-P(alt)))
        / per variant per class in model
    (Can report reference and varient scores separately if specified "damage_and_scores")

# Notes
    Requires bedtools to be loaded (module load bedtools) or accessible in PATH
    Always uses the positive strand.
    Assumes VCF variants to be coded from the plus/reference strand perspective!

# TODO
    Clean up script
    Add label attachment from a labels file.
'''

from __future__ import absolute_import, division, print_function
import os.path
import time
import sys
import re
import h5py
import numpy as np
import tensorflow as tf
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
flags.DEFINE_string('do', 'class', 'Select what to do default: predict the \'class\' of a sequence or \'damage\' per class; \'damage_and_scores\' will report the damages as well as the raw scores for reference and variant')
flags.DEFINE_string('slize', 'all', 'Comma separated list of start and end position of columns to slice out (0) indexed. Will use all if unspecified.')
# EXTERNAL files
flags.DEFINE_string('input', '', 'Must be a BED like file for \"--do class\" and a vcf like file for \"--do damage\"')
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

# PREPARATION ------------------------------------------------------------------
# import dl model architechture selected
dlmodel = __import__(FLAGS.dlmodel)

# prepare for column slizes if specified
if FLAGS.slize != 'all':
    slize_scheme = [x.strip() for x in FLAGS.slize.split(',')]
    slize_scheme = list(map(int, slize_scheme))

# GLOBAL OPTIONS ---------------------------------------------------------------
max_indel_size = 0

# TODO get INDELS and SNPs to work simulataneously

indel_offset = int(max_indel_size//2)
chunk_size = 100  # chunk size to read and process lines from input file

# HELPER FUNCTIONS -------------------------------------------------------------
# Helper get hotcoded sequence
def get_hot_coded_seq(sequence):
    """Convert a 4 base letter sequence to 4-row x-cols hot coded sequence"""
    # initialise empty
    hotsequence = np.zeros((len(sequence),4))
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

def predict(sess,
    get_sigmoid,
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
    predictions = np.zeros((cases, FLAGS.num_classes))  # init empty predictions array
    for step in range(batches_to_run):
        test_batch_start = step * FLAGS.batch_size
        test_batch_end = step * FLAGS.batch_size + FLAGS.batch_size
        test_batch_range=range(test_batch_start, test_batch_end)
        feed_dict = {
              seqs_placeholder: seqs[test_batch_range],
              keep_prob_inner_placeholder: 1.0,
              keep_prob_outer_placeholder: 1.0
              }
        tmp_sigmoid = sess.run(sigmoid_op, feed_dict=feed_dict)
        tmp_sigmoid = np.asarray(tmp_sigmoid)
        tmp_sigmoid = np.squeeze(tmp_sigmoid)
        # add to the empty prediction scores array
        predictions[step*FLAGS.batch_size:step*FLAGS.batch_size+FLAGS.batch_size,] = tmp_sigmoid

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
        tmp_sigmoid = sess.run(sigmoid_op, feed_dict=feed_dict)
        tmp_sigmoid = np.asarray(tmp_sigmoid)
        tmp_sigmoid = np.squeeze(tmp_sigmoid)
        # workaround for single value prediction (only use last remaining corresponding predicitons)
        predictions[-remaining:,] = tmp_sigmoid[-remaining:]

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

def make_bed_chunk_vcf(lines):
    ''' Module Function: Make Bed out of Chunk of VCF Variants
    Args:
        lines: chunk of lines in bed-like format
    Returns:
        bed: pybedtools Bed object of bed regions fitted to the bp_context of the
            network to employ. (extended or recentered if necessary)
        var_dict: dictionary linkning each tag, ref, vars and sequence
        tag_list: list of tags in order for maintaining order for processing and printing
    '''
    line_bed = []  # init empty string to store bed coords
    var_dict = {}  # init a var_dictionary
    tag_list = []  # init a list of tags
    line_counter = 0

    for line in lines:
        if re.match('^#', line):  # skip comment and header lines
            continue
        line_split = line.split()
        tmp_chr = line_split[0]
        # format chromosome
        if not re.match('^chr', tmp_chr):
            tmp_chr = re.sub('^Chr', 'chr', tmp_chr)
            tmp_chr = re.sub('^CHR', 'chr', tmp_chr)
            if not re.match('^chr', tmp_chr):
                tmp_chr = 'chr' + tmp_chr
        tmp_pos = int(line_split[1])
        # create name tag if no name provided
        if line_split[2] == '.':
            tmp_tag = line_split[0] + ':' + str(line_split[1])
        else:
            tmp_tag = line_split[2]
        # get <bp_context> bases around Variant
        # +1 for bed coordinates required but vcf coords are 0-based s o no +1 in tm_start and no +1 in tmp_end
        tmp_start = tmp_pos - int(FLAGS.bp_context//2) - indel_offset  # var pos will be leftbase of center
        tmp_end = tmp_pos + int(FLAGS.bp_context//2) + indel_offset # extend 2*50 bp more to allow for indels up to 100bp

        # get ref and var bases --> assemble var_dict
        tmp_ref = line_split[3]
        tmp_var = line_split[4]
        # split multiple vars
        if ',' in tmp_var:
            tmp_vars = [x.strip() for x in tmp_var.split(',')]
            tmp_vars = list(map(str, tmp_vars))
            for v in iter(tmp_vars):
                tmp_tag_v = tmp_tag + ':' + tmp_ref + ':' + v
                tag_list.append(tmp_tag_v)
                var_dict[tmp_tag_v] = {}  # init
                var_dict[tmp_tag_v]['chr'] = tmp_chr
                var_dict[tmp_tag_v]['pos'] = tmp_pos
                var_dict[tmp_tag_v]['ref'] = tmp_ref
                var_dict[tmp_tag_v]['var'] = v
                var_dict[tmp_tag_v]['tag'] = tmp_tag_v
                var_dict[tmp_tag_v]['refline'] = line_counter # add which line in bed/seq chunks corresponds
                var_dict[tmp_tag_v]['varline'] = -1  # hold open for assigning var seq line later
        else:
            tag_list.append(tmp_tag)
            var_dict[tmp_tag] = {}  # init
            var_dict[tmp_tag]['chr'] = tmp_chr
            var_dict[tmp_tag]['pos'] = tmp_pos
            var_dict[tmp_tag]['ref'] = tmp_ref
            var_dict[tmp_tag]['var'] = tmp_var
            var_dict[tmp_tag]['tag'] = tmp_tag
            var_dict[tmp_tag]['refline'] = line_counter # add which line in bed/seq chunks corresponds
            var_dict[tmp_tag]['varline'] = -1  # hold open for assigning var seq line later

        # append to storage for pybedtools
        line_bed.append((tmp_chr, tmp_start, tmp_end, tmp_tag))

        line_counter += 1

    # create BedTool object as bed file
    bed = BedTool(line_bed)

    return bed, var_dict, tag_list

''' START '''
# check if existent --> else create out_dir and Init Output File ---------------
if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

if FLAGS.do == 'class':
    out_file = FLAGS.out_dir + '/class_scores_' + FLAGS.name_tag + '.bed'
elif FLAGS.do in ['damage', 'damage_and_scores']:
    out_file = FLAGS.out_dir + '/total_damage_scores_' + FLAGS.name_tag + '.bed'
    out_file2 = FLAGS.out_dir + '/logfold_damage_scores_' + FLAGS.name_tag + '.bed'
else:
    print('Please Select \"class\" of \"damage\" as value for \"--do\" ... exiting ...')
    sys.exit()

if FLAGS.do == 'damage_and_scores':
    out_file3 = FLAGS.out_dir + '/referece_class_scores_' + FLAGS.name_tag + '.bed'
    out_file4 = FLAGS.out_dir + '/variant_class_scores_' + FLAGS.name_tag + '.bed'

# Set Genome -------------------------------------------------------------------
fasta = pybedtools.BedTool(FLAGS.genome)

# Load Model -------------------------------------------------------------------
# Create a session
config = tf.ConfigProto();
if FLAGS.run_on == 'gpu':
    config.gpu_options.visible_device_list = str(FLAGS.gpu)
config.allow_soft_placement = True

# Launch Session and retrieve stored OPs and Variables
with tf.Session(config = config) as sess:
    # load meta graph and restore weights
    saver = tf.train.import_meta_graph(FLAGS.model + '.meta')
    saver.restore(sess, FLAGS.model)
    # get placeholders and ops ------------------------------------------------
    graph = tf.get_default_graph()
    seqs_placeholder = graph.get_tensor_by_name("seqs:0")
    labels_placeholder = graph.get_tensor_by_name("labels:0")
    keep_prob_inner_placeholder = graph.get_tensor_by_name("keep_prob_inner:0")
    keep_prob_outer_placeholder = graph.get_tensor_by_name("keep_prob_outer:0")
    logits = tf.get_collection("logits")[0]
    sigmoid_op = tf.sigmoid(logits)

    # Start working through input file in chunks -------------------------------
    done_count = 0
    with open(out_file, "w") as fw:
        with open(FLAGS.input, "r") as infile:
            # open second outfile if damage predictions
            if FLAGS.do in ['damage', 'damage_and_scores']:
                fw2 = open(out_file2, "w")
            if FLAGS.do == 'damage_and_scores':
                fw3ref = open(out_file3, "w")
                fw4var = open(out_file4, "w")
            # Read File in chunks of chunk_size lines
            while True:
                lines = list(islice(infile, chunk_size))
                if not lines:
                    break
                # Create Bed Chunk to extract sequences from input Chunk -------
                if FLAGS.do == 'class':
                    chunk_bed = make_bed_chunk_bed(lines)
                elif FLAGS.do in ['damage', 'damage_and_scores']:
                    # get chunk bed and a variant dictionary back from the vcf
                    chunk_bed, var_dict, tag_list = make_bed_chunk_vcf(lines)


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

                    # For VCF add Variant sequences and link to them in var_dict ---
                    # for each variant tag in var_dict
                    if FLAGS.do in ['damage', 'damage_and_scores']:
                        replace_base = int(FLAGS.bp_context//2) + indel_offset - 1  # -1 for 0-based indexing
                        for t in tag_list:
                            tmp_seq = list(chunk_seqs[var_dict[t]['refline']])  # get reference sequence
                            # add variant
                            # handle deletion as zero length strings
                            tmp_ref = var_dict[t]['ref']
                            if tmp_ref == '.':
                                tmp_ref = ''
                            tmp_var = var_dict[t]['var']
                            if tmp_var == '.':
                                tmp_var = ''
                            tmp_ref_length = len(tmp_ref)  # save a reference temp to handle multiple base reference variant

                            # check reference sequence
                            if ''.join(tmp_seq[replace_base:(replace_base+tmp_ref_length)]) == tmp_ref:
                                    tmp_seq[replace_base:(replace_base+tmp_ref_length)] = tmp_var
                                    tmp_seq = ''.join(tmp_seq)
                                    chunk_seqs.append(tmp_seq)  # add to chunk seqs
                                    var_dict[t]['varline'] = len(chunk_seqs) - 1 # reference in dictionary
                            # Else replace reference base with base specifed in vcf and flag as warning
                            else:
                                tmp_ref_seq = list(chunk_seqs[var_dict[t]['refline']])
                                old_ref_base = ''.join(tmp_ref_seq[replace_base:(replace_base+tmp_ref_length)])
                                tmp_ref_seq[replace_base:(replace_base+tmp_ref_length)] = tmp_ref
                                chunk_seqs[var_dict[t]['refline']] = ''.join(tmp_ref_seq)
                                tmp_ref_seq[replace_base] = tmp_var
                                tmp_ref_seq = ''.join(tmp_ref_seq)
                                chunk_seqs.append(tmp_ref_seq)  # add to chunk seqs
                                var_dict[t]['varline'] = len(chunk_seqs) - 1 # reference in dictionary
                                print('Warning replacing reference sequence base %s with ref base %s provided in vcf file for %s. Please encode variants from reference sequence perspective!' % (old_ref_base, var_dict[t]['ref'], var_dict[t]['tag']))


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
                chunk_predictions = predict(
                    sess,
                    sigmoid_op,
                    seqs_placeholder,
                    chunk_hotseqs,
                    keep_prob_inner_placeholder,
                    keep_prob_outer_placeholder)
                # Slice relevant class predicitons if specified
                if FLAGS.slize != 'all':
                    chunk_predictions = chunk_predictions[:, slize_scheme]

                # Aggregate and Print Bed Regions with predicitons -------------
                if FLAGS.do == 'class':
                    for i in range(chunk_predictions.shape[0]):
                        tmp_string = str(chunk_bed[i])
                        tmp_string = tmp_string.strip() + '\t' + '\t'.join(map(str,chunk_predictions[i,:])) + '\n'
                        fw.write(tmp_string)

                # Match variants and Calculate Damage score --------------------
                elif FLAGS.do in ['damage', 'damage_and_scores']:
                    for t in tag_list:  # for each variant
                        damage_list = []
                        log_odds_list = []

                        if FLAGS.do == 'damage_and_scores':
                            ref_score_list = []
                            var_score_list = []

                        for i in range(chunk_predictions.shape[1]):
                            # predicted scores
                            tmp_ref_score = chunk_predictions[var_dict[t]['refline'], i]
                            tmp_var_score = chunk_predictions[var_dict[t]['varline'], i]
                            # calc damage score(s)
                            if FLAGS.do == 'damage_and_scores':
                                ref_score_list.append(tmp_ref_score)
                                var_score_list.append(tmp_var_score)

                            # damage
                            tmp_damage = tmp_ref_score - tmp_var_score
                            # relative lof fold change of odds
                            tmp_log_odds_damage = abs(log(tmp_ref_score/(1-tmp_ref_score)) - log(tmp_var_score/(1-tmp_var_score)))
                            damage_list.append(tmp_damage)
                            log_odds_list.append(tmp_log_odds_damage)

                        # print out --------------------------------------------
                        out_string_1 = [var_dict[t]['chr'],
                            var_dict[t]['pos'],
                            var_dict[t]['tag'],
                            var_dict[t]['ref'],
                            var_dict[t]['var'],
                            '\t'.join(map(str, damage_list))]

                        out_string_2 = [var_dict[t]['chr'],
                            var_dict[t]['pos'],
                            var_dict[t]['tag'],
                            var_dict[t]['ref'],
                            var_dict[t]['var'],
                            '\t'.join(map(str, log_odds_list))]

                        # Direct towards separate files for damage predition types!
                        fw.write('\t'.join(map(str, out_string_1)) + '\n')
                        fw2.write('\t'.join(map(str, out_string_2)) + '\n')

                        # report scores if flagged so
                        if FLAGS.do == 'damage_and_scores':

                            out_string_3 = [var_dict[t]['chr'],
                                var_dict[t]['pos'],
                                var_dict[t]['tag'],
                                var_dict[t]['ref'],
                                var_dict[t]['var'],
                                '\t'.join(map(str, ref_score_list))]

                            out_string_4 = [var_dict[t]['chr'],
                                var_dict[t]['pos'],
                                var_dict[t]['tag'],
                                var_dict[t]['ref'],
                                var_dict[t]['var'],
                                '\t'.join(map(str, var_score_list))]

                            fw3ref.write('\t'.join(map(str, out_string_3)) + '\n')
                            fw4var.write('\t'.join(map(str, out_string_4)) + '\n')

                # Report Progress
                done_count += 1
                print('Processed lines: %s' % (chunk_size * done_count))

print("Finished ...")

# close up
fw.close()
if FLAGS.do in ['damage', 'damage_and_scores']:
    fw2.close()
if FLAGS.do == 'damage_and_scores':
    fw3ref.close()
    fw4var.close()
