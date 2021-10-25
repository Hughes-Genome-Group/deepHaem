"""
Convert the raw sequence and the lables to hdf5 data/arrays for faster batch reading.
Split data into training, test and validation set. Save training and test set in same file.
Will store a .h5 file with the labels and sequences and a coord file per test/valid and train set
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import h5py
import argparse
import sys
from operator import itemgetter

# Define arguments -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="""Take a raw sequence and a labels bed like file and encode and store
    both as numpy arrays. Split up into traiing, test and validation samples.""")
parser.add_argument('in_file', type=str,
                    help='Five column file. [chr start end comma separated IDs to split and raw sequence].')
parser.add_argument('--split_mode', dest='split_mode', default='random', choices=['random', 'chr'],
    help="""Specify how to split up the data into training, test
    and validation set. []chr] - select chromomes from which
    features are attritbuted to the different sets. [random]
    - split the features by random sampling. Needs --chr_test
    --chr_valid or --frac_test --frac_valid declared respectively.
    Default = [random]""")
parser.add_argument('--frac_test', type=float, dest='frac_test', default=0.05,
    help='Fraction of total set to sample into test set. (Float > 1.0)')
parser.add_argument('--frac_valid', type=float, dest='frac_valid', default=0.05,
    help='Fraction of total set to sample into validation set. (Float > 1.0)')
parser.add_argument('--chr_test', nargs='+', dest='chr_test', default='chr20',
    help="""Select one ore more space separated chromosome (chr1 chr2 chr3) to use
     as test chromosomes. Default = chr20 Only if split_mode = 'chr' """)
parser.add_argument('--chr_valid', nargs='+', dest='chr_valid', default='chr21',
    help="""Select one ore more space separated chromosome (chr1 chr2 chr3) to use
     as validation chromosomes. Default = chr20 Only if split_mode = 'chr' """)
parser.add_argument('--save_prefix', dest='save_prefix', default='./data_set',
    help='Prefix to store the training/ test and validation sets. Default = ./data_set')
parser.add_argument('--seed', dest='seed', type=int, default=1234,
    help='Random seed for sampling.')
parser.add_argument('--trim_seq', dest='trim_seq', type=int, default=0,
    help='Number of bp to rim the sequence from both ends (default 0).')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=936,
    help='Specify number of classes.')
parser.add_argument('--store_bool', dest='store_bool', type=bool, default=False,
    help='Indicate if to store the 1-hot encoded sequence and labels as bool dtype (convert in train script etc.).')

# Parse arguments
args = parser.parse_args()


if args.store_bool:
    print("Stoing as boolean ...")

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

print("\n# === Creating a Training, Test and Validation Set from provided input === #")

# Set seed for random sampling -------------------------------------------------
np.random.seed(args.seed)

# init binary representatons --------------------------------------------------
# Inititialize Classes strucutre to store
if args.num_classes < 2:
    sys.exit("No distinct classes specified: %s !" % args.num_classes)
num_ids = args.num_classes  # get number of unique ids
print("\nNumber of distinct labels: " + str(num_ids))
# make a look-up dictionary with a binary label per id
bin_look_up = {}
for i in range(num_ids):
    if args.store_bool:
        bin_look_up[i] = np.zeros((num_ids), dtype=np.bool)
        bin_look_up[i][i] = True
    else:
        bin_look_up[i] = np.zeros((num_ids), dtype=np.uint8)
        bin_look_up[i][i] = 1

# test print some bin_look_up_lines
print("Test Print first bin_look_up lines:")
print(bin_look_up[0][0:4])
print(bin_look_up[1][0:4])
#print(bin_look_up[2][0:4])
#print(bin_look_up[3][0:4])

# Read in data -----------------------------------------------------------------
# init single empty binary label array for access later
if args.store_bool:
    label_bin_init = np.zeros(num_ids,  dtype=np.bool)
else:
    label_bin_init = np.zeros(num_ids,  dtype=np.uint8)

print("\nReading lines ...")

# read data in, split into vectors
with open(args.in_file, "r") as f:
    chroms = []
    start = []
    stop = []
    label = []
    for i,l in enumerate(f):
        l = l.rstrip()
        l = l.split("\t")
        chroms.append(l[0])
        start.append(l[1])
        stop.append(l[2])
        # get first sequence to estimate length and format
        if i == 0:
            temp_seq = l[4]
            # trim if desired
            if args.trim_seq > 0:
               temp_seq = temp_seq[args.trim_seq:-args.trim_seq]
            temp_seq = get_hot_coded_seq(temp_seq)

# Sample Test/ Validation and Training set according to selected mode -----------
input_rows = np.array(range(len(chroms)))  # make an array of input rows to sample from once

# if to split based on fractions randomly form all chromosomes
if args.split_mode == 'random':
    print("\nSampling randomly across chromosomes.")
    to_sample_test = round(len(input_rows) * args.frac_test)  # get fractions
    to_sample_valid = round(len(input_rows) * args.frac_valid)
    to_sample_train = len(input_rows) - to_sample_test - to_sample_valid
    print("%s Test cases\n%s Validation cases\n%s Training cases left." %
        (int(to_sample_test), int(to_sample_valid), int(to_sample_train)))
    # sample and get test and valid rows
    tmp_sampled = np.random.choice(input_rows, size=int(to_sample_test+to_sample_valid), replace=False)
    test_rows = tmp_sampled[range(int(to_sample_test))]
    test_rows = np.sort(test_rows)
    valid_rows = tmp_sampled[range(int(to_sample_test), int(to_sample_test+to_sample_valid))]
    valid_rows = np.sort(valid_rows)
    # prune remaining training cases
    training_rows = np.delete(input_rows, tmp_sampled)

elif args.split_mode == 'chr':
    print("\nSetting specifc chromosomes as test and validatipon set:")
    print("Using %s as Test, %s as Validation and the remaining as Training cases" % (args.chr_test, args.chr_valid))

    test_rows = []
    valid_rows = []

    # match row numbers against chromosomes
    for i in range(len(chroms)):
        if chroms[i] in args.chr_test:
            test_rows.append(i)
        if chroms[i] in args.chr_valid:
            valid_rows.append(i)
    # prune remaining training cases
    training_rows = np.delete(input_rows, (test_rows + valid_rows))
    print("%s Test cases \n%s Validation cases\n%s Training cases left." %
        (int(len(test_rows)), int(len(valid_rows)), int(len(training_rows))))
    test_rows = input_rows[test_rows,]
    valid_rows = input_rows[valid_rows,]
    # print(test_rows)
print("\nSampled into sets ...")

# write training/test/validation set coords ------------------------------------
print("\nStoring Coordinates ...")
write_train_coords = open(args.save_prefix + "_training_coords.bed", "w")
for tr in training_rows:
    write_train_coords.write("%s\t%s\t%s\n" % (chroms[tr], start[tr], stop[tr]))
write_test_coords = open(args.save_prefix + "_test_coords.bed", "w")
for tr in test_rows:
    write_test_coords.write("%s\t%s\t%s\n" % (chroms[tr], start[tr], stop[tr]))
write_valid_coords = open(args.save_prefix + "_validation_coords.bed", "w")
for tr in valid_rows:
    write_valid_coords.write("%s\t%s\t%s\n" % (chroms[tr], start[tr], stop[tr]))

# Initialize training and validation data in hdf5 files ------------------------
print("\nInitializing hdf5 Storage Files ...")
# and already store labels
train_h5f = h5py.File(args.save_prefix + "_training_data.h5", 'w')
if args.store_bool:
    set_train_seq = train_h5f.create_dataset('training_seqs', (training_rows.shape[0], temp_seq.shape[0], temp_seq.shape[1]) , dtype='b')
    set_test_seq = train_h5f.create_dataset('test_seqs', (test_rows.shape[0], temp_seq.shape[0], temp_seq.shape[1]), dtype='b')
    set_train_label = train_h5f.create_dataset('training_labels', (training_rows.shape[0], num_ids), dtype='b')
    set_test_label = train_h5f.create_dataset('test_labels', (test_rows.shape[0], num_ids), dtype='b')
else:
    set_train_seq = train_h5f.create_dataset('training_seqs', (training_rows.shape[0], temp_seq.shape[0], temp_seq.shape[1]) , dtype='ui8')
    set_test_seq = train_h5f.create_dataset('test_seqs', (test_rows.shape[0], temp_seq.shape[0], temp_seq.shape[1]), dtype='ui8')
    set_train_label = train_h5f.create_dataset('train_labels', (training_rows.shape[0], num_ids), dtype='ui8')
    set_test_label = train_h5f.create_dataset('test_labels', (test_rows.shape[0], num_ids), dtype='ui8')

# Validation
valid_h5f = h5py.File(args.save_prefix + "_validation_data.h5", 'w')
if args.store_bool:
    set_valid_seq = valid_h5f.create_dataset('validation_seqs', (valid_rows.shape[0], temp_seq.shape[0], temp_seq.shape[1]), dtype='b')
    set_valid_label = valid_h5f.create_dataset('validation_labels', (valid_rows.shape[0], num_ids), dtype='b')
else:
    set_valid_seq = valid_h5f.create_dataset('validation_seqs', (valid_rows.shape[0], temp_seq.shape[0], temp_seq.shape[1]), dtype='ui8')
    set_valid_label = valid_h5f.create_dataset('validation_labels', (valid_rows.shape[0], num_ids), dtype='ui8')

# Last run through file get, convert and store sequence
print("\nRunning through raw file again, converting sequences and store in sets ...")
with open(args.in_file, "r") as f:
    seq = []
    # make iterators
    test_i = 0
    valid_i = 0
    train_i = 0
    skip_count = 0
    for i,l in enumerate(f):
        l = l.rstrip()
        l = l.split("\t")

        # get label and sum up binary array to represent all classes
        label = l[3].split(",")  # split by commata
        label = np.array(label, dtype='i')

        # Sum up classes into binary representation
        label_bin = label_bin_init
        for j in range(len(label)):
            label_bin = label_bin + bin_look_up[label[j]]

        # get sequence
        seq = l[4]
        if args.trim_seq > 0:
           seq = seq[args.trim_seq:-args.trim_seq]
        # get first sequence length
        if i == 0:
            seq_length = len(seq)
            print("Converting and storing sequences of length %s bp." % (seq_length))
        # check sequence length matches
        if len(seq) < seq_length:
            # skip otherwise
            skip_count = skip_count + 1
            continue
        # convert to one hot coded
        seq = get_hot_coded_seq(seq)

        # match and write to respective hdf5 file
        # also store label
        if i in test_rows[:]:
            set_test_seq[test_i,] = seq
            set_test_label[test_i,] = label_bin
            # test_rows = np.delete(test_rows, test_i)
            test_i += 1
        elif i in valid_rows[:]:
            set_valid_seq[valid_i,] = seq
            set_valid_label[valid_i,] = label_bin
            # valid_rows = np.delete(valid_rows, valid_i)
            valid_i += 1
        else:
            set_train_seq[train_i,] = seq
            set_train_label[train_i,] = label_bin
            train_i += 1
        if i % 10000 == 0:
            print('Written lines ... %s' % (i))

    print("Skipped %s elements with sequence length != %s" % (skip_count, seq_length))

# Close
train_h5f.close()
valid_h5f.close()

print("\nSaved the data Data.\n")
