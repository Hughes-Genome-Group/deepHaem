#!/bin/bash

# location ------------------------------
SCRIPT_PATH="/path/to/deepHaem"
DATA_PATH="/path/to/deepHaem_bool_sample_split_holdout_training_data.h5"

# Architecture parameters ----------------------------
# Number of convolutional layers
conv_layers=5
# Specify the hidden units per convolutional layer in a comma separated string
hidden_units_scheme='300,600,600,900,900'
# Specify the convolutional filter widths
# per convolutional layer in a comma separated string
kernel_width_scheme='20,10,8,4,8'
# Specify the max pooling width
# per convolutional layer in a comma separated string
max_pool_scheme='3,4,5,10,4'
# Specify if an additional Fully connected layer should be used before the Final
# fully connected layer predicitng the chromatin feature classes
upstream_connected=False
# Specify how many hidden units the extra fully connected layer should have
upstream_connections=100
# specify in which datatype the data have been stored ["bool", "int8", "int32"]
stored_dtype="bool"
# number of basepairs in the DNA sequence input: use according to data processed
bp_context=1000
# specify number of chromatin feature classes included in the data compendium
# equals the number of outputs of the final fully connected layer
num_classes=5

# Training parameters ----------------------------------------------------------
# Maximum number of training epochs to run
max_epoch=30
# Size of training batches (recommended 50-100)
batch_size=100
# Shuffle the schedule by which training examples are processed for every epoch
shuffle=True
# Specify the learning rate
learning_rate=0.0001
# Specify dropout keep probabilites (1 - dropout rate) for the convolutional
# layers (inner) and the fully connected layers outer)
keep_prob_inner=0.8
keep_prob_outer=0.7
# Specify the alpha value for the L2 regularization
l2_strength=0.001
# Specify epsilon parameter for adam optimizer
epsilon=0.1
# device id of GPU to use. Training script will only use one GPU and mask
# the ones not specified if multiple cards are available
gpu=0
# specify after how may training batches the training loss should be reported
# to STDOUT
report_every=50


# Run  the training script ---------------------------

date

train_dir="./my_deephaem_model"

python ${SCRIPT_PATH}/run_training_deephaemwindow.py --train_file ${DATA_PATH} \
	--max_epoch ${max_epoch} \
	--batch_size ${batch_size} \
	--keep_prob_inner ${keep_prob_inner} \
	--keep_prob_outer ${keep_prob_outer} \
	--l2_strength ${l2_strength} \
	--shuffle ${shuffle} \
	--conv_layers ${conv_layers} \
	--hidden_units_scheme ${hidden_units_scheme} \
	--kernel_width_scheme ${kernel_width_scheme} \
	--max_pool_scheme ${max_pool_scheme} \
	--upstream_connected ${upstream_connected} \
	--upstream_connections ${upstream_connections} \
	--learning_rate ${learning_rate} \
	--epsilon ${epsilon} \
	--train_dir ${train_dir} \
	--gpu ${gpu} \
	--report_every ${report_every} \
	--stored_dtype ${stored_dtype} \
	--bp_context ${bp_context} \
	--num_classes ${num_classes}

date
