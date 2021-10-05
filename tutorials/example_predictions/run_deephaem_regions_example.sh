#!/bin/bash
date

input=regions_examples.bed

# bedtools are available the script will call it internally

python  /path/to/deepHaem/run_deploy_net.py --run_on cpu \
        --gpu 0 \
        --batch_size 10 \
	      --out_dir predictions_regions_out \
        --name_tag regions_example \
        --input ${input} \
        --model /path/to/my_deephaem_model/model \
        --genome /path/to/reference_genome/hg19.fa \
        --do class \
        --rounddecimals 5 \
        --bp_context 1000 \
        --num_classes 916 \
        --slize 1,2,3,4,5

date
