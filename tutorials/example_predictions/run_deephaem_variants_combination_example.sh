#!/bin/bash

date

# bedtools are available the script will call it internally

input=variants_combination_example.bed

python  /path/to/deepHaem/run_deploy_shape_combination.py --run_on cpu \
        --batch_size 10 \
	--out_dir predictions_variants_combination_out \
        --name_tag variants_combination_examples_hg38 \
        --input ${input} \
        --model /path/to/my_deephaem_model/model \
        --genome /path/to/reference_genome/hg19.fa \
        --bp_context 1000 \
        --num_classes 916 \
        --add_window 0 \
        --slize 1,2,3,4,5

date
