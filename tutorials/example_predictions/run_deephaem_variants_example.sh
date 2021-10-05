#!/bin/bash

# bedtools are available the script will call it internally

date

python /path/to/deepHaem/run_deploy_net.py --dlmodel deepHaemWindow \
	--batch_size 3 \
	--out_dir ./deepHaem_variants_example_out \
	--name_tag variants_deepHaem \
	--do damage_and_scores \
	--input variants_example.vcf \
  --model /path/to/my_deephaem_model/model \
  --genome /path/to/reference_genome/hg19.fa \
	--slize '0,1,2,3,4,5' \
	--bp_context 1000 \
	--rounddecimals 5 \
	--num_classes 916 \
	--run_on cpu

echo "Finished ..."

date
