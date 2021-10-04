#!/bin/bash

# This example script runs through the process of urning a set of chromatin feature
# peaks into a training set for deepHaem.
# It requires an installation of bedtools to be available to call
# and a set of chromatin feature peaks
# For demonstration this script uses an example of 5 ENCODE DNase-seq peak calls
# subsampled to 10,000 peaks each. For creating effective training datasets we
# highly recommend creating a large and diverse set of chromatin feature data.

# Set up =======================================================================

# specify current working directory
OUTDIR='.'

# chrom sizes file hg19 example provided
# retrieve for example from UCSC table browser
CHROM_SIZES_FILE='./example_data/hg19_chrom_sizes.txt'

cd ${OUTDIR}

# 1) make 200 bp windoes (genomic bins) across the whoke genome ================
bedtools makewindows -g /t1-data/user/hugheslab/rschwess/database/chrom_sizes/hg19_chrom_sizes.txt -w 200 -s 200 >binned_genome_hg19_200bp_200incr.bed

## 2) Intersect single files with bins
#mkdir -p reintersects

#find ../bed_links/heart_beds/  -name "*bed" | xargs -P 4 -I % sh -c 'name=`basename % .bed`; echo $name; bedtools intersect -f 0.5 -c -a binned_genome_hg19_200bp_200incr.bed -b % | cut -f 4 >./reintersects_heart/${name}_reintersect.bed'


# paste together
echo  -ne "chr\tstart\tend" >binned_genome_heart_reoverlaps_count.bed
for i in `ls ./reintersects_heart | grep reintersect | perl -ne '$_=~s/_reintersect.bed//g; print $_;'`; do echo -ne "\t$i" >>binned_genome_heart_reoverlaps_count.bed; done
echo -ne "\n" >>binned_genome_heart_reoverlaps_count.bed


paste `find ./reintersects_heart/ -name '*bed'` >paste_heart_temp1

paste binned_genome_hg19_200bp_200incr.bed paste_heart_temp1 >>binned_genome_heart_reoverlaps_count.bed


# prepare labels files
#head -n 1 binned_genome_full_reoverlaps_count.bed | perl -lane 'print join("\n", @F);' | perl -lane '$_=~s/wgEncodeAwg//g; $_=~s/\.narrowPeak.gz//g; print $_;' | tail -n +4 | cat -n >labels_extra30_full.txt
head -n 1 binned_genome_heart_reoverlaps_count.bed | perl -lane 'print join("\n", @F);' | perl -lane '$_=~s/wgEncodeAwg//g; $_=~s/\.narrowPeak.gz//g; print $_;' | tail -n +4 | cat -n >labels_heart.txt

# FILTER ==================================================================================
# remove chrX and chrY
cat binned_genome_heart_reoverlaps_count.bed | grep -v chrX | grep -v chrY | grep -v chrM >binned_genome_heart_reoverlaps_count_chrfiltered.bed

# filter out all zero entries
cat binned_genome_heart_reoverlaps_count_chrfiltered.bed | perl -lane '@c = @F; splice @c, 0, 3; $co = join(",", @c); print $_ if $co=~/[^0,]+/;' >binned_genome_heart_chrom_zero_fil.bed

# extend windows to 1000 bp where possible, collapse classes to commas, extract sequence
tail -n +2 binned_genome_heart_chrom_zero_fil.bed | perl -lane '$F[1]-=400; $F[1] = 0 if $F[1] < 0; $F[2]+=400; print join("\t", @F);' | perl /t1-data/user/hugheslab/rschwess/scripts/utility/machine_learning_related/make_coord_label_seq_from_coord_countmatrix.pl hg19 - training.data.heart.txt


# -------------

# EXTRA FILTER STEP: Get onyl bins intersecting with a "original TF" bin from DeepSEA or a WIMM data set.
# prepare TF deepsea data
# cat allTFs.pos.trimmed.bed | sort -k1,1 -k2,2n | bedtools merge -i - > allTFs.pos.trimmed.sorted.merged.bed
# prepare all DNase and TF data from WIMM and ENCODE
# Get a File from WIMM data and Deepsea TF AND DNase bed files for filtering
# zcat `find /t1-data/user/hugheslab/rschwess/machine_learning/deepHaem/data/bed_links/deepsea_used/ -name '*Tfbs*'` `find /t1-data/user/hugheslab/rschwess/machine_learning/deepHaem/data/bed_links/deepsea_used/ -name '*Dnase*'` | cut -f 1,2,3 | sort -k1,1 -k2,2n | bedtools merge -d 10 -i - >union_deepsea_tf_and_dnase_combined_merged.bed
# make a combined bed file for filtering
# cat union_deepsea_tf_and_dnase_combined_merged.bed union_wimm_combined_elements.bed | sort -k1,1 -k2,2n | bedtools merge -i - >temp_for_post_binning_filtering_with_alldnase.bed
# cat allTFs.pos.trimmed.bed union_wimm_combined_elements.bed | sort -k1,1 -k2,2n | bedtools merge -i - >temp_for_post_binning_filtering_with_deepseatf_wimmall.bed

# filter
# tail -n +2 binned_genome_full_reoverlaps_count_chrfiltered.bed | bedtools intersect -u -a - -b temp_for_post_binning_filtering_with_alldnase.bed >binned_genome_full_reoverlaps_count_selected.bed
# tail -n +2 binned_genome_full_reoverlaps_count_chrfiltered.bed | bedtools intersect -u -a - -b temp_for_post_binning_filtering_with_deepseatf_wimmall.bed >binned_genome_full_tfandwimmonly_reoverlaps_count_selected.bed

# filter out all zero entries
#cat binned_genome_full_chrom_count_chrfiltered.bed | perl -lane '@c = @F; splice @c, 0, 3; $co = join(",", @c); print $_ if $co=~/[^0]+/;' >binned_genome_full_chrom_zero_fil.bed

# extend windows to 1000 bp where possible, collapse classes to commas, extract sequence
#cat binned_genome_full_chrom_zero_fil.bed | perl -lane '$F[1]-=400; $F[1] = 0 if $F[1] < 0; $F[2]+=400; print join("\t", @F);' | perl ../../../../scripts/utility/machine_learning_related/make_coord_label_seq_from_coord_countmatrix.pl hg19 - training.data.full.txt

# cat binned_genome_full_tfandwimmonly_reoverlaps_count_selected.bed | perl -lane '$F[1]-=400; $F[1] = 0 if $F[1] < 0; $F[2]+=400; print join("\t", @F);' | perl ../../../../scripts/utility/machine_learning_related/make_coord_label_seq_from_coord_countmatrix.pl hg19 - training.data.full.tfandwimmonly.txt

## Get Reverse complement of training data to feed
#cat training.data.full.tfandwimmonly.txt | perl -lane '$F[4] = reverse $F[4]; $F[4]=~s/A/B/g; $F[4]=~s/T/A/g; $F[4]=~s/B/T/g; $F[4]=~s/C/D/g; $F[4]=~s/G/C/g; $F[4]=~s/D/G/g; print join("\t", @F);' >reverse_temp
#cat training.data.full.tfandwimmonly.txt reverse_temp >training.data.full.tfandwimmonly.revcomp.txt
#rm -f reverse_temp




# handle that one with test chr7 and validation 8/9
## make python hdf5 training data
# qsub prepare_h5py_data.sh
