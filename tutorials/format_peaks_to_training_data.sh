#!/bin/bash

# This example script runs through the process of urning a set of chromatin feature
# peaks into a training set for deepHaem.
# It requires an installation of bedtools and perl to be available to call
# and a set of chromatin feature peaks
# For demonstration this script uses an example of 5 ENCODE DNase-seq peak calls
# subsampled to 10,000 peaks each. For creating effective training datasets we
# highly recommend creating a large and diverse set of chromatin feature data.

# Set up =======================================================================

# specify current working directory
OUTDIR='./example_data_processing'

# chrom sizes file hg19 example provided
# retrieve for example from UCSC table browser
CHROM_SIZES_FILE='../example_data/hg19_chrom_sizes.txt'

# whole genome fasta file with index (not provided as example file)
# requires whole genome fasta file and fasta index present in same diretory
WHOLE_GENOME_FASTA='/Users/ron/data_local/extra_tutorial_data/hg19.fa'

# python script to extract DNA sequence of interest
# uses pysam and requires pysam module available
PYTHON_SEQ_HELPER='../get_genome_seq_helper.py'

cd ${OUTDIR}

# 1) make 200 bp windoes (genomic bins) across the whoke genome ================
bedtools makewindows -g ${CHROM_SIZES_FILE} -w 200 -s 200 >binned_genome_hg19_200bp_200incr.bed

# 2) Intersect single peak call files with genomic bins  =======================
mkdir -p reintersects
# intersect each file with genomic windows yielding a column indicating no overlap = 0 or overlap = 1
# extracing only that column for every peak call file
find ../example_data/example_peaks/  -name "*bed" | xargs -P 2 -I % sh -c 'file=%; echo $file; name=`basename ${file} .bed`; echo $name; bedtools intersect -f 0.5 -c -a binned_genome_hg19_200bp_200incr.bed -b ${file} | cut -f 4 >./reintersects/${name}_reintersect.bed'

# 3) paste together genomic coordinates and peak call intersections ============
# create header line
echo -ne "chr\tstart\tend" >binned_genome_reoverlaps_count.bed
for i in `ls ./reintersects | grep reintersect | perl -lane '$_=~s/_reintersect.bed//g; print $_;'`;
do
  echo -ne "\t$i" >>binned_genome_reoverlaps_count.bed
done
echo -e "\n" >>binned_genome_reoverlaps_count.bed

# paste together single itnersect columns
paste `find ./reintersects/ -name '*bed'` >paste_temp

# paste together coordinates and intersect columns
paste binned_genome_hg19_200bp_200incr.bed paste_temp >>binned_genome_reoverlaps_count.bed

# 4) Filter genomic windows ====================================================
# filter out all zero entries, genomic windows with no intersecting chromatin feature peaks
cat binned_genome_reoverlaps_count.bed | perl -lane '@c = @F; splice @c, 0, 3; $co = join(",", @c); print $_ if $co=~/[^0,]+/;' >binned_genome_chrom_zero_fil.bed

# OPTIONAL: remove sex chromosomes chrX and chrY
cat binned_genome_chrom_zero_fil.bed | grep -v chrX | grep -v chrY | grep -v chrM >binned_genome_chrom_zero_fil_chr_fil.bed

# 5) Extend windows to 1000 bp where possible and collapse labels to comma separated column
tail -n +2 binned_genome_chrom_zero_fil_chr_fil.bed | perl -lane '$F[1]-=400; $F[1] = 0 if $F[1] < 0; $F[2]+=400; print "$F[0]\t$F[1]\t$F[2]\t".join(",", @F[3 .. $#F]);' >binned_genome_chrom_zero_fil_chr_fil_extend_collapse.bed

# 6) Extract reference DNA sequnce. This pthon helper script use pysam but any method to achieve this works
python3 ${PYTHON_SEQ_HELPER} --input ./binned_genome_chrom_zero_fil_chr_fil_extend_collapse.bed --genome_file ${WHOLE_GENOME_FASTA} --output ./training_instances_dataset.txt


# 7) Prepare labels files =======================================================
head -n 1 binned_genome_reoverlaps_count.bed | perl -lane 'print join("\n", @F);' | tail -n +4 | cat -n | perl -lane '$F[0]--; print "$F[0]\t$F[1]";' >labels_dataset.txt

# relevant outputs going forward are labels_dataset.txt & training_instances_dataset.txt
# the labels file should have this format
# 0       ENCFF021QCV_10k_sample
# 1       ENCFF022BIA_10k_sample
# 2       ENCFF037AJZ_10k_sample
# 3       ENCFF042RGX_10k_sample
# 4       ENCFF102CCA_10k_sample

# the training instances file should have this format
# chr1    633400  634400  1,0,0,0,0      TACT...
# chr1    633600  634600  1,0,0,0,0      TAGA...
# chr1    830800  831800  1,0,0,0,0       AGGA...

# 8) CLEAN UP ==================================================================
# rm -rf binned_* ./reintersects paste*
