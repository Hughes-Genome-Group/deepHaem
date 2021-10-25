'''
Take a count martix of format chr\tstart\tend\tlabels etc. with an optional header line scaned for "#" at start
From the provided genomefile extract the DNA sequence per window
Print new file with chr start end label seq in traindata.txt
usage: python make_coord_label_seq_from_coord_countmatrix.pl <whole_genome_fasta> <input_countmatrix.txt> <output_traindata> <output_label_number>
'''

import argparse
import sys
import pysam
import re

# Define arguments -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="""Take a count martix of format chr\tstart\tend\toverlapLabelA\toverlapLabelB etc. with an optional header Line scane for "start" keyword
    take labels from header if present and replace by numbers --> print number to label table
    From the provided genomefile extract the DNA sequence per window
    Collapse the numeric lables and print new file with chr start end label seq in traindata.txt""")
parser.add_argument('--input', type=str, dest='inputfile',
                    help='Tab separated file of format chr\tstart\tend\toverlapLabelA\toverlapLabelB')
parser.add_argument('--genome_file', type=str, dest='genome_file',
    help='Whole genome fasta file with .fai present.')
parser.add_argument('--output', type=str, dest='output_traindata', default='instances.data.txt',
                    help='Destination forinstances of the data set output.')
args = parser.parse_args()

# open input
with open(args.inputfile, "r") as fr:
    with open(args.output_traindata, "w") as fw:
        with pysam.Fastafile(args.genome_file) as fa:

            for line in fr:
                # handle header lines
                if re.match('^#', line):
                    print('handeling header line')
                    fw.write(line.rstrip() + '\tsequence\n')
                    continue

                chrom, start, end, labels = line.split()
                start = int(start)
                end = int(end)

                # split and format labels
                labels = labels.split(',')
                conc_labels = ""
                m = 0
                for l in labels:
                    l = int(l)
                    if l >= 1:
                        conc_labels = conc_labels + ',' + str(m)
                    m = m + 1
                # remove first comma
                conc_labels = conc_labels.strip(',')

                # get DNA sequence
                seq = fa.fetch(reference=chrom, start=start, end=end).upper()
                # print all
                fw.write('%s\t%s\t%s\t%s\t%s\t\n' % (chrom, start, end, conc_labels, seq))
