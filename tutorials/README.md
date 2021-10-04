# Formatting data for training with deepHaem

### Requirements

* bedtools
* python 3.5+ with the following packages available
  * numpy
  * h5py
  * pysam (used to extract reference DNA sequence, can be achieved by samtools or similar)
* command line perl calls can be replaced by awk or similar

Core input is a collection of peak call of chromatin data (e.g DNase-seq, ATAC-seq, ChIP-seq).
For demonstration we provide a set of subsampled open chromatin peaks from ENCODE under [./example_data/example_peaks](./exmaple_data/example_peaks). These contain a random sample of 10,000 peaks.
For training deepHaem models we recommend a much larger and more diverse compendium of chromatin feature data.
Rather than training a very bespoke model on a handful of chromatin feature data, we recommend adding
any user peak data to a base compendium of chromatin feature data. A good reference are the compendia used in [DeepSEA](http://deepsea.princeton.edu/help/) or [Basset](https://github.com/davek44/Basset).

In addition a chromosome sizes file specifying the dimensions of each available
chromosome. These fields can be downloaded from the UCSC table browser or derived using ucsctools.
This repository contains an example of a genome sizes file for [hg19](./example_data/hg19_chrom_sizes.txt).

### Workflow

#### 1) Assemble chromatin feature peaks

Download and format all peak files as necessary. For demonstration, use the [./example_data/example_peaks](./exmaple_data/example_peaks) provided. The provided processing bash script assumes a three+ column, tab-separated bed
format specifying (chromosome start stop) in 0-based coordinates.
Additional columns are ignored.

#### 2) Create a plain text dataset

These steps take the peak calls and create a plain text dataset, effectively regions and the underlying DNA sequence with the provided chromatin features. There are multiple ways of creating such a dataset.
We generally follow the process used in [DeepSEA](http://deepsea.princeton.edu/help/). First he genome is split into 200 bp windows. Then these genomic windows are intersected with all peak files. For every 200 bp window, every chromatin feature peak that overlaps the window by at least 50 % is attributed to that window. Genomic windows that have no peak overlaps are discarded. Finally the genomic windows are extended by 400 bp to either site giving a 1 kb input sequence that is characterised by what chromatin features are overlapping with its central 200 bp.  

The bash script [format_peaks_to_training_data.sh](./format_peaks_to_training_data.sh) walks through the
formatting process, deepSEA style, using the example peaks. For processing your own data adapt the directories and
potentially the columns extracted if using different peak formats.

The primary output is a plain text file training_instances_dataset.txt contains all genomic windows with at least one chromatin feature overlap. The file is tab separated in the format (chr start end chromatin_feature_ids, DNA_sequence). The chromatin features associated are listed as 0-based indexes in a single comma separated column.
The secondary output file labels_dataset.txt links labels to chromatin features.

[./example_data_processing/](./example_data_processing/) contains examples of how these files should look after running the formatting (only first 5 lies for instances file).  
[./example_data/example_training_set_format_for_processing.txt](./example_data/example_training_set_format_for_processing.txt) gives an example of how the first lines of that file would look like for a bigger dataset.

#### 3) Split up training dataset and store in hdf5 files

The python script [make_training_data_bool_avail.py](./make_training_data_bool_avail.py) uses as input the plain text dataset from step 2), splits up training, test and validation set and stores them in hdf5 format for the deepHaem scripts. Each line in the plain text dataset constitutes an instance. Note that we used the convention where test data are evaluated after every training epoch and validation data are hold out during the training and evaluation process and kept for validating the final models. Two split-modes can be specified either sampling the test and validation instances given specified fractions or by holding out entire specified chromosomes for the test and validation set. Practically, we observed models that generalise better when using hold out chromosomes.

Output are 2x .h5 files and 3x coordinate files. The coordinate files detail which genomic windows are used in which set.
The h5 files store the chromatin feature labels and DNA sequences. The training_set h5 file contains the training and test data. The validation_set h5 file holds the validation data.

Every sequence is stored as sequence_length x 4 with hot encoding for the DNA bases yielding a tensor of [instances x sequence_length x 4].
Labels are stored as on hot encoded labels yielding a vector of [instances x number_of_classes]

Example how to run the training set creation in hold out chromosome mode:
Note "store_bool" set to True will store sequence representation and labels as boolean values to save storage.
```
python ./make_training_data_bool_avail.py --seed 1234 \
  --split_mode chr \
  --chr_test chr11 chr12 \
  --chr_valid chr15 chr16 chr17 \
  --save_prefix ./example_data_processing/deepHaem_bool_chr1112151617_holdout \
  --num_classes 5 \
  --store_bool 'True' \
  ./example_data_processing/training.instances.dataset.txt
```

Example how to run the training set creation when sampling test and validation set:

```
python ./make_training_data_bool_avail.py --seed 1234 \
  --split_mode random \
  --frac_test 0.2 \
  --frac_valid 0.2 \
  --save_prefix ./example_data_processing/deepHaem_bool_sample_split_holdout \
  --num_classes 5 \
  --store_bool 'True' \
  ./example_data_processing/training.instances.dataset.txt
```


# Training a new model

ToDo

# Making predictions

ToDo
