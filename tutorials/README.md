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

[./example_data_processing/](./example_data_processing/) contains examples of how these files should look after running the formatting (only first 5 lines for instances file).  
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

### Requirements

* Python 3.5+ with the folowing packages installed
  * Tensorflow v1.8.0+ with GPU support
  * numpy
  * h5py
 * Training new models requires an available GPU

* training h5 file from data pre-processing step

### Workflow

Run [../run_training_deephaemwindow.py](../run_training_deephaemwindow.py) pointing to the pre-processed training data and specifying the desired network architecture and training parameters. The new model will be trained in a new specified directory. For inspecting training progress use tensorboard.
```
tensorboard -logdir=./my_new_model_dir
```
Monitor the training loss and test loss over epochs.  

Training time depends on the specified architecture such as number of ayers, hidden units, sequence input size, number of chromatin classes and training instances provided. For model comparable to the provided ones, expect to train the model overnight 8 - 12 hrs after which inspection of the training and validation error gives useful insights on how the training is progressing. The majority of training progress is usually achieved in the first 24 hrs or training. Final models can be trained for longer. The training script only saves models that have a lower test set error ten previous ones.   

An example bash script to run model training is provided in [./example_model_training](./example_model_training).


### Evaluating a trained model

To evaluate a trained model on the validation data (or again on the test or training data) use [../run]](./example_model_training).

```
python /path/to/deepHaem/run_test_accuracy.py --test_on 'valid' \
  --test_file /scratch/ron/deepHaem_bool_chr_split_holdout_validation_data.h5 \
  --model ./best_checkpoint-276670 \
  --graph ./best_checkpoint-276670.meta \
  --test_dir valid_data_out \
  --name_tag eval_ery_only \
  --savetxt 'True' \
  --roc 'True' \
  --prc 'True' \
  --num_classes 4384 \
  --slize 0,1,3,4,5,6,7,8,9,10
```

Arguments:

| Argument |  Default | Description |
| ------------------ |:----------:| :-----------------|
| --dlmodel  | deepHaemWindow | Specifcy the DL model file to use e.g. <deepHaemWindow>.py |
| --test_file | None | Input Training and Test Sequences and Labels in hdf5 format "test_seqs", "test_labels" labeled data. Or the validation file labels validation_seqs etc.|
| --test_on' | test | Either "test" or "valid": select if to test accuracy on the test or validation set |
| --model | None | Full path to checkpoint of model to be tested |
| --graph | None | Full path to .meta graph file of the model to be tested |
| --batch_size | 100 | Batch size to process instances |
| --test_dir | test_data_out | Name or full path to directory to store the test data output |
| --name_tag | eval | 'Nametag to add to filenames |
| --slize| all | 'Comma separated list of start and end position 0.based indexed of chromatin feature classifiers (outputs) to evaluate. e.g. '0,1,2,3,5,6' Use labels file from data preprocessing to select ids. Default 'all' will evaluate on all classifiers.')
| --only | 0 | Set number of first lines to use for testing (if 0 will do all). |
| --savetxt | False | Select if to store scores and labels as txt files for use downstream. |
| --roc | False | Calculate and plot ROCurves per classifier True/False |
| --prc | False | Calculate and plot PrecisionRecallCurves per classifier True/False |
| --bp_context | 1000 | Specify number of basepairs of DNA sequence input. |
| --num_classes | 919 | Specify the number of chromatin features (output neurons). |
| --run_on | gpu | Select where to run on 'cpu' or 'gpu' (if available). |
| --gpu | 0 | Select device id of a single available GPU and mask the rest. |
| --roc_auc | False | Define if to print ROC AUC values False/True. |
| --prc_auc | False | Define if to print PRC AUC values False/True. |

# Making predictions

Three scripts are provided to serve different prediction needs.
All require bedtools to be available and the python modules pysam and pybedtools.
The scripts also need access to a reference genome matching the genomic
coordinates provided in the input files. An .fai index of the same name needs to
present in the reference genome directory.

Example commands and example input files are provided in
[./example_predictions](./example_predictions). Check out the note on variant
formatting in there as well

* [../run_deploy_net.py](../run_deploy_net.py) Deploy a trained model to predict
 the class a sequence (bed coordinates and reference genome link)
 belongs to or predict the impact of a sequence variants provided in vcf-like
 format with minimum columns: (chr pos id ref_base var_base).
 Can also be used to predict chromatin feature classes over regions
 provided in bed like format minimum columns: (chr start end). **Note** that bed
 input assumes 0.based, half open bed format genomic coordinates and vcf format
 assumes 1.based.

* [../run_deploy_shape_combination.py](../run_deploy_shape_combination.py)
Predict impact of variants on the chromatin feature classes. This version will
apply all provided variants to the same sequence for example a combination of
multiple SNPs and Indels. Requires a bed-like file input with the following
columns: (chr start end sequence) where the 4th column 'sequence' specifies
which variants to apply to the genomic window specified by chr start end
(in 0.based bed index). You can supply **DNA sequence bases** in upper
case, **.** to indicate deletions or **reference** to indicate that the reference
DNA sequence from the supplied reference genome should be used. This can be
useful for delineating the DNA sequence to predict over.

* [../run_deploy_seq_only.py](../run_deploy_seq_only.py) is a slightly optimized
version to predict the chromatinfeatures over reference DNA sequence only given
bed regions as input. Supply a bed like file with minimum columns: (chr start end).
Alternative input can be a fasta file to produce class scores per supplied DNA sequence.


# Additional Scripts

### Saliency

To calculate saliency scores use [../run_saliency.py](../run_saliency.py).

In addition to a trained model this script requires the python module pysam
available to load and a human reference genome. Supply genomic regions in bed
format and select a single chromatin feature classifier (0.based.index from
labels file) to calculate the saliency over the specfid regions.
The script needs to rebuild the network architecture so supply the architecture
parameters used to train the respective model.

```
python  /stopgap/fgenomics/rschwess/scripts/epigenome_nets/deepHaem/run_saliency.py --run_on cpu \
        --gpu 0 \
        --gradient_input 'sigmoid' \
        --select 7 \
        --batch_size 3 \
        --out_dir saliency_out_examplet \
        --name_tag saliency_example \
        --input /path/to/my_example_regions.bed \
        --model /path/to/my_deephaem_model/model \
        --genome /path/to/hg19.fa \
        --rounddecimals 5 \
        --conv_layers 5 \
        --hidden_units_scheme '1000,1000,1000,1000,1000' \
        --kernel_width_scheme '20,10,8,4,8' \
        --max_pool_scheme '3,4,5,10,4' \
        --upstream_connected True \
        --upstream_connections 100 \
        --bp_context 1000 \
        --num_classes 4384 \
```

Extra arguments:

| Argument |  Default | Description |
| ------------------ |:----------:| :-----------------|
| --select | 0 | Index of chromatin feature classifier (0.based.index if output neurons) to use for the saliency calculation. Only a single classifier per run is supported. |
 validation file labels validation_seqs etc.|
| --gradient_input | sigmoid | Select "sigmoid" or "logit" (score before sigmoid transformation) relative to which to calculate the saliency score. |
| --rounddecimals | 10 | Select the number of decimal places to round saliency scores to. |
| --model | None | Full path to checkpoint of model to be tested |
| --batch_size | 3 | Batch size to process the supplied regions split up into instances. |
| --out_dir | predictions_dir | Name or full path to directory to store the test data output. |
| --name_tag | pred | 'Nametag to add to filenames |
| --bp_context | 1000 | Specify number of basepairs of DNA sequence input. |
| --num_classes | 919 | Specify the number of chromatin features (output neurons). |
| --run_on | gpu | Select where to run on 'cpu' or 'gpu' (if available). |
| --gpu | 0 | Select device id of a single available GPU and mask the rest. |
| --stored_dtype | float32 | Indicate what data format sequence and labels where stored in. ["bool", "int8", "int32", ...] |


### Save convolutional weigths

Use [../run_save_weights.py](../run_save_weights.py) as example script to save
convolutional filter weights as numpy arrays for plotting or transfer learning.
