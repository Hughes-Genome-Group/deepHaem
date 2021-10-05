### Note on variant formats

Use **.** to indicate deletions or encode indels from the perspective of the
first left unchanged base pair: e.g. **CATC** **C** for a **ATC** deletion.
**Note** that depending on the source of your variants InDels might be encoded
differently e.g. via **-**. **reference** can be used to indicate that no
alteration is to be made to the
sequence retrieved from the reference genome.


### Prediction scripts arguments


#### [run_deploy_net.py](../../run_deploy_net.py)

| Argument |  Default | Description |
| -------- |:----------:| :-----------------|
| --dlmodel  | deepHaemWindow | Specifcy the DL model file to use e.g. <deepHaemWindow>.py |
| --batch_size | 10 | Batch size to process instances. |
| --out_dir | predictions_dir | Name or full path to directory to store the predictions. |
| --name_tag | eval | 'Nametag to add to filenames |
| --do | class | Select what to do: predict the 'class' of a sequence region or the 'damage' score per class (whenn variants provided); 'damage_and_scores' will report the damages as well as the raw scores for reference and variant |
| --slize | all | 'Comma separated list of start and end position 0.based indexed of chromatin feature classifiers (outputs) to use for the predictions e.g. '0,1,2,3,5,6' Use labels file from data preprocessing to select ids. Default 'all' will predict using all classifiers. Technically, all classifiers are used for the prediction and slize is just used to crop the output to the desired columns so this does not increase speed but saves output space. |
| --rounddecimals | 10 | Select the number of decimal places to round probability and damage scores to. |
| --input | None | Input region or variant file (bed/vcf) like. |
| --model | None | Full path to checkpoint of model to be tested. |
| --genome | None | Full path to refrence genome used to extract DNA sequence of the regions o interest. |
| --bp_context | 1000 | Specify number of basepairs of DNA sequence input. |
| --num_classes | 919 | Specify the number of chromatin features (output neurons). |
| --run_on | gpu | Select where to run on 'cpu' or 'gpu' (if available). |
| --gpu | 0 | Select device id of a single available GPU and mask the rest. |

---

#### [run_deploy_shape_combination.py](../../run_deploy_shape_combination.py)

| Argument |  Default | Description |
| -------- |:----------:| :-----------------|
| --dlmodel  | deepHaemWindow | Specifcy the DL model file to use e.g. <deepHaemWindow>.py |
| --batch_size | 1 | Batch size to process instances. |
| --out_dir | predictions_dir | Name or full path to directory to store the predictions. |
| --name_tag | eval | 'Nametag to add to filenames |
| --slize | all | 'Comma separated list of start and end position 0.based indexed of chromatin feature classifiers (outputs) to use for the predictions e.g. '0,1,2,3,5,6' Use labels file from data preprocessing to select ids. Default 'all' will predict using all classifiers. Technically, all classifiers are used for the prediction and slize is just used to crop the output to the desired columns so this does not increase speed but saves output space. |
| --input | None | Input region or variant file (bed/vcf) like. |
| --model | None | Full path to checkpoint of model to be tested. |
| --genome | None | Full path to reference genome used to extract DNA sequence of the regions o interest. |
| --bp_context | 1000 | Specify number of basepairs of DNA sequence input. |
| --num_classes | 919 | Specify the number of chromatin features (output neurons). |
| --padd_ends | none | Specify if to padd with half times bp_context N's to make predictions over chromosome ends [left, right, none, both]. |
| --add_window | 0 | Basepairs to add around variants of interest supplied for prediction and visualisation later. |
| --bin_size | 1 | Bin sizes (strides over the genomic window) to use running over the new sequence. Default is 1 but can be used to run predictions over windows with larger increments to reduce computation and output. |
| --run_on | gpu | Select where to run on 'cpu' or 'gpu' (if available). |
| --gpu | 0 | Select device id of a single available GPU and mask the rest. |

-----

#### [run_deploy_seq_only.py](../../run_deploy_seq_only.py)

| Argument |  Default | Description |
| ------------- |:----------:| :-------------|
| --dlmodel  | deepHaemWindow | Specifcy the DL model file to use e.g. <deepHaemWindow>.py |
| --batch_size | 10 | Batch size to process instances. |
| --out_dir | predictions_dir | Name or full path to directory to store the predictions. |
| --name_tag | eval | 'Nametag to add to filenames |
| --do | class | Select what to do default: predict the 'class' of a sequence or 'damage' per class or get class scores persequence (fasta file input): 'seq'.)
| --slize | all | 'Comma separated list of start and end position 0.based indexed of chromatin feature classifiers (outputs) to use for the predictions e.g. '0,1,2,3,5,6' Use labels file from data preprocessing to select ids. Default 'all' will predict using all classifiers. Technically, all classifiers are used for the prediction and slize is just used to crop the output to the desired columns so this does not increase speed but saves output space. |
| --input | None | Input region or variant file (bed/vcf) like. |
| --model | None | Full path to checkpoint of model to be tested. |
| --genome | None | Full path to reference genome used to extract DNA sequence of the regions o interest. |
| --bp_context | 1000 | Specify number of basepairs of DNA sequence input. |
| --num_classes | 919 | Specify the number of chromatin features (output neurons). |
| --run_on | gpu | Select where to run on 'cpu' or 'gpu' (if available). |
| --gpu | 0 | Select device id of a single available GPU and mask the rest. |
