# deepHaem
Implementation of a deep convolutional neuronal network for predicting chromatin features from DNA sequence.

The repository contains a flexible tensorflow implementation of a convolutional neuronal network with max pooling. The basic architecture is built on the principles described in DeepSEA (http://deepsea.princeton.edu/help/) and Basset (https://github.com/davek44/Basset). It is comprised of multiple layers of convolutional filters followed by ReLU and max pooling and a fully connected layer in the end. An additional pre-final fully connected layer can be switch on as well. The number of convolutional layers is flexible and selected as hyperparameter in the beginning of the training procedure. Batch normalization is optional.


### Contents

* **deepHaemWindow.py** : Model file, implementation of the model. Flexible construction according to hyperparameters provided.
* **run_deploy_net.py** : Deploy a trained model to predict the class a sequence (bed coordinates and reference genome link) belongs to or predict the impact of a sequence variant provided in vcf format.
* **run_deploy_seq_only.py** : Deploy the network to predict the chroamtin feature classes associated with a sequence. Optimized for running on a longer sequence only, no variants.
* **run_deploy_shape_combination.py** : Predict impact of variants on the chromatin feature classes. This version will aplly all provided variants to the same sequence. E.g. combination of multiple SNPs, Indels, SVs.
* **run_save_weigths.py** : Helper script to store the learned filter weigths in a numpy array
* run_test_accuracy.py** : Test prediction accuracy on test, validaiotn or novel data set. Allows to extract only the predictions of selected chromatin classes. Make plots and optional save labels and predicitons for visualizing in a different programme.
* **run_training_deephaemwindow.py** : train a deepHaem model  using the hyperparameters (architecture specifications and learning process parameters) provided.

### Requirements

* Python 3.5 +
* Tensorflow v1.8.0 +
* Pre-processed chromatin feature data

### Data

The basic data format is DNA sequences each assoicated with a set of chromatin features. DeepHaem require training, test and validation data stored as tensors in numpy arrays. Each set consists of the one-hot encoded sequences a 3D tensor of dimensitons (num_examples, seq_length, 4) and the labels a 2D tensor of 1's and 0's representing the chromatin features a given sequence is associated with dimensions (num_examples, num_of_chrom_features). We recommend storing data in hdf5 format. The training script provided reads the data from a hdf5 file and expects training and test set to be stored in the same file, while the validation data is provided in a separate file. Sequences and labels are expected as "training_seqs", "training_labels", "test_seqs", "test_labels", "validation_seqs" and "validation_labels"; entries in the hdf5 file. The training script is straight forward to adjust for reading the data in a different format. For saving space sequences and labels can be stored as unsigned integers or boolean values. Parse the respective data type used to the training script for translation.

The training (and test and validation data) used in the DeepSEA publication are the best starting point. They are similarly formated to the format required for running deepHaem (http://deepsea.princeton.edu/help/). Basset has another publcily available set comprising over 600 open chromatin assays across cell types (https://github.com/davek44/Basset). The input is 600 bp per site and the data pre-processing is peak based rather then windowed bin based (DeepSEA).

For how to create your own data set, refer to the workflow used in DeepSEA or Basset. We will add pointer to processing your own data in the future. Once you have a bed like file listing chromosome start, chromatin feature classes and the sequence (see example: ./data_preprocessing/example_training_set_format_for_processing.txt) you can use https://github.com/rschwess/RonsUtilityBox/blob/master/machine_learning_related/make_training_data.py to split the data into training, test and validation set and store it as numpy arrays in hdf5 format.

### Models
**./models** contains links to already trained models.
