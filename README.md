# deepHaem
Implementation of a deep convolutional neuronal network for predicting chromatin features from DNA sequence.

The repository contains a flexible tensorflow implementation of a convolutional neuronal network with max pooling. The basic architecture is built on the principles described in DeepSEA (http://deepsea.princeton.edu/help/) and Basset (https://github.com/davek44/Basset). It is comprised of multiple layers of convolutional filters followed by ReLU and max pooling and a fully connected layer in the end. An additional pre-final fully connected layer can be switch on as well. The number of convolutional layers is flexible and selected as hyperparameter in the beginning of the training procedure. Batch normalization is optional.


### Contents

* **deepHaemWindow.py** : Model file, implementation of the model. Flexible construction according to hyperparameters provided.
* **run_deploy_net.py** : Deploy a trained model to predict the class a sequence (bed coordinates and reference genome link) belongs to or predict the impact of a sequence variant provided in vcf format.
* **run_deploy_seq_only.py** : Deploy the network to predict the chromatin feature classes associated with a sequence. Optimized for running on a longer sequence only, no variants.
* **run_deploy_shape_combination.py** : Predict impact of variants on the chromatin feature classes. This version will apply all provided variants to the same sequence. E.g. combination of multiple SNPs, Indels, SVs.
* **run_saliency.py** : Calulate saliency scores over provided bed regions, selecting a single chromatin feature class (output neuron).
* **run_save_weigths.py** : Helper script to store the learned convolutional filter weigths in a numpy array.
* **run_test_accuracy.py** : Test prediction accuracy on test, validation or novel data set. Allows to extract only the predictions of selected chromatin classes. Make plots and optional save labels and predicitons for visualizing in a different programme.
* **run_training_deephaemwindow.py** : train a deepHaem model  using the hyperparameters (architecture specifications and learning process parameters) provided.

### Requirements

* Python 3.5+ with the following packages installed
  * Tensorflow v1.8.0+
  * numpy
  * h5py
  * pysam
  * pybedtools (and bedtools installed and available)
* Utitlity scripts require bedtools to be installed


### Models
[./models] contains links to trained models.

### Data

To train new models you will need chromatin feature data as peak calls. In addition to cell types and assays of bespoke interest we highly recommend training models with a large data compendium. A good reference are the compendia used in [DeepSEA](http://deepsea.princeton.edu/help/) or [Basset](https://github.com/davek44/Basset). Users should compile a training dataset from a large compendium and their bespoke data and train a model for the whole set.

### Workflow

Example workflows for formatting data, training a model and making predictions are outline under [./tutorials](./tutorials).
