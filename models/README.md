## 1) deepHaem model for DeepSEA dataset

* Model trained with the deepHaem implementation and architecture modifications.
* 5 Convolutional layers with ReLU and max pooling
* More pooling to reduce the size of the final fully connected layer.
* download link: http://datashare.molbiol.ox.ac.uk/public/rschwess/deepHaem/model_deephaem_endpool_deepsea_model.tar.gz


## 2) deepHaem model using DeepSEA data compendium supplemented by erythroid data
* This is the model used for transfer learning in [deepC](https://github.com/rschwess/deepC).
* Model trained with the deepHaem implementation and architecture modifications.
* 5 Convolutional layers with ReLU and max pooling
* More pooling to reduce the size of the final fully connected layer.
* download link: http://datashare.molbiol.ox.ac.uk/public/rschwess/deepHaem/model_deephaem_endpool_erythroid_model.tar.gz
* download archive includes the saved model weights for all convolutional filters
* full [list](http://datashare.molbiol.ox.ac.uk/public/rschwess/deepHaem/table_for_github_deephaem_dataset_labels_with_peaks.xlsx) of data and peak calls used with download link

## 3) deepHaem mouse model using ENCODE data compendium
* This is the model sued for transfer learning in [deepC](https://github.com/rschwess/deepC).
* Model trained with the deepHaem implementation and architecture modifications.
* 5 Convolutional layers with ReLU and max pooling
* More pooling to reduce the size of the final fully connected layer.
* download link: http://datashare.molbiol.ox.ac.uk/public/rschwess/deepHaem/model_deephaem_endpool_mouse_encode.tar.gz
* download archive includes the saved model weights for all convolutional filters

 ## 4) deepHaem human model using a larger ENCODE data compendium supplemented by selected erythroid and immune cell data

* total of 4384 chromatin features used
* 5 Convolutional layers with ReLU and max pooling, 1000 hidden units per layer
* 2 Fully Connected layers, second to alst layer withh 100 hidden units
* More pooling to reduce the size of the fully connected layers.
* download link: http://datashare.molbiol.ox.ac.uk/public/rschwess/deepHaem/model_deephaem_endpool_4k_data_model.tar.gz
* full [list](http://datashare.molbiol.ox.ac.uk/public/rschwess/deepHaem/model_deephaem_endpool_4k_data_model/labels_4k_full_chrom_with_descriptions_curated.xlsx) of data and peak calls used with download link
