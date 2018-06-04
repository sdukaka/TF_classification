import tensorflow as tf
from datasets import flowers

slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', "/wuwei/TF_classification/data")

# Creates a TF-slim DataProvider which reads the dataset in the background
# during both training and testing
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])