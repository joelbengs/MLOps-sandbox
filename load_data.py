#load_dataset is the general loader
#load_dataset_builder is for inspection of datasets
from datasets import load_dataset, Image, load_dataset_builder, get_dataset_split_names, get_dataset_config_names
from transformers import AutoFeatureExtractor

#Specify dataset from hugginface hub
url_to_dataset = "zh-plus/tiny-imagenet"
split = "valid"

#Inspect the dataset
#ds_builder = load_dataset_builder(url_to_dataset)
#print("Available splits are: ", get_dataset_split_names(url_to_dataset))
#print("Available confgurations are: ", get_dataset_config_names(url_to_dataset))
#print("Description (if any): ", ds_builder.info.description)
#print("Features: ", ds_builder.info.features)

#Load the dataset into a Dataset object (not an IterableDataset)
ds = load_dataset(url_to_dataset, split=split, cache_dir="./datasets")

