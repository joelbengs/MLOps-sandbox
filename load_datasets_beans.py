#load_dataset is the general loader
#load_dataset_builder is for inspection of datasets
from datasets import load_dataset, Image
from transformers import AutoFeatureExtractor
from torchvision.transforms import RandomRotation
from PIL import Image as PILImage
import io

# Dataset load with relative cache location
# The default cache directory is ~/.cache/huggingface/datasets. Change the cache location by setting the shell environment variable, HF_DATASETS_CACHE to another directory:
# $ export HF_DATASETS_CACHE="/path/to/another/directory"
dataset = load_dataset("beans", split="train", cache_dir='./datasets')
dataset[0]["image"] # would work in a notebook

rotate = RandomRotation(degrees=(0, 90))

def transforms(examples):
    examples["pixel_values"] = [rotate(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset.set_transform(transforms)
dataset[0]["pixel_values"]

# Custom function to view image from dataset using PIL and io libraries
def ViewImage(image_location):
    # Convert the image data from JpegImageFile to bytes
    image_bytes = io.BytesIO()
    image_location.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Open the image using PIL
    PILImage.open(image_bytes).show()

ViewImage(dataset[0]["image"])

# Model
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Metric
# you can change where a metric is cached with the cache_dir parameter:
# metric = load_metric('glue', 'mrpc', cache_dir="MY/CACHE/DIRECTORY")