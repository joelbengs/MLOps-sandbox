from datasets import load_dataset
import evaluate
from PIL import Image as PILImage
import io
import json
import importlib.util
import pandas as pd
import ast # for conversion of csv string to dictionary


# Specify dataset from hugginface hub and local paths
url_to_dataset = "zh-plus/tiny-imagenet"
split = "valid"
metadata_file_path = "datasets/zh-plus___tiny-imagenet/default/0.0.0/5a77092c28e51558c5586e9c5eb71a7e17a5e43f/dataset_info.json"
classes_path = "datasets/zh-plus___tiny-imagenet/default/0.0.0/5a77092c28e51558c5586e9c5eb71a7e17a5e43f/classes.py"

# Load the dataset into a Dataset object (not an IterableDataset)
ds = load_dataset(url_to_dataset, split=split, cache_dir="./datasets")

# Load metadata
with open(metadata_file_path, "r") as file:
    metadata = json.load(file)
n_codes = metadata["features"]["label"]["names"]
print("length of n_codes: ", len(n_codes))

# Load classes
spec = importlib.util.spec_from_file_location("classes", classes_path)
classes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classes)
n_code2label_complete = classes.i2d  # name of dictorionary in classes.py

# Create label converter dictionaries, three ways in this case! we have ids = 0,1,2,3... and n_codes = n01629819, n01641577, n01644900, n01698640... and labels = "European fire salamander, Salamandra salamandra", "bullfrog, Rana catesbeiana", "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui", "common newt, Triturus vulgaris"
id2n_code = {} # length 200
n_code2id = {} # lenght 200, use if-statement to check if n_code is in n_code2id, else assign -1
for i, n_code in enumerate(n_codes):
    id2n_code[i] = n_code
    n_code2id[n_code] = i

id2label = {} # length 200
label2id = {} # length 200, use if-statement to check if n_code is in i2d_dict, else assign -1
for i, n_code in enumerate(n_codes):
    if n_code in n_code2label_complete:
        label = n_code2label_complete[n_code]  # Get the corresponding label from class_dict
        label2id[label] = i
        id2label[i] = label
    else:
        print(f"Class {n_code} not found in class_dict")

#label 2 n_code and back, both length 80k
label2n_code_complete = {v: k for k, v in n_code2label_complete.items()}

# EXTRACT PREDICTIONS AND GROUND TRUTH
# Get CSV file with predictions, only one column, rename it to "best_guess"
csv_file_path = "./predictions/preds_run_1.csv"
preds = pd.read_csv(csv_file_path, index_col=0, usecols=[0, 1])
preds = preds.rename(columns={"0": "best_guess"})

# Get the predicted labels
def get_label(dict_string):
    dict_value = ast.literal_eval(dict_string)
    return dict_value['label']

preds["best_guess"] = preds["best_guess"].apply(get_label)
preds_labels = preds["best_guess"].tolist()
preds_ids = [label2id[i] if i in label2id else -1 for i in preds_labels] # -1 for predictions outside the tinyImgNet dataset
preds_n_codes = [label2n_code_complete[i] for i in preds_labels]
print()
print(preds.tail())

# Get ground truth
truth_ids = ds['label']
truth_labels = [id2label[i] for i in truth_ids]
truth_n_codes = [id2n_code[i] for i in truth_ids]

# Basic accuracy
accuracy = evaluate.load("accuracy")
predictions = preds_ids
references = truth_ids
print(accuracy.compute(references=references, predictions=predictions))

# metric during inference
""" for model_inputs, gold_standards in evaluation_dataset:
    predictions = model(model_inputs)
    metric.add_batch(references=gold_standards, predictions=predictions)
metric.compute() """
#metric = evaluate.load("MeaunIoU")


""" for each sample
the id of the sample
the label of the prediction, first row
convert label to id
send to metric """

# Custom function to view image from dataset using PIL and io libraries
def ViewImage(image_location):
    # Convert the image data from JpegImageFile to bytes
    image_bytes = io.BytesIO()
    image_location.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Open the image using PIL
    PILImage.open(image_bytes).show()

# View the first image in the dataset
# ViewImage(ds["image"][0])

