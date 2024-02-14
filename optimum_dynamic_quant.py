from optimum.onnxruntime import ORTModelForImageClassification
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoFeatureExtractor
from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator

from PIL import Image
from pathlib import Path
import io
import os

# Adopted from https://www.philschmid.de/optimizing-vision-transformer

model_id = "nateraw/vit-base-beans"
onnx_path = Path("onnx") # creates a directory "onnx" where onnx-format models (preQ/postQ) will be stored.
cache_dir_datasets='./datasets'
cache_dir_models = "./models"

dataset = load_dataset("beans", split="train", cache_dir=cache_dir_datasets)
dataset[0]["image"] # would work in a notebook
#image = Image.open(dataset[0]["image"])

# load vanilla transformers and convert from pytorch/tensorflow to onnx
model = ORTModelForImageClassification.from_pretrained(model_id, export=True, cache_dir=cache_dir_models)
preprocessor_from_base_model = AutoFeatureExtractor.from_pretrained(model_id)

# save checkpoint of model (onnx-format) and preprocessor (just json file) in onnx directory
model.save_pretrained(onnx_path)
preprocessor_from_base_model.save_pretrained(onnx_path)

# build a pipeline with vanilla model and preprossesor
vanilla_clf = pipeline("image-classification", model=model, feature_extractor=preprocessor_from_base_model)

# inference
print()
print("--- INFERENCE ---")
print(vanilla_clf(dataset[0]["image"]))
print()

# Custom function to view image from dataset
def ViewImage(image_location):
    # Convert the image data from JpegImageFile to bytes
    image_bytes = io.BytesIO()
    image_location.save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    Image.open(image_bytes).show()

ViewImage(dataset[0]["image"])

# Quantizer
# More info: https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/quantization
# DOCS: https://huggingface.co/docs/optimum/onnxruntime/package_reference/quantization#optimum.onnxruntime.ORTQuantizer
dynamic_quantizer = ORTQuantizer.from_pretrained(model) # create a quantizer for the model

#Configure Quantizer
# Define the quantization strategy. Use avx512_vnni instead of arm64 on intel. tensorrt also available
# DOCS: https://huggingface.co/docs/optimum/onnxruntime/package_reference/configuration#optimum.onnxruntime.AutoQuantizationConfig
quanitzation_config = AutoQuantizationConfig.arm64(is_static=False, per_channel=False, use_symmetric_activations=False, use_symmetric_weights=False, nodes_to_quantize=None, nodes_to_exclude=['/vit/embeddings/patch_embeddings/projection/Conv_quant'], operators_to_quantize=['MatMul', 'Attention', 'LSTM', 'Gather', 'Transpose', 'EmbedLayerNormalization'])

# Quantize the model and save (as onnx, in directory onnx) (name seems to automatically become model_quantized.onnx)
model_quantized_path = dynamic_quantizer.quantize(
    save_dir=onnx_path,
    quantization_config=quanitzation_config
)

# model size
size_fp32 = os.path.getsize(onnx_path / "model.onnx") / (1024*1024) # appends paths, retrieves size of file and and divides by scalar
size_quantized = os.path.getsize(onnx_path / "model_quantized.onnx")/(1024*1024)

print(f"FP32 Model file size: {size_fp32:.2f} MB")
print(f"Quantized Model file size: {size_quantized:.2f} MB")

# Build pipeline. Optimum is a superset of transformers, so it supports pipelines from transformers!
model_quantized = ORTModelForImageClassification.from_pretrained(onnx_path, file_name="model_quantized.onnx")
preprocessor_from_onnx = AutoFeatureExtractor.from_pretrained(onnx_path) #finds the preprocessor_config.json file in the directory onnx
q8_clf = pipeline("image-classification", model=model_quantized, feature_extractor=preprocessor_from_onnx)

# Test inference
print("--- QUANTIZED PIPELINE INFERENCE ---")
print(q8_clf(dataset[0]["image"]))

# Evaluator pipeline from the evaluate package!
eval =evaluator("image-classification")
eval_dataset = load_dataset("beans",split=["test"], cache_dir=cache_dir_datasets)[0]


print(eval_dataset)

results = eval.compute(
    model_or_pipeline=q8_clf,
    data=eval_dataset,
    metric="accuracy",
    input_column="image",
    label_column="labels",
    label_mapping=model.config.label2id, #automatic label2id conversion!
    strategy="simple",
)

print(f"Vanilla model: 96.88%")
print(f"Quantized model: {results['accuracy']*100:.2f}%")
print(f"The quantized model achieves {round(results['accuracy']/0.9688,4)*100:.2f}% accuracy of the fp32 model")





# SELF ATTEMPT AT USING ONNX FORMAT FOR INFERENCE WITHOUT PIPELINE
""" # Path to the quantized ONNX model file

import onnxruntime as ort
import shutil
import numpy as np

cache_dir_quantized_models = "./models/quantized"

#remove old quantized model
if Path(cache_dir_quantized_models).exists():
    shutil.rmtree(cache_dir_quantized_models)
Path(cache_dir_quantized_models).mkdir(parents=True, exist_ok=True)

# Quantize the model and save (as ONNX)
model_quantized_onnx = dynamic_quantizer.quantize(
    save_dir=cache_dir_quantized_models,
    quantization_config=quanitzation_config
)

onnx_model_path = Path(cache_dir_quantized_models) / "model_quantized.onnx"  # Adjust the filename as needed

# Load the quantized ONNX model with ONNX Runtime
session = ort.InferenceSession(str(onnx_model_path))

# Load the same old feature extractor/preprocessor
preprocessor_quantized = preprocessor

# Function to preprocess the image (because we are not using the pipleine from transformers with ONNX)
def preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = preprocessor_quantized(images=image, return_tensors="np")
    return inputs["pixel_values"]

# Function for inference with ONNX Runtime
def run_onnx_inference(image_path):
    inputs = preprocess_image(image_path)
    # ONNX Runtime expects inputs as name:array dictionary
    onnx_inputs = {session.get_inputs()[0].name: inputs}
    logits = session.run(None, onnx_inputs)[0]
    return logits

# Example inference (adjust the image path as necessary)
image_path = dataset[0]["image"]
logits = run_onnx_inference(image_path)

print("--- QUANTIZED INFERENCE ---")
print(logits)
print()
 """

""" PIPELINE APPROACH
#build quantized model from ONNX
model_quantized = ORTModelForImageClassification.from_pretrained(model_quantized_onnx, export=True, cache_dir=cache_dir_quantized_models)
preprocessor_quantized = AutoFeatureExtractor.from_pretrained(model_quantized_onnx)

# quantized model pippeline
quantized_clf = pipeline("image-classification", model=model_quantized, feature_extractor=preprocessor_quantized)
 """
""" # quantized inference
print()
print("--- QUANTIZED INFERENCE ---")
print(quantized_clf(dataset[0]["image"]))
print() """