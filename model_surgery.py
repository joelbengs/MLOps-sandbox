from transformers import ViTFeatureExtractor, TFViTForImageClassification
import tensorflow as tf
from PIL import Image
import requests

# adapted from https://stackoverflow.com/questions/71482661/how-to-modify-base-vit-architecture-from-huggingface-in-tensorflow
print("--- JOEL BENGS ---")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224", cache_dir="./models") #The model from the  16x16 words-paper
model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224", cache_dir="./models")

inputs = feature_extractor(images=image, return_tensors="tf")
outputs = model(**inputs)
logits = outputs.logits
# Model does prediction of 1000 ImageNet classes - but how does it know that only 1000 classes are used? In this case, the model was actually fine-tuned on ImageNet1K.
predicted_class_idx = tf.math.argmax(logits, axis=-1)[0] #grab the index of the best guess. alternative: logits.argmax(-1).item()
print("Predicted class: ", model.config.id2label[int(predicted_class_idx)]) #model has function for conversion id2label!
image.show()

print(model.layers[0].embeddings.patch_embeddings.projection)
print(model.layers[0].embeddings.dropout)
print()
print(model.layers[0].encoder.layer[0].attention.self_attention.query)
print(model.layers[0].encoder.layer[0].attention.self_attention.key)
print(model.layers[0].encoder.layer[0].attention.self_attention.value)
print(model.layers[0].encoder.layer[0].attention.self_attention.dropout)
print(model.layers[0].encoder.layer[0].attention.dense_output.dense)
print(model.layers[0].encoder.layer[0].attention.dense_output.dropout)