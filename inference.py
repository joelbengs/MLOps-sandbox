from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor, ImageClassificationPipeline
from PIL import Image
from loguru import logger
import pandas as pdc
import torch


#model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"
model_name = 'google/vit-base-patch16-224'

model_path = f"./models/{model_name}"

model = AutoModelForImageClassification.from_pretrained(model_path, device_map="cuda:0", load_in_8bit=True)

image_processor = AutoImageProcessor.from_pretrained(model_path)
classifier = ImageClassificationPipeline(model=model, image_processor=image_processor)

img = Image.open('./parrots.png')
#processed = image_processor(img)
#processed["pixel_values"] = processed["pixel_values"].to(torch.float16)

preds = classifier(img)

df = pdc.DataFrame(preds)
df.to_csv("./results/preds.csv", index=False)
