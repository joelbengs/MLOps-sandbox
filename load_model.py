from transformers import pipeline

#model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"
model_name = 'google/vit-base-patch16-224'
classifier = pipeline(model=model_name, task="image-classification")

classifier.save_pretrained(f"./models/{model_name}")