from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator
import evaluate

# LOAD
# initialize the metric with load function
# compute the metric with compute function

accuracy = evaluate.load("accuracy")
print(accuracy.description)
print(accuracy.features)

# MULTIPLE metrics
# if needing to store intermediate results, use accurac.add(truth, preds) followed my accuracy.compute
# also works with add_batch(list of truth, list of preds) followed by accuracy.compute
print(accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1]))


# COMBINE
# if intereseted in several metrics, you can bundle them using evaluate.combine(). Then all can be computed with one call to compute.

# SAVING results can be done with full control
# result = accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1])
# hyperparams = {"model": "bert-base-uncased"}
# evaluate.save("./results/"experiment="run 42", **result, **hyperparams)

# Inference can be performed using evaluate.evaluator() + a pipeline.

pipe = pipeline("text-classification", model="lvwerra/distilbert-imdb", device=0)
data = load_dataset("imdb", split="test", cache_dir="./datasets").shuffle().select(range(1000))
metric = evaluate.load("accuracy")

task_evaluator = evaluator("text-classification")

results = task_evaluator.compute(model_or_pipeline=pipe, data=data, metric=metric,
                       label_mapping={"NEGATIVE": 0, "POSITIVE": 1},)

print(results)


from evaluate.visualization import radar_plot

data = [
   {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
   {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
   {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6}, 
   {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
   ]
model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
plot = radar_plot(data=data, model_names=model_names)
plot.show()
