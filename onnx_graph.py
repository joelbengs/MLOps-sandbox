# Required library import
import onnx
print(" HELLO FROM ONNX_GRAPH.PY --- This script prints out the nodes of an onnx-model")

# Define the path to your ONNX model
onnx_path = "./onnx/model.onnx"

# Load the model
model = onnx.load(onnx_path)

# Access the graph
graph = model.graph

# List to hold node names
node_names = []

# Iterate over the nodes and extract their names
for node in graph.node:
    node_names.append(node.name)

# Display the node names
print(node_names)