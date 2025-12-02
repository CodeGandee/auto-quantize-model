# How to Print ONNX Model Graph in Readable Text Format

This guide covers methods for inspecting and printing ONNX model graphs in human-readable text format using Python libraries.

## Using `onnx.helper.printable_graph()` (Recommended)

The standard and most commonly used method for printing ONNX graphs in text format is the `printable_graph()` function from the `onnx.helper` module.

### Basic Usage

```python
import onnx

# Load the ONNX model
model = onnx.load("model.onnx")

# Validate the model is well-formed
onnx.checker.check_model(model)

# Print a human-readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
```

This will display:
- Input and output tensors with their names, data types, and shapes
- All nodes (operators) with their input/output connections
- Initializers (constant weights and parameters)
- The overall computational flow

### Example Output Format

```text
graph torch_jit (
  %input[FLOAT, 1x3x224x224]
) initializers (
  %conv1.weight[FLOAT, 64x3x7x7]
  %conv1.bias[FLOAT, 64]
  %bn1.weight[FLOAT, 64]
  %bn1.bias[FLOAT, 64]
) {
  %11 = Conv[dilations=[1,1], group=1, kernel_shape=[7,7], pads=[3,3,3,3], strides=[2,2]](%input, %conv1.weight, %conv1.bias)
  %12 = BatchNormalization[epsilon=1e-05, momentum=0.9](%11, %bn1.weight, %bn1.bias, %bn1.running_mean, %bn1.running_var)
  %output = Relu(%12)
  return %output
}
```

## Using `onnx.printer.to_text()` (Alternative)

The `onnx.printer` module provides another way to convert ONNX objects to text representation.

```python
import onnx
from onnx import printer

# Load model
model = onnx.load("model.onnx")

# Convert entire model to text
text_representation = printer.to_text(model)
print(text_representation)
```

This method provides a more complete representation including model metadata, IR version, and opset imports.

## Detailed Component Inspection

For more granular inspection of specific graph components:

### Inspecting Model Inputs

```python
print("** INPUTS **")
for input_tensor in model.graph.input:
    print(f"Name: {input_tensor.name}")
    print(f"Type: {input_tensor.type.tensor_type.elem_type}")
    print(f"Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
    print()
```

### Inspecting Nodes (Operators)

```python
print("** NODES **")
for node in model.graph.node:
    print(f"Name: {node.name}")
    print(f"Op Type: {node.op_type}")
    print(f"Inputs: {list(node.input)}")
    print(f"Outputs: {list(node.output)}")
    print(f"Attributes: {[(attr.name, onnx.helper.get_attribute_value(attr)) for attr in node.attribute]}")
    print()
```

### Inspecting Initializers (Weights)

```python
import numpy as np
from onnx import numpy_helper

print("** INITIALIZERS **")
for init in model.graph.initializer:
    print(f"Name: {init.name}")
    print(f"Data Type: {init.data_type}")
    print(f"Shape: {init.dims}")
    
    # Convert to numpy array to see values
    np_array = numpy_helper.to_array(init)
    print(f"NumPy shape: {np_array.shape}")
    print(f"Sample values: {np_array.flat[:5]}")  # First 5 values
    print()
```

### Inspecting Model Outputs

```python
print("** OUTPUTS **")
for output_tensor in model.graph.output:
    print(f"Name: {output_tensor.name}")
    print(f"Type: {output_tensor.type.tensor_type.elem_type}")
    print(f"Shape: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")
    print()
```

## Using `print(model)` for Full Proto Representation

The simplest method for debugging is to print the entire model protobuf:

```python
import onnx

model = onnx.load("model.onnx")

# Print the entire model proto (can be very verbose)
print(model)

# Or use format() for similar output
print(format(model))
```

This shows the complete protobuf representation including all metadata, but can be very verbose for large models.

## Helper Function for Printable Node Details

```python
def print_node_details(node, prefix=""):
    """Print detailed information about a node"""
    print(onnx.helper.printable_node(node, prefix=prefix))
```

## Printing Specific Attributes

```python
def print_attribute(attr):
    """Print attribute in readable format"""
    print(onnx.helper.printable_attribute(attr))
```

## Comparing Model Versions

When comparing two ONNX models (e.g., FP16 vs INT8 quantized):

```python
import onnx

fp16_model = onnx.load("model_fp16.onnx")
int8_model = onnx.load("model_int8_qdq.onnx")

print("=" * 80)
print("FP16 MODEL GRAPH")
print("=" * 80)
print(onnx.helper.printable_graph(fp16_model.graph))

print("\n" + "=" * 80)
print("INT8 QUANTIZED MODEL GRAPH")
print("=" * 80)
print(onnx.helper.printable_graph(int8_model.graph))

# Count differences
print(f"\nFP16 model has {len(fp16_model.graph.node)} nodes")
print(f"INT8 model has {len(int8_model.graph.node)} nodes")
print(f"Additional Q/DQ nodes: {len(int8_model.graph.node) - len(fp16_model.graph.node)}")
```

## Saving Graph to Text File

```python
import onnx

model = onnx.load("model.onnx")

# Save to text file
with open("model_graph.txt", "w") as f:
    f.write(onnx.helper.printable_graph(model.graph))

print("Graph saved to model_graph.txt")
```

## Tensor Data Type Reference

When inspecting tensors, the data types are represented as integers. Here's a quick reference:

```python
from onnx import TensorProto

# Common tensor types
type_mapping = {
    TensorProto.FLOAT: "FLOAT (FP32)",
    TensorProto.UINT8: "UINT8",
    TensorProto.INT8: "INT8",
    TensorProto.UINT16: "UINT16",
    TensorProto.INT16: "INT16",
    TensorProto.INT32: "INT32",
    TensorProto.INT64: "INT64",
    TensorProto.STRING: "STRING",
    TensorProto.BOOL: "BOOL",
    TensorProto.FLOAT16: "FLOAT16 (FP16)",
    TensorProto.DOUBLE: "DOUBLE (FP64)",
    TensorProto.UINT32: "UINT32",
    TensorProto.UINT64: "UINT64",
    TensorProto.BFLOAT16: "BFLOAT16",
    TensorProto.FLOAT8E4M3FN: "FLOAT8_E4M3FN",
    TensorProto.FLOAT8E5M2: "FLOAT8_E5M2",
}

def dtype_to_string(dtype_int):
    """Convert ONNX dtype integer to readable string"""
    return type_mapping.get(dtype_int, f"UNKNOWN({dtype_int})")
```

## Visual Inspection (Non-Text Alternative)

For graphical visualization (not text-based but worth mentioning):

**Netron** is the most popular tool for visual ONNX model inspection:
```bash
# Install
pip install netron

# Run web interface
netron model.onnx
```

This opens an interactive web interface showing the graph structure visually.

## Practical Use Cases

### 1. Debugging Quantization

```python
# Check for Q/DQ nodes after quantization
for node in model.graph.node:
    if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
        print(f"Found {node.op_type}: {node.input} -> {node.output}")
```

### 2. Finding Specific Operators

```python
# Find all Conv operators
conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
print(f"Found {len(conv_nodes)} Conv layers")
for node in conv_nodes:
    print(onnx.helper.printable_node(node))
```

### 3. Checking Model Opset Version

```python
print(f"IR Version: {model.ir_version}")
print(f"Opset Imports:")
for opset in model.opset_import:
    print(f"  Domain: {opset.domain or 'default'}, Version: {opset.version}")
```

## Troubleshooting

### Issue: `printable_graph()` Returns Nothing

If the function doesn't print anything, make sure to explicitly print the return value:

```python
# Correct
print(onnx.helper.printable_graph(model.graph))

# Incorrect (won't display anything)
onnx.helper.printable_graph(model.graph)
```

### Issue: Output Too Large

For very large models, redirect output to a file:

```python
import sys

with open("large_model_graph.txt", "w") as f:
    sys.stdout = f
    print(onnx.helper.printable_graph(model.graph))
    sys.stdout = sys.__stdout__
```

## References

- [ONNX Python API - Helper Functions](https://onnx.ai/onnx/api/helper.html)
- [ONNX Python API - Printer Module](https://onnx.ai/onnx/api/printer.html)
- [ONNX Python API Overview](https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html)
- [ONNX GitHub Repository](https://github.com/onnx/onnx)
- [Netron - Model Visualizer](https://github.com/lutzroeder/netron)
