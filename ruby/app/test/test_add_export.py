# test_add_export.py
import torch
import torch.nn as nn

class AddOne(nn.Module):
    def forward(self, x):
        return x + 1

model = AddOne()
dummy = torch.zeros(1, 3, dtype=torch.float32)  # shape [1, 3]

torch.onnx.export(
    model,
    dummy,
    "models/test_add.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
print("Exported models/test_add.onnx")
