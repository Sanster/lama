from pathlib import Path

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# load full precision model
model_path = Path("/Users/cwq/data/models/big-lama/models/big-lama-1024x1024.mlmodel")
model_fp32 = ct.models.MLModel(str(model_path))
print("Loaded full precision model")

model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
model_fp16.save(str(model_path.parent / f"{model_path.stem}-fp16{model_path.suffix}"))
