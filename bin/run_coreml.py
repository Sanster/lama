import time
import cv2
from pathlib import Path

import coremltools as ct
from PIL import Image
import numpy as np

s = 800

image = Image.open("/Users/cwq/code/github/lama-cleaner/assets/dog.jpg").resize((s, s))
# image = np.asarray(image)
# image = image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
# print(image.shape)

mask = (
    Image.open("/Users/cwq/code/github/lama-cleaner/examples/dog_mask.png")
    .convert("L")
    .resize((s, s))
)
mask = np.asarray(mask)
mask = Image.fromarray(mask)
# print(mask.shape)

model_path = Path(
    f"/Users/cwq/data/models/big-lama/models/big-lama-{s}x{s}-tmp.mlpackage"
)
start = time.time()
model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
print(f"loaded model: {time.time() - start}s")


for _ in range(3):
    start = time.time()
    # Make a prediction using Core ML
    out_dict = model.predict({"image": image, "mask": mask})
    print(f"forward time: {time.time() - start}s")

key = list(out_dict.keys())[0]
pil_img = out_dict[key]
pil_img.save("result.png")
print(pil_img.mode)
# res = (out_dict[key] * 255).astype(np.uint8)
# res = res[0].transpose(1, 2, 0)
# print(res.shape, res.dtype, res.max(), res.min())
# res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
# print(cv2.imwrite("result.jpg", res))
# a = 0
