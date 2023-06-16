import os
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers


class JITWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, mask):
        mask = torch.where(mask > 0.5, 1.0, mask)
        batch = {"image": image, "mask": mask}
        out = self.model(batch)
        out = out["inpainted"] * 255
        out = torch.clamp(out, 0, 255)
        return out


@hydra.main(config_path="../configs/prediction", config_name="default.yaml")
def main(predict_config: OmegaConf):
    register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

    train_config_path = os.path.join(predict_config.model.path, "config.yaml")
    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    checkpoint_path = os.path.join(
        predict_config.model.path, "models", predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location="cpu"
    )
    model.eval()
    jit_model_wrapper = JITWrapper(model)

    image = torch.rand(1, 3, 512, 512)
    mask = torch.rand(1, 1, 512, 512)
    output = jit_model_wrapper(image, mask)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    image = image.to(device)
    mask = mask.to(device)
    traced_model = torch.jit.trace(jit_model_wrapper, (image, mask), strict=False).to(
        device
    )

    import coremltools as ct

    # Using image_input in the inputs parameter:
    # Convert to Core ML program using the Unified Conversion API.
    # 动态尺寸
    # input_image_shape = ct.Shape(
    #     shape=(
    #         1,
    #         3,
    #         ct.RangeDim(lower_bound=512, upper_bound=1024, default=512),
    #         ct.RangeDim(lower_bound=512, upper_bound=1024, default=512),
    #     )
    # )
    # input_mask_shape = ct.Shape(
    #     shape=(
    #         1,
    #         1,
    #         ct.RangeDim(lower_bound=512, upper_bound=1024, default=512),
    #         ct.RangeDim(lower_bound=512, upper_bound=1024, default=512),
    #     )
    # )

    # 枚举尺寸
    # input_image_shape = ct.EnumeratedShapes(
    #     shapes=[
    #         [1, 3, 512, 512],
    #         [1, 3, 1024, 1024],
    #     ],
    #     default=(1, 3, 512, 512),
    # )
    # input_mask_shape = ct.EnumeratedShapes(
    #     shapes=[
    #         [1, 1, 512, 512],
    #         [1, 1, 1024, 1024],
    #     ],
    #     default=(1, 1, 512, 512),
    # )

    # 固定尺寸
    s = 720
    input_image_shape = ct.Shape(
        shape=[1, 3, s, s],
    )
    input_mask_shape = ct.Shape(
        shape=[1, 1, s, s],
    )

    coreml_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        inputs=[
            ct.ImageType(
                name="image",
                scale=1 / 255,
                shape=input_image_shape,
                color_layout=ct.colorlayout.RGB,
            ),
            ct.ImageType(
                name="mask",
                scale=1 / 255,
                shape=input_mask_shape,
                color_layout=ct.colorlayout.GRAYSCALE,
            ),
        ],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
    )
    coreml_model.save(
        str(Path(checkpoint_path).parent / f"big-lama-{s}x{s}-fp32.mlpackage")
    )

    save_path = Path(checkpoint_path).with_suffix(".pt")

    print(f"Saving big-lama.pt model to {save_path}")
    traced_model.save(save_path)

    print(f"Checking jit model output...")
    jit_model = torch.jit.load(str(save_path))
    jit_output = jit_model(image, mask)
    diff = (output - jit_output).abs().sum()
    print(f"diff: {diff}")


if __name__ == "__main__":
    main()
