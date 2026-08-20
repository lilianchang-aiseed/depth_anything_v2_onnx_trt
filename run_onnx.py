"""Run Depth Anything V2 ONNX inference without model export code.

Example:
    python3 run_onnx.py infer \
        checkpoints/depth_anything_v2_vits_dynamic.onnx \
        --image input.png --output depth_color.png
"""

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Optional

import cv2
import matplotlib
import numpy as np
import typer


class InferenceDevice(StrEnum):
    cpu = auto()
    cuda = auto()


app = typer.Typer()


@app.callback()
def callback():
    """Depth Anything V2 ONNX Runtime inference CLI."""


def multiple_of_14(value: int) -> int:
    if value % 14 != 0:
        raise typer.BadParameter("Value must be a multiple of 14.")
    return value


@app.command()
def infer(
    model_path: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Path to ONNX model."
        ),
    ],
    image_path: Annotated[
        Path,
        typer.Option(
            "-i",
            "--img",
            "--image",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to input image.",
        ),
    ],
    height: Annotated[
        int,
        typer.Option(
            "-h",
            "--height",
            min=14,
            help="Height at which to perform inference. The input image will be resized to this.",
            callback=multiple_of_14,
        ),
    ] = 518,
    width: Annotated[
        int,
        typer.Option(
            "-w",
            "--width",
            min=14,
            help="Width at which to perform inference. The input image will be resized to this.",
            callback=multiple_of_14,
        ),
    ] = 518,
    device: Annotated[
        InferenceDevice, typer.Option("-d", "--device", help="Inference device.")
    ] = InferenceDevice.cuda,
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output depth map. If not given, show visualization.",
        ),
    ] = None,
):
    """Depth-Anything V2 inference using ONNXRuntime. No dependency on PyTorch."""
    import onnxruntime as ort

    # Preprocessing, implement this part in your chosen language:
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)[None].astype("float32")

    # Inference
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = False
    # For inspecting applied ORT-optimizations:
    # sess_options.optimized_model_filepath = "weights/optimized.onnx"
    providers = ["CPUExecutionProvider"]
    if device == InferenceDevice.cuda:
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers
    )
    binding = session.io_binding()
    ort_input = session.get_inputs()[0].name
    binding.bind_cpu_input(ort_input, image)
    ort_output = session.get_outputs()[0].name
    binding.bind_output(ort_output, device.value)

    session.run_with_iobinding(binding)  # Actual inference happens here.

    depth = binding.get_outputs()[0].numpy().squeeze()

    # Match the Spectral_r visualization used by Depth-Anything-V2/run.py.
    depth_range = float(depth.max() - depth.min())
    if depth_range > np.finfo(np.float32).eps:
        depth = (depth - depth.min()) / depth_range * 255.0
    else:
        depth = np.zeros_like(depth)
    depth = depth.astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    depth = (cmap(depth)[..., :3] * 255)[:, :, ::-1].astype(np.uint8)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

    if output_path is None:
        cv2.imshow("depth", depth)
        cv2.waitKey(0)
    else:
        cv2.imwrite(str(output_path), depth)


if __name__ == "__main__":
    app()
