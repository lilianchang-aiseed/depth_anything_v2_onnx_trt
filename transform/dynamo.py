"""
modified pth -> onnx file based on below repo: 
https://github.com/fabio-sim/Depth-Anything-ONNX/tree/main

======
usage
transform: 
python3 transform/dynamo.py export \
  --encoder vits \
  --batch-size 0 \
  --height 0 \
  --width 0 \
  --output checkpoints/depth_anything_v2_vits_dynamic.onnx

inference: 
python3 transform/dynamo.py infer \
    checkpoints/depth_anything_v2_vits_dynamic.onnx \
    -i assets/examples
"""
from enum import StrEnum, auto
from pathlib import Path
import sys
from typing import Annotated, Optional

import cv2
import matplotlib
import numpy as np
import torch
import typer

# Allow this script to import the package after being moved into transform/.
TRANSFORM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRANSFORM_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from depth_anything_v2.config import Encoder, Metric
from depth_anything_v2.dpt import DepthAnythingV2


LOCAL_CHECKPOINT = PROJECT_DIR / "checkpoints" / "depth_anything_v2_vits.pth"


class ExportFormat(StrEnum):
    onnx = auto()
    pt2 = auto()


class InferenceDevice(StrEnum):
    cpu = auto()
    cuda = auto()


app = typer.Typer()


@app.callback()
def callback():
    """Depth-Anything Dynamo CLI"""


def multiple_of_14(value: int) -> int:
    if value % 14 != 0:
        raise typer.BadParameter("Value must be a multiple of 14.")
    return value


@app.command()
def export(
    encoder: Annotated[Encoder, typer.Option()] = Encoder.vits,
    metric: Annotated[
        Optional[Metric], typer.Option(help="Export metric depth models.")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save exported model.",
        ),
    ] = None,
    format: Annotated[
        ExportFormat, typer.Option("-f", "--format", help="Export format.")
    ] = ExportFormat.onnx,
    batch_size: Annotated[
        int,
        typer.Option(
            "-b",
            "--batch-size",
            min=0,
            help="Batch size of exported ONNX model. Set to 0 to mark as dynamic.",
        ),
    ] = 1,
    height: Annotated[
        int,
        typer.Option(
            "-h",
            "--height",
            min=0,
            help="Height of input image. Set to 0 to mark as dynamic.",
            callback=multiple_of_14,
        ),
    ] = 518,
    width: Annotated[
        int,
        typer.Option(
            "-w",
            "--width",
            min=0,
            help="Width of input image. Set to 0 to mark as dynamic.",
            callback=multiple_of_14,
        ),
    ] = 518,
    opset: Annotated[
        int,
        typer.Option(
            max=17,
            help="ONNX opset version of exported model. Defaults to 17.",
        ),
    ] = 17,
    use_dynamo: Annotated[
        bool,
        typer.Option(
            help="Use TorchDynamo (Beta) for ONNX export. Only supports static shapes and opset 18."
        ),
    ] = False,
):
    """Export Depth-Anything V2 using TorchDynamo."""
    if encoder != Encoder.vits:
        raise typer.BadParameter(
            "Only the local vits checkpoint is available; use --encoder vits.",
            param_hint="--encoder",
        )
    if metric is not None:
        raise typer.BadParameter(
            "The local checkpoint is the relative-depth model and does not "
            "support --metric.",
            param_hint="--metric",
        )
    if not LOCAL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Local checkpoint not found: {LOCAL_CHECKPOINT}"
        )

    if torch.__version__ < "2.3":
        typer.echo(
            "Warning: torch version is lower than 2.3, export may not work properly."
        )

    if output is None:
        output = (
            PROJECT_DIR
            / "checkpoints"
            / f"depth_anything_v2_{encoder}_{opset}.{format}"
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    config = encoder.get_config(metric)
    model = DepthAnythingV2(
        encoder=encoder.value,
        features=config.features,
        out_channels=config.out_channels,
    )
    typer.echo(f"Loading local checkpoint: {LOCAL_CHECKPOINT}")
    state_dict = torch.load(LOCAL_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    if format == ExportFormat.onnx:
        if use_dynamo:
            typer.echo(
                "Exporting to ONNX using TorchDynamo (Beta). Only supports static shapes and opset 18."
            )
            onnx_program = torch.onnx.dynamo_export(
                model, torch.randn(batch_size or 1, 3, height or 518, width or 518)
            )
            onnx_program.save(str(output))
        else:  # Use TS exporter.
            typer.echo("Exporting to ONNX using legacy JIT tracer.")
            input_dynamic_axes = {}
            output_dynamic_axes = {}
            if batch_size == 0:
                input_dynamic_axes[0] = "batch_size"
                output_dynamic_axes[0] = "batch_size"
            if height == 0:
                input_dynamic_axes[2] = "height"
                output_dynamic_axes[1] = "height"
            if width == 0:
                input_dynamic_axes[3] = "width"
                output_dynamic_axes[2] = "width"
            torch.onnx.export(
                model,
                torch.randn(batch_size or 1, 3, height or 140, width or 140),
                str(output),
                input_names=["image"],
                output_names=["depth"],
                opset_version=opset,
                dynamic_axes={
                    "image": input_dynamic_axes,
                    "depth": output_dynamic_axes,
                },
                dynamo=False,
            )
    elif format == ExportFormat.pt2:
        batch_dim = torch.export.Dim("batch_size")
        export_program = torch.export.export(
            model.eval(),
            (torch.randn(2, 3, height or 518, width or 518),),
            dynamic_shapes={
                "x": {0: batch_dim},
            },
        )
        torch.export.save(export_program, output)


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
