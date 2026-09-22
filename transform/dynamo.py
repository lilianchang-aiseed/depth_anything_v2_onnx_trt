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

metric VKITTI (raw output is depth in metres, maximum 80 m):
python3 transform/dynamo.py export \
  --encoder vits \
  --metric outdoor \
  --output checkpoints/depth_anything_v2_metric_vkitti_vits.onnx

custom trained metric checkpoint (dynamic batch, static 322x322 image):
python3 transform/dynamo.py export \
  --custom \
  --checkpoint metric_depth/out_data/runs/RUN_NAME/best.pt \
  --encoder vits \
  --batch-size 0 \
  --height 322 \
  --width 322 \
  --output metric_depth/out_data/runs/RUN_NAME/best_dynamic_b322.onnx

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
from depth_anything_v2.dpt import DepthAnythingV2 as RelativeDepthAnythingV2
from metric_depth.model_metric_depth.dpt import (
    DepthAnythingV2 as MetricDepthAnythingV2,
)


LOCAL_CHECKPOINTS = {
    None: PROJECT_DIR / "checkpoints" / "depth_anything_v2_vits.pth",
    Metric.indoor: (
        PROJECT_DIR
        / "checkpoints"
        / "depth_anything_v2_metric_hypersim_vits.pth"
    ),
    Metric.outdoor: (
        PROJECT_DIR
        / "checkpoints"
        / "depth_anything_v2_metric_vkitti_vits.pth"
    ),
}


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
    custom: Annotated[
        bool,
        typer.Option(
            "--custom",
            help="Use the local custom metric-depth architecture.",
        ),
    ] = False,
    checkpoint_path: Annotated[
        Optional[Path],
        typer.Option(
            "--checkpoint",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Weights to load; required with --custom.",
        ),
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
    if encoder != Encoder.vits and not custom:
        raise typer.BadParameter(
            "Only the local vits checkpoint is available; use --encoder vits.",
            param_hint="--encoder",
        )
    if custom and metric is not None:
        raise typer.BadParameter(
            "--custom and --metric select different model architectures; "
            "use only one.",
            param_hint="--custom/--metric",
        )
    if custom and checkpoint_path is None:
        raise typer.BadParameter(
            "--checkpoint is required with --custom.",
            param_hint="--checkpoint",
        )
    if not custom and checkpoint_path is not None:
        raise typer.BadParameter(
            "--checkpoint is reserved for --custom so architecture and "
            "weights are selected explicitly.",
            param_hint="--checkpoint",
        )
    if (metric is not None or custom) and (height == 0 or width == 0):
        raise typer.BadParameter(
            "Metric export requires static --height and --width with the "
            "legacy tracer; use the 518x518 defaults.",
            param_hint="--height/--width",
        )
    checkpoint = checkpoint_path if custom else LOCAL_CHECKPOINTS[metric]
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            f"Local checkpoint not found: {checkpoint}"
        )

    if torch.__version__ < "2.3":
        typer.echo(
            "Warning: torch version is lower than 2.3, export may not work properly."
        )

    if output is None:
        model_kind = "custom_metric" if custom else str(encoder)
        output = (
            PROJECT_DIR
            / "checkpoints"
            / f"depth_anything_v2_{model_kind}_{opset}.{format}"
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_model_config = None
    if custom:
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(
                "Custom checkpoint must contain a 'model' state_dict.")
        state_dict = payload["model"]
        checkpoint_model_config = payload.get("model_config")
        checkpoint_encoder = payload.get("encoder")
        if checkpoint_encoder is not None and checkpoint_encoder != encoder.value:
            raise ValueError(
                f"Checkpoint encoder={checkpoint_encoder!r}, but "
                f"--encoder={encoder.value!r}")
        max_depth = float(payload.get("max_depth", 20.0))
        checkpoint_size = payload.get("img_size")
        if checkpoint_size is not None and (height, width) != (
                int(checkpoint_size), int(checkpoint_size)):
            typer.echo(
                f"Warning: checkpoint trained at {checkpoint_size}x"
                f"{checkpoint_size}, exporting at {height}x{width}.")
        architecture_metric = Metric.indoor
    else:
        state_dict = payload
        max_depth = None
        architecture_metric = metric

    config = encoder.get_config(architecture_metric)
    model_args = {
        "encoder": encoder.value,
        "features": config.features,
        "out_channels": config.out_channels,
    }
    if metric is None and not custom:
        model = RelativeDepthAnythingV2(**model_args)
    else:
        if not custom:
            max_depth = 20.0 if metric == Metric.indoor else 80.0
        metric_model_class = MetricDepthAnythingV2
        if custom and checkpoint_model_config:
            architecture_id = checkpoint_model_config.get("architecture_id")
            if architecture_id == "fisheye_metric_dpt_custom_softplus_v1":
                from metric_depth.model_metric_depth.dpt_custom import (
                    DepthAnythingV2 as CustomFileMetricDepthAnythingV2,
                )
                metric_model_class = CustomFileMetricDepthAnythingV2
            elif architecture_id not in (
                None,
                "fisheye_metric_softplus_v1",
            ):
                raise ValueError(
                    f"Unsupported custom checkpoint architecture: "
                    f"{architecture_id!r}. Register its model class in "
                    "transform/dynamo.py before export."
                )
        elif custom and "depth_head.scratch.output_conv2.0.weight" in state_dict:
            # Legacy locally-trained checkpoints predate model_config. Their
            # flat output_conv2 keys identify the official sigmoid DPT head;
            # the newer Softplus head uses output_conv2.0.0/.0.2 instead.
            from metric_depth.model_metric_depth.dpt_vkitti import (
                VKITTIDepthAnythingV2,
            )
            metric_model_class = VKITTIDepthAnythingV2
            typer.echo(
                "Legacy checkpoint head detected: metric sigmoid "
                f"with max_depth={max_depth:g} m"
            )
        model = metric_model_class(**model_args, max_depth=max_depth)
        model_label = "custom" if custom else metric.value
        typer.echo(f"Metric model: {model_label}, max_depth={max_depth:g} m")
    if custom and checkpoint_model_config:
        saved_id = checkpoint_model_config.get("architecture_id")
        expected_id = getattr(model, "ARCHITECTURE_ID", None)
        if saved_id and expected_id and saved_id != expected_id:
            raise ValueError(
                f"Checkpoint architecture={saved_id!r}, but --custom currently "
                f"selects {expected_id!r}. Add/select the matching model class "
                "before exporting this checkpoint."
            )
    typer.echo(f"Loading local checkpoint: {checkpoint}")
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
