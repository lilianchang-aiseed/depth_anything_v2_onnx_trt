"""Run Depth Anything V2 ONNX inference without model export code.

Example:
    # One image
    python3 run_onnx.py infer \
        checkpoints/depth_anything_v2_vits_dynamic.onnx \
        --image input.png --output depth_color.png

    # All images in a folder
    python3 run_onnx.py infer \
        checkpoints/depth_anything_v2_vits_dynamic.onnx \
        --img-path ./images --out-dir ./depth_results \
        --save-numpy

    # Read the ``left`` image directly from every NPZ in a folder
    python3 run_onnx.py infer checkpoints/depth_anything_v2_vits_dynamic.onnx \
        --img-path ../sml/dataset/train_1_0-5/train_1_0-5_supp_2 \
        --out-dir ../sml/dataset/train_1_0-5/train_1_0-5_dav2_2 \
        --save-numpy

    # Use Da_v2 result, get or generate da_v2 depth inference result beforehand
    python3 sml/fit_global_scale_shift.py \
        --data "$COMMON_SHARE/bags/nx-2.0/0825/train_1_0-5_dataset" \
        --da-dir sml/train_1_0-5_dav2 \
        --pair 1_0 \
        --out sml/fitted_scale_shift_1_0.yaml
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import cv2
import matplotlib
import numpy as np
import typer

from frame_mask_utils import prepare_frame


class InferenceDevice(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


class BooleanValue(str, Enum):
    false = "False"
    true = "True"


app = typer.Typer()


@app.callback()
def callback():
    """Depth Anything V2 ONNX Runtime inference CLI."""


def multiple_of_14(value: int) -> int:
    if value % 14 != 0:
        raise typer.BadParameter("Value must be a multiple of 14.")
    return value


def colorize_depth(depth: np.ndarray, cmap) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size and float(finite.max()) > float(finite.min()):
        normalized = (depth - finite.min()) / (finite.max() - finite.min())
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    else:
        normalized = np.zeros(depth.shape, dtype=np.float32)
    return (cmap(np.clip(normalized, 0, 1))[..., :3] * 255)[:, :, ::-1].astype(np.uint8)


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
            "--img-path",
            exists=True,
            readable=True,
            help="Path to one image or a folder of images.",
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
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            file_okay=False,
            help="Output directory. Required when --img-path is a folder.",
        ),
    ] = None,
    save_numpy: Annotated[
        bool,
        typer.Option("--save-numpy", help="Also save raw float32 DA-V2 output as NPY."),
    ] = False,
    npz_key: Annotated[
        str,
        typer.Option("--npz-key", help="Image field used when an input file is NPZ."),
    ] = "left",
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Resume a folder run by skipping complete existing outputs.",
        ),
    ] = False,
    frame_mask: Annotated[
        BooleanValue,
        typer.Option(
            "--frame-mask",
            help="Fill the black outer frame before inference and mask it afterward.",
        ),
    ] = BooleanValue.false,
    frame_mask_debug_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--frame-mask-debug-dir",
            file_okay=False,
            help="Save padded input and before/after-mask visualizations.",
        ),
    ] = None,
):
    """Depth-Anything V2 inference using ONNXRuntime. No dependency on PyTorch."""
    import onnxruntime as ort

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npz"}
    if image_path.is_dir():
        if output_path is not None:
            raise typer.BadParameter("Use --out-dir, not --output, for a folder.")
        if output_dir is None:
            raise typer.BadParameter("--out-dir is required for a folder.")
        image_paths = sorted(
            path for path in image_path.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
    else:
        image_paths = [image_path]
    if not image_paths:
        raise typer.BadParameter(f"No supported images found in {image_path}")
    if save_numpy and output_path is None and output_dir is None:
        raise typer.BadParameter("--save-numpy requires --output or --out-dir.")
    frame_mask_enabled = frame_mask == BooleanValue.true
    if frame_mask_debug_dir is not None and not frame_mask_enabled:
        raise typer.BadParameter("--frame-mask-debug-dir requires --frame-mask True.")

    # Create the ONNX session once and reuse it for every image.
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
    ort_input = session.get_inputs()[0].name
    ort_output = session.get_outputs()[0].name
    output_device = (
        "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    )
    # cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    cmap = matplotlib.colormaps.get_cmap("inferno")


    processed = 0
    skipped = 0
    for index, path in enumerate(image_paths, 1):
        destination = output_path
        if output_dir is not None:
            relative = path.relative_to(image_path) if image_path.is_dir() else Path(path.name)
            destination = output_dir / relative.with_suffix(".jpg")
        if (
            skip_existing
            and destination is not None
            and destination.is_file()
            and (not save_numpy or destination.with_suffix(".npy").is_file())
        ):
            skipped += 1
            continue

        if path.suffix.lower() == ".npz":
            try:
                with np.load(path, allow_pickle=False) as sample:
                    raw_image = np.asarray(sample[npz_key]).copy()
            except (KeyError, ValueError) as error:
                typer.echo(f"Skip unreadable NPZ image {path}: {error}")
                continue
        else:
            raw_image = cv2.imread(str(path))
        if raw_image is None:
            typer.echo(f"Skip unreadable image: {path}")
            continue
        if raw_image.ndim == 2:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        if raw_image.ndim != 3 or raw_image.shape[2] != 3:
            typer.echo(f"Skip unsupported image shape {raw_image.shape}: {path}")
            continue
        h, w = raw_image.shape[:2]
        valid_mask = None
        inference_image = raw_image
        if frame_mask_enabled:
            inference_image, valid_mask = prepare_frame(raw_image)
        image = cv2.cvtColor(inference_image, cv2.COLOR_BGR2RGB) / 255.0
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = image.transpose(2, 0, 1)[None].astype("float32")

        binding = session.io_binding()
        binding.bind_cpu_input(ort_input, image)
        binding.bind_output(ort_output, output_device)
        session.run_with_iobinding(binding)
        raw_depth = binding.get_outputs()[0].numpy().squeeze().astype(np.float32)
        raw_depth = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        raw_depth_before_mask = raw_depth.copy()
        if valid_mask is not None:
            raw_depth = np.where(valid_mask, raw_depth, np.nan).astype(np.float32)
        depth = colorize_depth(raw_depth, cmap)

        if frame_mask_debug_dir is not None:
            debug_sample_dir = frame_mask_debug_dir / path.stem
            debug_sample_dir.mkdir(parents=True, exist_ok=True)
            before_vis = colorize_depth(raw_depth_before_mask, cmap)
            after_vis = colorize_depth(raw_depth, cmap)
            cv2.imwrite(str(debug_sample_dir / "01_padded_input.jpg"), inference_image)
            cv2.imwrite(str(debug_sample_dir / "02_depth_before_valid_mask.jpg"), before_vis)
            cv2.imwrite(str(debug_sample_dir / "03_depth_after_valid_mask.jpg"), after_vis)
            panels = [inference_image.copy(), before_vis, after_vis]
            labels = ["Padded input", "Depth before mask", "Depth after mask"]
            for panel, label in zip(panels, labels):
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
                cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(debug_sample_dir / "frame_mask_montage.jpg"),
                        cv2.hconcat(panels))

        if destination is None:
            cv2.imshow("depth", depth)
            cv2.waitKey(0)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), depth):
                raise RuntimeError(f"Failed to save {destination}")
            if save_numpy:
                np.save(destination.with_suffix(".npy"), raw_depth)
        processed += 1
        if index == 1 or index % 100 == 0 or index == len(image_paths):
            typer.echo(f"Progress {index}/{len(image_paths)}: {path}")
    typer.echo(f"Finished: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    app()
