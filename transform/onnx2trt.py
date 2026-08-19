# scripts from https://github.com/vnk8071/depth-anything-triton-deepstream
# usage: python3 transform/onnx2trt.py --mode fp16 --workspace 2

"""
python3 transform/onnx2trt.py \
  --onnx checkpoints/depth_anything_v2_vits_dynamic.onnx \
  --output checkpoints/depth_anything_v2_vits_dynamic.engine \
  --mode fp16 \
  --workspace 2 \
  --min-shape 1x3x280x280 \
  --opt-shape 1x3x518x518 \
  --max-shape 1x3x700x700
"""

import sys
import warnings
import argparse
from pathlib import Path

try:
    import tensorrt as trt
except ImportError as exc:
    raise SystemExit(
        "TensorRT Python bindings are not installed in this Python environment. "
        "Run this script in the NVIDIA TensorRT/JetPack environment."
    ) from exc


TRANSFORM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRANSFORM_DIR.parent
DEFAULT_ONNX = PROJECT_DIR / "checkpoints" / "depth_anything_v2_vits.onnx"
DEFAULT_ENGINE = PROJECT_DIR / "checkpoints" / "depth_anything_v2_vits.engine"
DEFAULT_MIN_SHAPE = (1, 3, 280, 280)
DEFAULT_OPT_SHAPE = (1, 3, 518, 518)
DEFAULT_MAX_SHAPE = (1, 3, 700, 700)

warnings.simplefilter("ignore", category=DeprecationWarning)


def parse_shape(value):
    try:
        shape = tuple(int(item) for item in value.lower().replace("x", ",").split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shape must use NCHW integers, for example 1x3x518x518"
        ) from exc
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape must contain four positive NCHW values")
    if shape[1] != 3:
        raise argparse.ArgumentTypeError("Depth Anything input must have 3 channels")
    if shape[2] % 14 or shape[3] % 14:
        raise argparse.ArgumentTypeError("height and width must be multiples of 14")
    return shape


def shape_text(shape):
    return "x".join(str(dimension) for dimension in shape)


class EngineBuilder:
    def __init__(
        self,
        onnx_file_path,
        save_path,
        mode,
        log_level="ERROR",
        max_workspace_size=1,
        strict_type_constraints=False,
        int8_calibrator=None,
        min_shape=DEFAULT_MIN_SHAPE,
        opt_shape=DEFAULT_OPT_SHAPE,
        max_shape=DEFAULT_MAX_SHAPE,
        **kwargs,
    ):
        """build TensorRT model from onnx model.
        Args:
            onnx_file_path (string or io object): onnx model name
            save_path (string): tensortRT serialization save path
            mode (string): Whether or not FP16 or Int8 kernels are permitted during engine build.
            log_level (string, default is ERROR): tensorrt logger level, now
                INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
            max_workspace_size (int, default is 1):
                The maximum GPU temporary memory which the ICudaEngine can use at
                execution time. default is 1GB.
            strict_type_constraints (bool, default is False):
                When strict type constraints is set, TensorRT will choose
                the type constraints that conforms to type constraints.
                If the flag is not enabled higher precision
                implementation may be chosen if it results in higher performance.
            int8_calibrator (volksdep.calibrators.base.BaseCalibrator, default is None):
            calibrator for int8 mode,
                if None, default calibrator will be used as calibration data."""
        self.onnx_file_path = Path(onnx_file_path).expanduser().resolve()
        self.save_path = Path(save_path).expanduser().resolve()
        self.mode = mode.lower()
        assert self.mode in [
            "fp32",
            "fp16",
            "int8",
        ], f"mode should be in ['fp32', 'fp16', 'int8'], but got {mode}"

        self.trt_logger = trt.Logger(getattr(trt.Logger, log_level))
        self.builder = trt.Builder(self.trt_logger)
        self.network = None
        self.max_workspace_size = max_workspace_size
        self.strict_type_constraints = strict_type_constraints
        self.int8_calibrator = int8_calibrator
        self.min_shape = tuple(min_shape)
        self.opt_shape = tuple(opt_shape)
        self.max_shape = tuple(max_shape)
        for minimum, optimum, maximum in zip(
            self.min_shape, self.opt_shape, self.max_shape
        ):
            if not minimum <= optimum <= maximum:
                raise ValueError(
                    "Every profile dimension must satisfy min <= opt <= max"
                )
        self.dynamic_inputs = []

    def create_network(self, **kwargs):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(self.network, self.trt_logger)
        with self.onnx_file_path.open("rb") as f:
            print(f"Beginning ONNX parsing: {self.onnx_file_path}")
            flag = parser.parse(f.read())
        if not flag:
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")

        print("Completed parsing of ONNX file.")

        # Check if network has outputs, if not, raise an error
        if self.network.num_outputs == 0:
            raise RuntimeError("No output tensors found in the ONNX model.")

        for index in range(self.network.num_inputs):
            tensor = self.network.get_input(index)
            if any(dimension < 0 for dimension in tensor.shape):
                self.dynamic_inputs.append(tensor.name)
                print(f"Dynamic input: {tensor.name} {tuple(tensor.shape)}")

    def create_engine(self):
        config = self.builder.create_builder_config()
        workspace_bytes = int(self.max_workspace_size * (1 << 30))
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, workspace_bytes
            )
        else:
            config.max_workspace_size = workspace_bytes
        if self.mode == "fp16":
            assert self.builder.platform_has_fast_fp16, "not support fp16"
            config.set_flag(trt.BuilderFlag.FP16)
            # builder.fp16_mode = True
        if self.mode == "int8":
            assert self.builder.platform_has_fast_int8, "not support int8"
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = self.int8_calibrator
            # builder.int8_mode = True
            # builder.int8_calibrator = int8_calibrator

        if self.strict_type_constraints:
            strict_flag = getattr(
                trt.BuilderFlag,
                "STRICT_TYPES",
                getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None),
            )
            if strict_flag is not None:
                config.set_flag(strict_flag)

        if self.dynamic_inputs:
            profile = self.builder.create_optimization_profile()
            for input_name in self.dynamic_inputs:
                accepted = profile.set_shape(
                    input_name,
                    self.min_shape,
                    self.opt_shape,
                    self.max_shape,
                )
                if accepted is False:
                    raise RuntimeError(
                        f"TensorRT rejected the profile for {input_name}"
                    )
                print(
                    f"Profile {input_name}: min={self.min_shape}, "
                    f"opt={self.opt_shape}, max={self.max_shape}"
                )
            config.add_optimization_profile(profile)

        print(
            f"Building engine from {self.onnx_file_path}; this may take a while..."
        )
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.builder, "build_serialized_network"):
            serialized_engine = self.builder.build_serialized_network(
                self.network, config
            )
            if serialized_engine is None:
                raise RuntimeError("TensorRT failed to build the serialized engine")
            engine_bytes = bytes(serialized_engine)
        else:
            engine = self.builder.build_engine(self.network, config)
            if engine is None:
                raise RuntimeError("TensorRT failed to build the engine")
            engine_bytes = engine.serialize()
        print("Created engine successfully.")

        print(f"Saving TRT engine file to path {self.save_path}")
        with self.save_path.open("wb") as f:
            f.write(engine_bytes)
        print(f"Engine saved to {self.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
        help=f"Input ONNX model (default: {DEFAULT_ONNX})",
    )
    parser.add_argument(
        "--mode",
        default="fp16",
        help="use fp32 or fp16 or int8, default: fp16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ENGINE,
        help=f"Output TensorRT engine (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--workspace", default=2, type=float, help="Builder workspace size in GiB"
    )
    parser.add_argument(
        "--min-shape",
        type=parse_shape,
        default=DEFAULT_MIN_SHAPE,
        help=f"Minimum NCHW profile shape (default: {shape_text(DEFAULT_MIN_SHAPE)})",
    )
    parser.add_argument(
        "--opt-shape",
        type=parse_shape,
        default=DEFAULT_OPT_SHAPE,
        help=f"Optimal NCHW profile shape (default: {shape_text(DEFAULT_OPT_SHAPE)})",
    )
    parser.add_argument(
        "--max-shape",
        type=parse_shape,
        default=DEFAULT_MAX_SHAPE,
        help=f"Maximum NCHW profile shape (default: {shape_text(DEFAULT_MAX_SHAPE)})",
    )
    args = parser.parse_args()
    onnx_file_path = args.onnx.expanduser().resolve()
    engine_file = args.output.expanduser().resolve()
    if not onnx_file_path.is_file():
        parser.error(f"ONNX model not found: {onnx_file_path}")

    # 执行转化
    builder = EngineBuilder(
        onnx_file_path,
        engine_file,
        args.mode,
        log_level="WARNING",
        max_workspace_size=args.workspace,
        min_shape=args.min_shape,
        opt_shape=args.opt_shape,
        max_shape=args.max_shape,
    )
    builder.create_network()
    builder.create_engine()
