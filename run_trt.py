"""Depth Anything V2 TensorRT engine inference for images and videos."""

import argparse
import cv2
import glob
import os
from utils import DPTTrt, preprocess, postprocess
from frame_mask_utils import parse_bool, prepare_frame


def apply_valid_mask(image, valid_mask):
    if valid_mask is None:
        return image
    if valid_mask.shape != image.shape[:2]:
        valid_mask = cv2.resize(
            valid_mask.astype("uint8"),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    image[~valid_mask] = 0
    return image


def process_video(session, args, filenames):
    for k, filename in enumerate(filenames):
        print(f"Progress {k+1}/{len(filenames)}: {filename}")

        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

        output_path = os.path.join(
            args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".mp4"
        )
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            (frame_width, frame_height),
        )

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            valid_mask = None
            inference_frame = raw_frame
            if args.frame_mask:
                inference_frame, valid_mask = prepare_frame(raw_frame)
            image = preprocess(inference_frame, args.input_size)
            depth = session.inference(image)

            depth = postprocess(
                (frame_width, frame_height), depth, args.grayscale, args.crop_region
            )
            depth = apply_valid_mask(depth, valid_mask)

            if args.crop_region is not None:
                raw_frame_copy = raw_frame.copy()
                x, y, w, h = args.crop_region.split(" ")
                x, y, w, h = int(x), int(y), int(w), int(h)
                raw_frame_copy[y : y + h, x : x + w, :] = depth
                out.write(raw_frame_copy)
            else:
                out.write(depth)

        raw_video.release()
        out.release()


def process_images(session, args, filenames):
    for k, filename in enumerate(filenames):
        print(f"Processing {k+1}/{len(filenames)}: {filename}")

        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Skip unreadable image: {filename}")
            continue
        valid_mask = None
        inference_image = raw_image
        if args.frame_mask:
            inference_image, valid_mask = prepare_frame(raw_image)
        image = preprocess(inference_image, args.input_size)
        depth = session.inference(image)

        depth = postprocess(
            (raw_image.shape[1], raw_image.shape[0]),
            depth,
            args.grayscale,
            args.crop_region,
        )
        depth = apply_valid_mask(depth, valid_mask)

        if args.crop_region is not None:
            raw_image_copy = raw_image.copy()
            x, y, w, h = args.crop_region.split(" ")
            x, y, w, h = int(x), int(y), int(w), int(h)
            raw_image_copy[y : y + h, x : x + w, :] = depth
            output_image = raw_image_copy
        else:
            output_image = depth

        output_path = os.path.join(args.outdir, os.path.basename(filename))
        cv2.imwrite(output_path, output_image)


def main(args):
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"TensorRT engine not found: {args.model_path}")
    session = DPTTrt(args)

    if os.path.isfile(args.input_path):
        if args.input_path.endswith("txt"):
            with open(args.input_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.input_path]
    else:
        filenames = glob.glob(os.path.join(args.input_path, "**/*"), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    if args.input_type == "video":
        process_video(session, args, filenames)
    elif args.input_type == "image":
        process_images(session, args, filenames)
    else:
        raise ValueError("Unsupported input type. Choose either 'video' or 'image'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything V2")

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input video/image or directory containing videos/images.",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        required=True,
        choices=["video", "image"],
        help="Input type: video or image.",
    )
    parser.add_argument(
        "--input_size", type=int, default=518, help="Input size for the model."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./vis_output",
        help="Output directory for the processed videos/images.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder type for the model.",
    )
    parser.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="Do not apply colorful palette.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/depth_anything_v2_vits_dynamic.engine",
        help="Path to the serialized TensorRT .engine file.",
    )
    parser.add_argument(
        "--crop-region",
        type=str,
        default=None,
        help="Get value x,y,w,h with space i.e. 0 0 500 500",
    )
    parser.add_argument(
        "--frame-mask",
        type=parse_bool,
        default=False,
        metavar="{True,False}",
        help="Fill the black outer frame before inference and mask it afterward.",
    )
    args = parser.parse_args()
    main(args)
