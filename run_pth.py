import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from depth_anything_v2.dpt import (
    DepthAnythingV2 as RelativeDepthAnythingV2,
)
from metric_depth.depth_anything_v2.dpt import (
    DepthAnythingV2 as MetricDepthAnythingV2,
)
from frame_mask_utils import parse_bool, prepare_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--metric', action='store_true', help='decide which vits weight file to use')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--frame-mask', type=parse_bool, default=False,
                        metavar='{True,False}',
                        help='fill the black outer frame before inference and mask it afterward')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    if args.metric:
        depth_anything = MetricDepthAnythingV2(
            **model_configs[args.encoder],
            max_depth=80.0,  # VKITTI outdoor
        )
        model_path = (
            f"checkpoints/depth_anything_v2_metric_vkitti_{args.encoder}.pth"
        )
    else:
        depth_anything = RelativeDepthAnythingV2(
            **model_configs[args.encoder],
        )
        model_path = (
            f"checkpoints/depth_anything_v2_{args.encoder}.pth"
        )

    depth_anything.load_state_dict(
        torch.load(model_path, map_location="cpu"),
        strict=True,
    )

    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f'Skip unreadable image: {filename}')
            continue

        valid_mask = None
        inference_image = raw_image
        if args.frame_mask:
            inference_image, valid_mask = prepare_frame(raw_image)

        #test fisheye crop 
        # raw_image = raw_image[205:-205, 154:-154, :]
        # raw_image = cv2.resize(raw_image, (320, 320))
        # raw_image = cv2.resize(raw_image[205:-205, 159:-159, :], (320, 320))
        
        depth = depth_anything.infer_image(inference_image, args.input_size)
        if valid_mask is not None:
            depth = np.where(valid_mask, depth, np.nan).astype(np.float32)

        # ---from original repo ./metric_depth/run.py---
        # if args.save_numpy:
        #     output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
        #     np.save(output_path, depth)

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.jpg')
        fig, axes = plt.subplots(1, 1 if args.pred_only else 2, figsize=(8, 4), constrained_layout=True)
        depth_ax = axes if args.pred_only else axes[1]

        if not args.pred_only:
            axes[0].imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
            axes[0].axis('off')
            
        vmin, vmax = ((0.0, 80.0) if args.metric else
                      (float(np.nanmin(depth)), float(np.nanmax(depth))))
        image = depth_ax.imshow(depth, cmap='gray' if args.grayscale else 'Spectral_r', vmin=vmin, vmax=vmax)
        depth_ax.axis('off')
        fig.colorbar(image, ax=depth_ax, label='Depth (m)' if args.metric else 'Relative depth')
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
