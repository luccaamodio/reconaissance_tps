#!/usr/bin/env python3
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def split_components(image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    spaces = {
        "RGB": (rgb, ["R", "G", "B"]),
        "HSV": (hsv, ["H", "S", "V"]),
        "YCrCb": (ycrcb, ["Y", "Cr", "Cb"]),
    }
    return rgb, spaces


def save_grid(image_rgb, spaces, out_path: str):
    fig, axes = plt.subplots(4, 4, figsize=(13, 11))
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original (RGB)")

    row = 1
    for space_name, (arr, names) in spaces.items():
        axes[row, 0].imshow(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), cmap="gray")
        axes[row, 0].set_title(f"{space_name} (preview)")
        for i, comp_name in enumerate(names):
            comp = arr[:, :, i]
            axes[row, i + 1].imshow(comp, cmap="gray")
            axes[row, i + 1].set_title(f"{space_name}-{comp_name}")
        row += 1

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Display color-space components.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--out",
        default="TP2/results/components.png",
        help="Output image path",
    )
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    image_rgb, spaces = split_components(image_bgr)
    save_grid(image_rgb, spaces, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
