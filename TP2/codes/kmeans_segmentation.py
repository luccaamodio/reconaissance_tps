#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np


def to_feature_space(image_bgr: np.ndarray, color_space: str) -> np.ndarray:
    space = color_space.upper()
    if space == "RGB":
        arr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif space == "HSV":
        arr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    elif space == "YCRCB":
        arr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    else:
        raise ValueError("color_space must be one of: RGB, HSV, YCrCb")
    return arr


def colorize_labels(labels: np.ndarray, k: int) -> np.ndarray:
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    return palette[labels]


def main():
    parser = argparse.ArgumentParser(description="Unsupervised pixel clustering with K-Means.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--color-space", default="YCrCb", choices=["RGB", "HSV", "YCrCb"])
    parser.add_argument("--out-labels", default="TP2/results/kmeans_labels.png")
    parser.add_argument("--out-centers", default="TP2/results/kmeans_centers.txt")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    feat_img = to_feature_space(image_bgr, args.color_space).astype(np.float32)
    h, w, c = feat_img.shape
    X = feat_img.reshape(-1, c)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    compactness, labels, centers = cv2.kmeans(
        X,
        args.k,
        None,
        crit,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(h, w)
    labels_color = colorize_labels(labels, args.k)

    os.makedirs(os.path.dirname(args.out_labels), exist_ok=True)
    cv2.imwrite(args.out_labels, labels_color)

    os.makedirs(os.path.dirname(args.out_centers), exist_ok=True)
    with open(args.out_centers, "w", encoding="utf-8") as f:
        f.write(f"color_space={args.color_space}\n")
        f.write(f"k={args.k}\n")
        f.write(f"compactness={compactness:.6f}\n")
        f.write("centers:\n")
        for idx, center in enumerate(centers):
            f.write(f"  {idx}: {center.tolist()}\n")

    print(f"Saved labels: {args.out_labels}")
    print(f"Saved centers: {args.out_centers}")


if __name__ == "__main__":
    main()
