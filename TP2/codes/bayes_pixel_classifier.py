#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass

import cv2
import joblib
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainConfig:
    image_path: str
    out_model: str
    out_overlay: str
    color_space: str
    model_type: str


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


def collect_roi_pixels(image_bgr: np.ndarray, class_name: str) -> np.ndarray:
    # Use OpenCV built-in ROI picker to manually collect training pixels.
    rois = cv2.selectROIs(
        windowName=f"Select ROI for class {class_name}. ENTER to validate, ESC to finish.",
        img=image_bgr,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyAllWindows()

    pixels = []
    for x, y, w, h in rois:
        if w <= 0 or h <= 0:
            continue
        patch = image_bgr[y : y + h, x : x + w]
        if patch.size > 0:
            pixels.append(patch.reshape(-1, 3))

    if not pixels:
        return np.empty((0, 3), dtype=np.uint8)
    return np.vstack(pixels)


def train_model(config: TrainConfig):
    image_bgr = cv2.imread(config.image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {config.image_path}")

    print("Step 1/4: Select ROIs for positive class (p).")
    pos_rgb = collect_roi_pixels(image_bgr, "p")
    print("Step 2/4: Select ROIs for negative class (n).")
    neg_rgb = collect_roi_pixels(image_bgr, "n")

    if pos_rgb.shape[0] == 0 or neg_rgb.shape[0] == 0:
        raise RuntimeError("You must provide at least one ROI for each class.")

    pos_bgr = cv2.cvtColor(pos_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3)
    neg_bgr = cv2.cvtColor(neg_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3)

    pos_feat = to_feature_space(pos_bgr.reshape(-1, 1, 3), config.color_space).reshape(-1, 3)
    neg_feat = to_feature_space(neg_bgr.reshape(-1, 1, 3), config.color_space).reshape(-1, 3)

    X = np.vstack([pos_feat, neg_feat]).astype(np.float32)
    y = np.hstack([np.ones(pos_feat.shape[0], dtype=np.int32), np.zeros(neg_feat.shape[0], dtype=np.int32)])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if config.model_type == "naive":
        clf = GaussianNB(var_smoothing=1e-9)
    elif config.model_type == "gaussian":
        # Full-covariance Gaussian model (QDA) to represent multidimensional correlation.
        clf = QuadraticDiscriminantAnalysis(reg_param=1e-4)
    else:
        raise ValueError("model_type must be one of: naive, gaussian")

    clf.fit(Xs, y)

    model = {
        "clf": clf,
        "scaler": scaler,
        "color_space": config.color_space,
        "model_type": config.model_type,
    }
    os.makedirs(os.path.dirname(config.out_model), exist_ok=True)
    joblib.dump(model, config.out_model)

    overlay = predict_overlay(image_bgr, model)
    os.makedirs(os.path.dirname(config.out_overlay), exist_ok=True)
    cv2.imwrite(config.out_overlay, overlay)

    meta_path = os.path.splitext(config.out_model)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_path": config.image_path,
                "color_space": config.color_space,
                "model_type": config.model_type,
                "n_pos": int(pos_feat.shape[0]),
                "n_neg": int(neg_feat.shape[0]),
            },
            f,
            indent=2,
        )

    print(f"Saved model: {config.out_model}")
    print(f"Saved overlay: {config.out_overlay}")
    print(f"Saved metadata: {meta_path}")


def predict_overlay(image_bgr: np.ndarray, model: dict) -> np.ndarray:
    feat_img = to_feature_space(image_bgr, model["color_space"]).astype(np.float32)
    h, w, _ = feat_img.shape
    X = feat_img.reshape(-1, 3)
    Xs = model["scaler"].transform(X)
    yhat = model["clf"].predict(Xs).reshape(h, w)

    overlay = image_bgr.copy()
    mask = yhat == 1
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    overlay[mask] = cv2.addWeighted(overlay, 0.3, tint, 0.7, 0)[mask]
    return overlay


def infer(model_path: str, image_path: str, out_overlay: str):
    model = joblib.load(model_path)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    overlay = predict_overlay(image_bgr, model)
    os.makedirs(os.path.dirname(out_overlay), exist_ok=True)
    cv2.imwrite(out_overlay, overlay)
    print(f"Saved overlay: {out_overlay}")


def main():
    parser = argparse.ArgumentParser(description="Supervised pixel classification with Bayes.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train with manual ROIs")
    p_train.add_argument("--image", required=True)
    p_train.add_argument("--out-model", default="TP2/results/bayes_model.joblib")
    p_train.add_argument("--out-overlay", default="TP2/results/bayes_train_overlay.png")
    p_train.add_argument("--color-space", default="YCrCb", choices=["RGB", "HSV", "YCrCb"])
    p_train.add_argument("--model-type", default="naive", choices=["naive", "gaussian"])

    p_pred = sub.add_parser("predict", help="Predict with an existing model")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--image", required=True)
    p_pred.add_argument("--out-overlay", default="TP2/results/bayes_predict_overlay.png")

    args = parser.parse_args()

    if args.mode == "train":
        cfg = TrainConfig(
            image_path=args.image,
            out_model=args.out_model,
            out_overlay=args.out_overlay,
            color_space=args.color_space,
            model_type=args.model_type,
        )
        train_model(cfg)
    else:
        infer(args.model, args.image, args.out_overlay)


if __name__ == "__main__":
    main()
