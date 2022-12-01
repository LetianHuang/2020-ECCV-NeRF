"""
data_loader
===========
Provides
* Some datasets import functions for the NeRF project
---------------------------------------------------------
Author: LT H
Github: mofashaoye
"""
import json
import os

import cv2 as cv
import numpy as np


def load_blender(base_dir, resize_coef=2, background_w=True) -> dict:
    """
    Load the datasets whose type is blender
    =======================================
    Inputs:
        base_dir : str      The root directory of the datasets
        resize_coef : int   The transformation coefficient of the image size of the datasets
        background_w : bool Whether to remove the A channel from the RGBA image 
                            and set the background to white
    Outputs:
        datasets : dict     Datasets containing train set, validation set, and test set.
                            Each set contains three types of data, the image, 
                            the camera transformation matrix, and the focal length 
                            whose type are NumPy.NDArray
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        with open(os.path.join(base_dir, f"transforms_{split}.json"), "r") as f:
            dataset = json.load(f)
        images, poses = [], []
        for frame in dataset["frames"]:
            image = cv.imread(
                os.path.join(base_dir, frame["file_path"] + ".png"),
                cv.IMREAD_UNCHANGED
            )
            height, width, channel = image.shape
            if resize_coef >= 2:
                image = cv.resize(
                    image, (width // resize_coef, height // resize_coef), interpolation=cv.INTER_AREA
                )
            images.append(image)
            poses.append(np.array(frame["transform_matrix"]))
        images = np.array(images, dtype=np.float32) / 255.0
        if background_w:
            images = (images[..., :3] - 1.0) * images[..., -1:] + 1.0
        poses = np.array(poses, dtype=np.float32)
        # The focal length is calculated according to the field of view (FOV_x)
        # and the image width
        samples, height, width, channel = images.shape
        fov = float(dataset["camera_angle_x"])
        focal = 0.5 * width / np.tan(0.5 * fov) 
        
        datasets[split] = dict(images=images, poses=poses, focal=focal)
    return datasets
