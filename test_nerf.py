"""
test_nerf
===========
* You can run the module by `python test_nerf.py`
* You can get losses and PSNR of  test set from this module.  
---------------------------------------------------------
Author: LT H
Github: mofashaoye
"""
import os

import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from torch import nn

import data_loader
import nerf
import render

###################################################################
# NeRF Testing Hyperparameter
# (Use the same hyperparameters as the official implementation)
###################################################################
# OS parameters
DATA_BASE_DIR = "./data/nerf_synthetic/lego/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
RESIZE_COEF = 2
BACKGROUND_W = True

# Model parameters
POS_ENCODE_DIM = 10
VIEW_ENCODE_DIM = 4
DENSE_FEATURES = 256
DENSE_DEPTH = 8
DENSE_FEATURES_FINE = 256
DENSE_DEPTH_FINE = 8

# Render parameters
TNEAR = 2.0
TFAR = 6.0
NUM_SAMPLES = 64
NUM_ISAMPLES = 128
RAY_CHUNK = 32768
SAMPLE5D_CHUNK = 65536
#############################################################################


def calcPSNR(img1, img2) -> float:
    return PSNR(img1, img2)


def calcSSIM(img1, img2) -> float:
    return SSIM(img1, img2, channel_axis=2)


def test_nerf(
    datasets,
    net: nn.Module,
    loss_func,
    split="test",
):
    datasets = datasets[split]
    images, poses, focal = datasets["images"], datasets["poses"], datasets["focal"]
    samples, height, width, channel = images.shape
    # Pass relevant scene parameters, camera parameters,
    # geometric model (NeRF) into Volume Renderer
    renderer = render.VolumeRenderer(
        nerf=net,
        width=width,
        height=height,
        focal=focal,
        tnear=TNEAR,
        tfar=TFAR,
        num_samples=NUM_SAMPLES,
        num_isamples=NUM_ISAMPLES,
        background_w=BACKGROUND_W,
        ray_chunk=RAY_CHUNK,
        sample5d_chunk=SAMPLE5D_CHUNK,
        is_train=False,
        device=DEVICE
    )
    if not os.path.exists("./out/other_imgs"):
        os.mkdir("./out/other_imgs")
    loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    print(f"[Log] number of {split}sets' images: {len(images)}")
    for data_index in tqdm.trange(0, len(images)):
        image = images[data_index]
        pose = poses[data_index, :3, :4]

        # Volume renderer render to obtain predictive rgb (image)
        # Including ray generation, sample coordinates, positional encoding,
        # Hierarchical volume sampling, NeRF(x,d)=(rgb,density),
        # computation of volume rendering equation
        image_hat = renderer.render_image(pose, use_tqdm=False)

        render.save_img(
            image_hat, f"./out/other_imgs/{split}_hat_r_{data_index}.png"
        )

        # Calculate the loss and psnr
        loss = loss_func(
            torch.tensor(image),
            torch.tensor(image_hat)
        ).detach().item()
        loss_sum += loss
        psnr_sum += calcPSNR(image, image_hat)
        ssim_sum += calcSSIM(image, image_hat)

    loss_sum /= len(images)
    psnr_sum /= len(images)
    ssim_sum /= len(images)
    print(f"[Test] NeRF avg Loss in {split}set: {loss_sum}")
    print(f"[Test] NeRF avg PSNR in {split}set: {psnr_sum}")
    print(f"[Test] NeRF avg SSIM in {split}set: {ssim_sum}")


if __name__ == "__main__":
    # get datasets from data_loader module
    datasets = data_loader.load_blender(
        base_dir=DATA_BASE_DIR,
        resize_coef=RESIZE_COEF,
        background_w=BACKGROUND_W
    )
    # get nerf from nerf module
    net = nerf.NeRF(
        pos_dim=POS_ENCODE_DIM,
        view_dim=VIEW_ENCODE_DIM,
        dense_features=DENSE_FEATURES,
        dense_depth=DENSE_DEPTH,
        dense_features_fine=DENSE_FEATURES_FINE,
        dense_depth_fine=DENSE_DEPTH_FINE
    )
    # get loss function from torch.nn module
    loss_func = nn.MSELoss(reduction="mean")
    # Read the model of the largest epoch ever trained from the logs folder
    if os.path.exists("./out") and os.path.exists("./out/model"):
        for root, dirs, files in os.walk("./out/model"):
            if files is not None and len(files) >= 1:
                result = max(files, key=lambda name: int(name[11:-3]))
                path = os.path.join(root, result)
                print(f"[Model] NeRF model Loader from {path}")
                net.load_state_dict(torch.load(path))
    # Start testing
    test_nerf(
        datasets=datasets,
        net=net,
        loss_func=loss_func,
        split="test"
    )
