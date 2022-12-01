"""
render
======
Provides
* Volume Rendering with Radiance Fields in the paper
    --- implemented by VolumeRenderer
    --- cast rays and calculate the integral to solve the disintegration rendering equation
    --- The integral is computed by Monte Carlo integral method 
        Compute the dot product of the tensors through the sampling points
* save image
    --- using OpenCV
-------------------------------------------------------------------------------------
Author: LT H
Github: mofashaoye
"""
import cv2 as cv
import numpy as np
import torch
import tqdm


def get_screen_batch(
    height: int,
    width: int,
    render_batch_size: int,
    bias: int,
    device=torch.device("cpu"),
) -> torch.Tensor:
    """
    Get Screen Coordinates Batch
    ============================
    Inputs:
        height              : int                   scene's height
        width               : int                   scene's width
        render_batch_size   : int                   batch size of rendering
        bias                : int                   bias from [0,0]
        device              : torch.device          Output's device
    Output:
        coords              : torch.Tensor          batch coordinates of rendering
    """
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0, height - 1, height, device=device),
            torch.linspace(0, width - 1, width, device=device)
        ),
        dim=-1
    )
    coords = torch.reshape(coords, (-1, 2))
    coords = coords[bias: bias + render_batch_size].long()
    return coords


def save_img(img: np.ndarray, path):
    """
    Save Image Using OpenCV
    """
    cv.imwrite(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))


class VolumeRenderer:
    """
    VolumeRenderer
    ==============
    Render the scene represented by NeRF using volume rendering
    """

    def __init__(self, nerf, width, height, focal, tnear, tfar, num_samples, num_isamples, background_w, ray_chunk, sample5d_chunk, is_train, device) -> None:
        """
        VolumeRender Constructor
        ========================
        nerf            : Neural Radiance Fields (contains encoding, coarse net and fine net)
        width           : width of the rendering scene
        height          : height of the rendering scene
        focal           : focal length
        tnear           : $t_n$ in paper
        tfar            : $t_f$ in paper
        num_samples     : number of sampling
        num_isamples    : number of Hierarchical volume sampling
        ray_chunk       : chunk of ray casting
        sample5d_chunk  : chunk of net
        background_w    : whether or not transform image's background to white
        device          : device of the whole volume renderer
        is_train        : train or eval(just rendering or test)
        """
        self.nerf = nerf                        # Neural Radiance Fields (contains encoding, coarse net and fine net)
        self.width = width                      # width of the rendering scene
        self.height = height                    # height of the rendering scene
        self.focal = focal                      # focal length
        self.tnear = tnear                      # $t_n$ in paper
        self.tfar = tfar                        # $t_f$ in paper
        self.num_samples = num_samples          # number of sampling
        self.num_isamples = num_isamples        # number of Hierarchical volume sampling
        self.ray_chunk = ray_chunk              # chunk of ray casting
        self.sample5d_chunk = sample5d_chunk    # chunk of net
        self.background_w = background_w        # whether or not transform image's background to white
        self.device = device                    # device of the whole volume renderer

        self.nerf.to(device)                    # to device(CPU or GPU)
        
        self.train(is_train)                    # train or eval(just rendering or test)

    def train(self, is_train):
        """ Train or Eval(just rendering or test) """
        self.is_train = is_train
        self.nerf.train(is_train)

    def _generate_rays(self, camera2world: torch.Tensor):
        """
        Generate Camera Rays
        ====================
        Rays Directions: 
            first generate pixel coordinates [0,W-1] x [0, H-1]
            then transform the pixel coordinates to world coordinates
            Screen Space => Camera Space => World Space
        Rays Origins: 
            get from the Camera2World matrix
        """
        # Generating pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0, self.width - 1, self.width, device=self.device),
            torch.linspace(0, self.height - 1, self.height, device=self.device)
        )
        i = i.t()
        j = j.t()
        # pixel coordinates to camera coordinates
        # and camera coordinates to the world coordinates
        rays_d = torch.matmul(
            torch.stack([
            (i - 0.5 * self.width) / self.focal,
            -(j - 0.5 * self.height) / self.focal,
            -torch.ones_like(i, device=self.device)
            ], dim=-1),
            camera2world[:3, :3].t()
        )
        # Camera's World Coordinate
        rays_o = camera2world[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def _sample(self, rays):
        """ 
        Volume Sampling 
        ===============
        **we partition [tn,tf ] into N evenly-spaced bins and
        then draw one sample uniformly at random from within each bin**
        Input:
            rays: (rays_o, rays_d) tuple[tensor, tensor]
        Output:
            pos_locs: tensor [num_rays, num_samples, 3]  spatial locations used for the input of Coarse Net
            t: tensor       [num_rays, num_samples]  t of sampling 
        """
        # rays_o.shape=[num_rays, 3], rays_d.shape=[num_rays, 3]
        rays_o, rays_d = rays
        if self.is_train:
            t = torch.linspace(
                float(self.tnear), float(self.tfar), steps=self.num_samples + 1,
                device=self.device
            )
            t = t.expand((*rays_o.shape[:-1], self.num_samples + 1))
            lower, upper = t[..., :-1], t[..., 1:]
            # Linear interpolation [lower, upper] => [num_rays, num_samples]
            t = torch.lerp(lower, upper, torch.rand_like(lower))
        else:
            t = torch.linspace(
                float(self.tnear), float(self.tfar), steps=self.num_samples,
                device=self.device
            )
            t = t.expand((*rays_o.shape[:-1], self.num_samples))
        pos_locs = rays_o[..., None, :] + \
            rays_d[..., None, :] * t[..., :, None]
        # [num_rays, num_samples, 3]

        return pos_locs, t

    def _parse_voxels(self, voxels, t_vals, rays_d) -> dict:
        """
        The volume rendering integral equation
        was calculated by Monte Carlo integral method
        Inputs:
            voxels: tensor [num_rays, num_samples, 4]   results of NN forward
            t_vals: tensor [num_rays, num_samples]      t of sampling 
            rays_d: tensor [num_rays, 3]                rays' directions
        Output:
            rbg_map and cdf_map
            rbg_map: tensor [num_rays, 3]               RGB map of the rendering scene
            cdf_map: tensor [num_rays, num_samples + 1] CDF map (Cumulative Distribution Function)
        """
        t_delta = t_vals[..., 1:] - t_vals[..., :-1]
        t_delta = torch.cat(
            (t_delta, torch.tensor(
                [1e10], device=self.device).expand_as(t_delta[..., :1])),
            dim=-1
        )  # [num_rays, num_samples]
        t_delta = t_delta * torch.norm(rays_d[..., None, :], dim=-1)
        # [num_rays, num_samples, 3]
        c_i = torch.sigmoid(voxels[..., :3])
        # [num_rays, num_samples]
        alpha_i = 1 - torch.exp(-torch.relu(voxels[..., 3]) * t_delta)
        w_i = alpha_i * torch.cumprod(
            torch.cat(
                (torch.ones(
                    (*alpha_i.shape[:-1], 1), device=self.device), 1.0 - alpha_i + 1e-10),
                dim=-1
            ),
            dim=-1
        )[:, :-1]  # [num_rays, num_samples]
        rgb_map = torch.sum(
            w_i[..., None] * c_i,
            dim=-2,  # num_samples
            keepdim=False
        )  # [num_rays, 3]
        acc_opacity_map = torch.sum(w_i, dim=-1, keepdim=False)

        pdf_map = w_i[..., 1:-1] + 1e-5  # prevent nans
        pdf_map = pdf_map / torch.sum(pdf_map, -1, keepdim=True)
        cdf_map = torch.cumsum(pdf_map, dim=-1)
        cdf_map = torch.cat(
            (torch.zeros_like(cdf_map[..., :1]), cdf_map), dim=-1
        )

        if self.background_w:
            rgb_map = rgb_map + (1.0 - acc_opacity_map[..., None])

        return dict(
            rgb_map=rgb_map,
            cdf_map=cdf_map
        )

    def _hierarchical_sample(self, rays, t_vals: torch.Tensor, cdf: torch.Tensor):
        """
        Hierarchical volume sampling in paper
        =====================================
        **We sample a second set of Nf locations from this distribution
        using inverse transform sampling, evaluate our “fine” network at the union of the
        first and second set of samples, and compute the final rendered color of the ray Cf (r) using Eqn. 3 but using all Nc + Nf samples.**
        Inputs:
            rays: (rays_o, rays_d) tuple[tensor, tensor]
            t_vals: tensor [num_rays, num_samples] t of the result of uniform sampling (the function `_sample`)
            cdf: tensor [num_rays, num_samples + 1] CDF map (Cumulative Distribution Function)
        Output:
            pos_locs: tensor [num_rays, num_samples + num_isamples, 3]  spatial locations used for the input of Fine Net
            t_vals: tensor [num_rays, num_samples + num_isamples] t of sampling and hierarchical sampling
        """
        rays_o, rays_d = rays
        t_vals_mid = (t_vals[..., :-1] + t_vals[..., 1:]) * 0.5

        u = torch.rand(
            (*cdf.shape[:-1], self.num_isamples),
            device=self.device
        ).contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1, device=self.device), inds-1)
        above = torch.min(
            (cdf.shape[-1]-1) * torch.ones_like(inds, device=self.device), inds)
        inds_g = torch.stack([below, above], -1)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(t_vals_mid.unsqueeze(
            1).expand(matched_shape), 2, inds_g)
        denom = (cdf_g[..., 1]-cdf_g[..., 0])
        denom = torch.where(
            denom < 1e-5, torch.ones_like(denom, device=self.device), denom)
        t = (u-cdf_g[..., 0])/denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
        samples = samples.detach()

        t_vals, _ = torch.sort(torch.cat([t_vals, samples], -1), -1)

        pos_locs = rays_o[..., None, :] + \
            rays_d[..., None, :] * t_vals[..., :, None]
        # [num_rays, num_samples + num_isamples, 3]

        return pos_locs, t_vals

    def _cast_rays(self, rays, view_dirs):
        """
        Rays Casting
        ============
        1. sample spatial locations and t for coarse net of NeRF
        2. using spatial locations and view directions as inputs of the coarse net of NeRF 
            to get voxels (rgb density) of the spatial locations
        3. parse voxels (rgb density) to rgb map of the scene (calculate integral) 
            and cdf of the sampling t
        4. hierarchical sample spatial locations and t for fine net of NeRF
        5. using spatial locations obtained by [4] and view directions as inputs of the fine net of NeRF 
            to get voxels (rgb density) of the spatial locations
        6. return the results (rgb map) of [2] and [5]
        Inputs:
            rays: (rays_o, rays_d) 
            view_dirs: tensor  view directions [num_rays, 3]
        Outputs:
            rgb map of [2] and [5]
        """
        rays_o, rays_d = rays

        coarse_nerf_locs, t_coarse = self._sample((rays_o, rays_d))
        coarse_voxels = self._voxel_sample5d(
            torch.cat((coarse_nerf_locs,
                       view_dirs[..., None, :].expand_as(coarse_nerf_locs)), dim=-1),
            "coarse"
        )
        coarse_info = self._parse_voxels(coarse_voxels, t_coarse, rays_d)
        fine_nerf_locs, t_fine = self._hierarchical_sample(
            (rays_o, rays_d), t_coarse, coarse_info["cdf_map"]
        )
        fine_voxels = self._voxel_sample5d(
            torch.cat((fine_nerf_locs,
                       view_dirs[..., None, :].expand_as(fine_nerf_locs)), dim=-1),
            "fine"
        )
        fine_info = self._parse_voxels(fine_voxels, t_fine, rays_d)
        return coarse_info["rgb_map"], fine_info["rgb_map"]

    def _voxel_sample5d(self, x: torch.Tensor, net_type) -> torch.Tensor:
        """
        sample voxels (rgb density) using NeRF
        =======================================
        Inputs:
            x : spatial locations and view directions
            net_type: "coarse" or "fine" to select coarse net or fine net to forward
        Outputs:
            rgb + density
        """
        self.nerf.net_(net_type)
        if self.sample5d_chunk is None or self.sample5d_chunk <= 1:
            return self.nerf(x)
        return torch.cat(
            [self.nerf(x[i:i + self.sample5d_chunk])
             for i in range(0, x.shape[0], self.sample5d_chunk)],
            dim=0
        )

    def render(self, pose, select_coords=None):
        """
        render some pixels of the scene 
        (mainly used in training NeRF and `self.render_image`)
        Inputs:
            pose: camera pose (camera2world matrix)
            select_coords: some random pixels of the scene
        Outputs:
            rgb maps of coarse net and fine net 
        """
        rays_o, rays_d = self._generate_rays(pose)

        if select_coords is not None:
            rays_o = rays_o[select_coords[..., 0], select_coords[..., 1]]
            rays_d = rays_d[select_coords[..., 0], select_coords[..., 1]]

        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

        view_dirs = (rays_d / torch.norm(rays_d, dim=-1, keepdim=True)).float()
        # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        if self.ray_chunk is None or self.ray_chunk <= 1:
            return self._cast_rays((rays_o, rays_d), view_dirs)

        coarse_result, fine_result = [], []

        for i in range(0, rays_o.shape[0], self.ray_chunk):
            j = i + self.ray_chunk

            c, f = self._cast_rays((rays_o[i:j], rays_d[i:j]), view_dirs[i:j])
            coarse_result.append(c)
            fine_result.append(f)

        return torch.cat(coarse_result, dim=0), torch.cat(fine_result, dim=0)

    def render_image(self, pose, render_batch_size=1024, use_tqdm=True) -> np.ndarray:
        """
        render a scene (image)
        ======================
        * work in eval state (just rendering not for training)
        * NumPy.NDArray => torch.Tensor(CPU) => torch.Tensor(GPU) => torch.Tensor(CPU) => NumPy.NDArray
        Inputs:
            pose                : NumPy.NDArray         camera pose (camera2world matrix)
            render_batch_size   : int                   batch size of rendering default is 1024 
            use_tqdm            : bool                  whether or not use `tqdm` module
        Outputs:
            img: NumPy.NDArray
        """
        self.train(False)
        img_block_list = []
        pose = torch.tensor(pose, device=self.device)
        if use_tqdm:
            for epoch in tqdm.trange(0, self.height * self.width, render_batch_size):
                coords = get_screen_batch(
                    self.height, self.width, render_batch_size, epoch, device=self.device
                )
                _, image_fine = self.render(pose, select_coords=coords)
                img_block_list.append(image_fine.detach().cpu())
        else:
            for epoch in range(0, self.height * self.width, render_batch_size):
                coords = get_screen_batch(
                    self.height, self.width, render_batch_size, epoch, device=self.device
                )
                _, image_fine = self.render(pose, select_coords=coords)
                img_block_list.append(image_fine.detach().cpu())
        img = np.concatenate(img_block_list, axis=0)[:self.height * self.width].reshape(self.height, self.width, 3)
        return img
