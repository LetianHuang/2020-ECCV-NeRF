# Neural Radiance Fields (NeRF)

## 论文原文

[2020 ECCV *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*](https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/paper/NeRF.pdf) 

## 项目简介

* 该项目包含2020 ECCV NeRF论文原文及其复现代码，为学习论文以及官方实现后所做，代码中包含相关注释以及个人对论文的理解，仅供参考！
* 超参数采用的官方的超参数，显卡使用论文中**5.3**提到的NVIDIA V100 GPU，数据集使用官方 $lego$ 数据集，且对数据集合 $800\times800\times4$ 的图像做了处理，实际用于训练的是 $400\times400\times3$ 的图像，模型总共训练了 $200000$ 个epoch，总共耗时大约 $8\sim9$ 小时。
* 根据训练 $200000$ 个epoch得到的模型做体渲染，大约 $7\sim8$ 秒渲染一帧，显然该论文中的模型暂不能用于实时渲染。
* 复现了论文中**5.1**提到的Positional Encoding用于拟合了训练集图像的高频部分，因此最终做体渲染得到的图像高频部分并未丢失。
* 辐射场输入中加入了视角方向参数即view directions，对于渲染结果和视角方向强相关的材质（BRDF并非均匀分布，如Specular材质）也能得到很好的渲染结果。

## 环境与配置

* OS操作系统：Linux Ubuntu 18.04.4 LTS (GNU/Linux 4.15.0-180-generic x86_64)
* GPU显卡资源：Tesla V100-SXM2-32GB
* CUDA版本：11.4
* Python版本：3.8.12
* torch版本：1.9.0+cu102
* torchvision版本：0.10.0+cu102
* cudnn版本：8.2.4

其他Python模块依赖
```python
matplotlib
tqdm
opencv-python
```

## 3D模型可视化

通过 $360^{\circ}$ 全方位体渲染展示训练好的辐射场模型（一种隐式的几何表示）
<img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/render.gif"></img>

## 数据集介绍

数据集为官方提供数据集 $lego$ ，在该项目的`./data/nerf_synthetic/lego/*`目录下；该目录下包含三个文件夹和三个json文件，分别表示训练集`train`、验证集`val`、测试集`test`；json文件中`camera_angle_x`参数表示的是$fov_x$，`file_path`参数表示对应图像所在目录，`transform_matrix`表示相机坐标与世界坐标的坐标变换矩阵，在生成光线做Camera Ray Casting时需要用到。

为了提高模型训练速率以及防止显存不足，对原始数据集做了一定的修改，即将 $800\times800\times4$ 的图像resize成了 $400\times400\times3$ 的图像。

源码详情见`data_loader.py`模块

## 模型架构

<font size=2> 该模型不仅仅是神经网络用于训练的模型，同时也是作为体渲染管线中的几何模型，只不过这种几何模型是一种隐式表示，一种函数映射，而不是传统的多边形或者点云；而这个隐式表示的获得方式十分特殊，可以通过深度学习的技巧拟合这个函数映射 </font>

主要包含两个网络，一个是Coarse Net，另一个是用于层级采样的Fine Net（见论文**5.2**），这两种网络架构基本类似，只是用途不一样，Coarse Net主要生成用于层级采样的概率密度函数以及作为损失函数的一部分，Fine Net则主要是生成最终结果，当然也作为损失函数一部分；均是大致包括 $8\sim10$ 个全连接层（含激活层）的MLP。（详情见`nerf.py`模块的`_NeRF`类）

输入部分的pos_encoder和view_encoder这两个_Encoder对象则主要用于将输入映射到高维空间用于拟合高频信息，对应论文中的**5.1**的Positional Encoding。（详情见`nerf.py`模块的`_Encoder`类，该项目实现将Positional Encoding作为神经网络模型的一部分了，也是`torch.nn.Module`的子类）

$$
F_\Theta(p)=(p,\sin(2^0\pi{p}),\cos(2^0\pi{p}),\cdots,\sin(2^{L-1}\pi{p}),\cos(2^{L-1}\pi{p}))
$$ </br>

总的数据传递过程如下（源码详情见`nerf.py`模块的`NeRF`类）。其中输入 $p(x,y,z,d_x,d_y,d_z)$ 表示世界坐标系下的空间坐标 $(x,y,z)$ 以及视角方向 $(d_x,d_y,d_z)=(\theta,\phi)$ ，输出为该点体素 $(r,g,b,density)$ ，**注意输出的 $(r,g,b)$ 并不是最终该像素的颜色，最终生成图像的像素颜色需要通过体渲染器计算体渲染方程得到**

$$
p(x,y,z,d_x,d_y,d_z)\stackrel{Encoder}{\Longrightarrow}(p,\sin(2^0\pi{p}),\cos(2^0\pi{p}),\cdots,\sin(2^{L-1}\pi{p}),\cos(2^{L-1}\pi{p}))\stackrel{Coarse\,or\,{Fine}}{\Longrightarrow}{(r,g,b,\sigma)}
$$

TensorBoard可视化如下（用Coarse Net作为例子）
<img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/nerf_model.png"> </img>

## 体渲染器

体渲染方程如下：

$$
C(\boldsymbol{r})=\begin{equation*}
\int_{t_n}^{t_f} 
\exp(-\begin{equation*}
\int_{t_n}^{t} 
\sigma(\boldsymbol{r}(s)) 
\,\mathrm{d}s
\end{equation*}
)
\sigma(\boldsymbol{r}(t))
\boldsymbol{c}(\boldsymbol{r}(t),\boldsymbol{d}) 
\,\mathrm{d}z
\end{equation*}
$$


实际求解方程采用的是采样法，算法如下：

1. 生成光线。在 $[0,W-1]\times[0,H-1]$ 的屏幕空间中生成光线，并通过 $fov_x$ 和 $W$ 参数将屏幕坐标转换为相机坐标，再通过transform_matrix矩阵将相机坐标转为世界坐标，得到世界坐标系下的光束rays（对应项目中的`render.py`模块下`VolumeRenderer`类的`_generate_rays`方法）
2. 在光线上均匀采样。在步骤1中生成的rays上均匀采样，获得 $N_s$ 个样本点（对应论文中的**4**，项目中的`render.py`模块下`VolumeRenderer`类的`_sample`方法）
3. 采样点作为几何模型（NeRF coarse net）的输入获取体素信息。将步骤2中得到的采样点（包含视角方向）输入到NeRF几何模型coarse net中得到该采样点的体素（对应项目中`render.py`模块下`VolumeRenderer`类的`_voxel_sample5d`方法，该方法调用了`nerf.py`模块的`NeRF`模型的coarse_model）
4. 求解体渲染方程。根据步骤3得到的体素，计算点积求解渲染方程，得到图像的像素值以及density的PDF（对应论文中的**4**，项目中`render.py`模块下`VolumeRenderer`类的`_parse_voxels`方法）
5. 在光线上重要性采样。根据4得到的PDF在rays上重要性采样，获得 $N_f$ 个样本点（对应论文中的**5.2**，项目中的`render.py`模块下`VolumeRenderer`类的`_hierarchical_sample`方法）
6. 结合步骤2、5得到的采样点作为几何模型（NeRF fine net）的输入获取体素信息。（对应项目中`render.py`模块下`VolumeRenderer`类的`_voxel_sample5d`方法，该方法调用了`nerf.py`模块的`NeRF`模型的fine_model）
7. 再此求解体渲染方程。根据步骤6得到的体素，计算点积求解渲染方程，得到图像的像素值，作为实际渲染的图像（对应论文中的**5.2**，项目中`render.py`模块下`VolumeRenderer`类的`_parse_voxels`方法）

更多细节见`render.py`模块

## 模型训练

总共训练了 $200000$ 个epoch，优化器采用的Adam优化器，学习率采用的官方的 $0.0005$ ，且会因为迭代次数增加减小学习率，损失函数即为均方损失函数。

$$
L=\sum\limits_{\boldsymbol{r}\in{R}}[\Vert{\hat{C}_c(\boldsymbol{r})-C(\boldsymbol{r})}\Vert_2^2+\Vert{\hat{C}_f(\boldsymbol{r})-C(\boldsymbol{r})}\Vert_2^2]
$$

训练总时长约 $8\sim9$ 小时。

损失函数变化图像（可视化训练过程中生成的日志文件`./out/logs.txt`）
<font size=2>最初的若干个epoch因为刚开始训练且选取图像中心部分进行拟合，因此损失函数下降的十分迅速；之后之所以训练过程中会产生损失函数波动情况是因为每次迭代是随机选一张图像，并从该图像中随机选batch size个像素点做的训练；随着迭代次数增加波动的振幅越来越小，损失函数趋于收敛</font>
<img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/losses.gif"></img>

图像渲染变化（可视化训练过程中生成的日志文件`./out/imgs/*`）
<img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/train_img.gif"></img>

显卡利用率
<font size=2>由于代码实现最开始就将很多数据迁移到了GPU且很多运算都是矩阵运算，利于GPU并行，因此显卡利用率还不错；不过带来的问题是显存要求较高</font>
<img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/gpu.png"> </img>

源码详情见`train_nerf.py`模块，该模块可以通过`python train_nerf.py`指令运行

## 模型测试

验证集、测试集部分图像与训练后的模型渲染图像对比展示
<table>
<tr>
    <th>Ground Truth</th> <th>Predict</th>
</tr>
<tr>
<td width=50%><img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/data/nerf_synthetic/lego/test/r_0.png"></td><td width=50%><img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/test_0.png"> </td>
</tr>
<tr>
<td width=50%><img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/data/nerf_synthetic/lego/val/r_3.png"></td><td width=50%><img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/render_val_r_3.png"> </td>
</tr>
<tr>
<td width=50%><img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/data/nerf_synthetic/lego/val/r_9.png"></td><td width=50%><img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/render_val_r_9.png"> </td>
</tr>
<tr>
<td width=50%><img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/data/nerf_synthetic/lego/val/r_0.png"></td><td width=50%><img src="https://github.com/mofashaoye/2020-ECCV-NeRF/blob/main/out/other_imgs/render_val_r_0.png"> </td>
</tr>
</table>

测试集上模型的平均损失LOSS和平均峰值信噪比PSNR如下：

|  Data Set   | Number Of Images  |  LOSS   | PSNR  |
|  :----:  | :----:  |  :----:  | :----:  |
| Test Set | $200$ | $7.576\times{10^{-4}}$  | $31.43$ |
