# SRGAN - Pytorch and Keras 

同样是 `Super Resolution` 领域的一个经典文章，有了 `SRCNN` 的一个基础, 以及我们上次复现了 `VDSR` 这次的论文复现我们选择复现 `SRGAN`  
首先我们尝试下先使用比较简单且快捷的 `keras` 实现的方式，然后最后我们再把它搭建成我们比较常用的 `torch` 的方式。  
<br> 比较完美的就是我也通过这一次论文复现真正的接触到了更加前沿的深度学习神经网络架构


## SRGAN 论文重点
在实现 `SRGAN` 论文之前我们实现了我们的传统的 `VDSR` 以及更加前面我们实现过的 `SRCNN` 这两个都是比较入门的 `Super Resolution` 方向的论文。  

#### VDSR
`VDSR`  [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf)  
以及 `VDSR` 论文的重点为：
* 模型具有非常深的层
* 使用了残差学习和自适应梯度裁剪来加速模型的训练
* 将单一倍数的超分模型扩充到多个倍数  
  
与`SRCNN`一样，都是先将低分辨率输入双三次插值到高分辨率，再来进行模型的预测。
这里包括两个部分，VGG-like的深层网络模型，每一层卷积中均使用带padding的3x3卷积层，
并且随后都会添加一个ReLU来增强模型的非线性，这里与`SRCNN`、`SRCNN-Ex`都有着较大的改变。
然后最后使用残差学习来将模型预测到的结果element-wise的形式相加，来得到最终的结果。  

#### SRCNN
`SRCNN` [Image super-resolution using deep convolutional networks](https://ieeexplore.ieee.org/document/7115171/;jsessionid=sqmfzoJEerWjinbTLnm8TVyWaFJSTAXKVbNp_abvj-XrT4nB9Sf6!84601464)
`SRCNN` 是 `end-to-end`（端到端）的超分算法，所以在实际应用中不需要任何人工干预或者多阶段的计算.


## Datasets

之前我们的 `SRCNN` 使用的是 `cifar-10` , 那个数据集不是很好用，因为它的原始数据就是 `32 * 32` 大小的，
不太适合放大缩小。我们尝试试试其他的数据集看看效果会是怎样。  
然后我们选用一个比较大的数据集也就是我们的 `BSD500` 的数据集。
数据集选用的是 `BSD500` 但是我们截取成为 256*256



## Prerequisites
 * Tensorflow  > 2.0  
 * Keras > 2.0
也是想着通过这个项目去尝试使用 `Tensorflow 2.0` 以及架构在它上面的 `Keras` 然后复现一下超分辨比较经典的论文 `SRGAN` . 


## Usage
For training, `python trains.py`
<br>
For testing, `python trains.py` 暂时没有写对应的 `test.py`



## Problems
* 因为使用的是 `cifar10`的数据集，会出现的问题就是它的图像数据的大小是 32*32 的，
  所以没有做一些放大缩小的操作获取对应的 High Resolution Image -> Low Resolution Image 的操作。
  
* 做的 `Keras` 和 `Tensorflow` 的训练并没有像 `Pytorch` 一样使用 `tqdm` 模块去做一些操作。  
  
* `pytorch` 要非常注意一点就是它的 Tensor 和 `tensorflow` 或者 `keras` 不一样，可能 `tensorflow` `keras` 是以
  `Size * H * W * C` 而 `pytorch` 是以 `Size * C * H * W` 的方式去计算的，所以使用的数据需要通过 `torch.permute` 的 方式修改数据格式。  
    
  

## Result
  
以下是 `VDSR` 的 `result table` :  

| Dataset | Epochs | Module | Method     | psnr   |
|---------|------- |------  |------      | ------ |
| cifar10 | 500    | SRCNN  | tensorflow | 56.0   |
| cifar10 | 500    | SRCNN  | keras      | 25.9   |
| cifar10 | 500    | SRCNN  | pytorch    | 26.49  |

  
`tensorflow` 可能是因为数据集的问题导致 `psnr` 的计算会出现一些小的问题

因为数据集的使用问题，所以模型的训练是没有意义的。  
出于对`cifar`数据集的一个不了解，它是 `32*32`的，但是我将它 bicubic 放大成了 `128*128` 作为 ground true。  
然后训练数据 从 `32*32` resize 到 `32*32` 用邻近插值，然后又 bicubic 放大成 `128*128` 作为训练数据，这个是无效的训练。
所以训练效果直接爆炸。  
后续也不因数据集问题做更多的尝试和改进。整个内容当作对 `tensorflow > 2.0`  的一个入门尝试。

## References

👇是对`markdown`使用的一些了解

This repository is implementation of the ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/abs/1501.00092).

<center><img src=""></center>

## Train

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
| 91-image | 4 | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0) |



* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 
  * - I referred to this repository which is same implementation using Matlab code and Caffe model.
<br>

* [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 

