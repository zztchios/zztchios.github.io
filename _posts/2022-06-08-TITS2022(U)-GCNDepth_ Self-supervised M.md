Comment: ICCV 2021 camera-ready version. Code is available at https://github.com/ArminMasoumian/GCNDepth.git

**GCNDepth: Self-supervised Monocular Depth Estimation based on Graph Convolutional Network**

**GCNDepth: 基于图卷积网络的自监督单目深度估计**

> Author and Department
> 
> Armin et. al. 西班牙的大学； TITS在投，2022.

# Summary

> 写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。

这篇文章主要解决单目深度估计中几何外观和分布的保存问题，提出采用GCN提取non-Euclidean数据，网络包含两个并行的自编码器网络，也就是根据U-Net修改而来，第一个自编码器是生成深度图，第二个编码器是预测两个连续帧的自运动向量。损失函数作者采用的是重构损失(一阶范数)，重投影损失(SSIM loss)，平滑损失(一阶导数和二阶导数)。

# Abstract

> 分为三个部分：1.background 2.motivation 3.method 4. conclusion

- **Background**: 传统CNN不支持拓扑结构，只能在regular 图像领域操作，对于非欧几里得空间域并不能很好的应用。
    
- **Motivation**: 为了有效的捕获非欧几里得空间中的拓扑结构，作者提出一个基于GCN的网络模型提取非欧几里得数据。
    
- **Method**: 作者提出的方法分为两个部分，第一部分是自编码器是生成深度图(DepthNet)，第二个编码器是预测两个连续帧的自运动向量(PoseNet)。其中，**DepthNet**采用的是一个U-Net基础网络，在解码器中增加了图卷积上采样，帮助模型从低维特征中学习深度信息映射。**PoseNet**是一个回归网络，也是采用编解码结构，编码器输入为原始图片$I_s$和目标图片$I_t$，输出是512的特征映射；解码器是4层的上采样层。
损失函数作者采用的是重构损失(一阶范数)，重投影损失(SSIM loss)，平滑损失(一阶导数和二阶导数曲率)。
    
- **Conclusion**： 作者提出的GCNDepth方法能够从低维度中映射深度信息，也可以通过表示场景像素之间的关系来表示场景的拓扑结构。同时采用多种loss来约束训练中的**目标图像与重建图像的误差**，**光度重投影误差**和**平滑误差**。



## Contribution

- 作者提出了GCN网络用于自监督深度预测，通过迭代传播邻居信息构建深度图，学习节点(像素)的表示，从而提高深度图的精度。
    
- 为了利用多尺度像素空间相关性，作者提出了不同邻域尺度的解码器网络。

- 作者提出了一种组合**目标图像与重建图像的误差**，**光度重投影误差**和**平滑误差**的损失函数。其中，重投影损失用于处理对象遮挡，重建损失为了减少重建图像与目标之间的损失。反之，平滑损失用于保留对象的边和边界，减少纹理区域对估计深度的影响。

    

# Method(s)

> 作者解决问题的方法/算法是什么？是否基于前人的方法？基于了哪些？


## 3.1 Problem Defination

GCNDepth是多任务问题，有两个网络：DepthNet（深度预测）和PoseNet（ego-motion预测）。
(1) DepthNet将输入图像$I\in \mathbb{A}$生成深度图$D\in \mathbb{B}$，生成函数$\Psi:\mathbb{A}\rightarrow \mathbb{B}$

$$
D(p)=\Psi_D(I_s(p))\tag{1}
$$

(2) PoseNet考虑连续帧的视角转变，两帧分别是$I_s(p)$和$I_t(p)$，预测ego-motion矩阵：$E_{I_s\rightarrow I_t}=[r^T,t^T]$，其中$r=[\Delta \theta,\Delta \phi,\Delta \psi]^T$表示旋转矩阵，$t=[\Delta x,\Delta y,\Delta z]^T$代表平移矩阵，映射过程可以近似表示为：

$$
E_{I_s\rightarrow I_t}=\Psi_E(I_s(p), I_t(p))\tag{2}
$$

综合考虑depth和ego-motion，GCNDepth：
$$
\Psi({I_s(p) I_t(p)})=(D(p),E_{I_s\rightarrow I_t},I_{rec}(p))\tag{3}
$$

## 3.2 Graph Convolutional Network

CNN在场景物体细节方面产生重大损失。因此，需要采用GCN的拓扑结构增加隐含层的特征表示。帮助模型学习如何从低维映射到深度信息。

<div align="center"><img src="../../_resources/548172def5dd3abb3964d22726f6d1c6.png" width="600" height="XXX" class="jop-noMdConv"></div>



## 3.3 Self-supervised CNN-GCN Autoencoder

作者采用UNet作为auto-encoder的baseline，包含两个连续的子网：
>1. Encoder:将输入映射到高维特征表示；
>2. Decoder:将特征表示映射到重构的深度图中。


**DepthNet Encoder**

CNN能够利用图像数据的局部连通性和全局结构，在训练阶段通过提取有意义特征。作者说CNNs更加适合从全局场景中提取全局视觉特征(不应该是transformer么？)。编码器有5层深度层，最后四层是Resnet-50，第一层是一个1x1的卷积(CNN+BN+max-pooling)。如表1所示：

<div align="center"><img src="../../_resources/c2e81379423190219060e2662d6deccc.png" width="400" height="XXX" class="jop-noMdConv"></div>

**DepthNet Decoder**

解码器部分作者采用几何深度网络，提取物体的局部特征，并且通过生成深度拓扑图保持节点间的深度图，下图说，初始化图的邻接矩阵是基于encoder最后一层生成特征的节点数量(邻接矩阵，需要进一步看代码)。

<div align="center"><img src="../../_resources/589b84478fd62da968d9c393d38c11f6.png" width="600" height="XXX" class="jop-noMdConv"></div>

作者采用4层GCN重构深度图，每层都是upconvolution层，并将其与来自编码器网络相应层的相应特征映射连接，并使用其一层GCN进行上采样粗深度预测。这种方法保留从较粗特征图中传递的高级信息(GCN提取的)以及较低层特征图中提取的细粒度局部信息(inverse convolution)。每一次concatenate都会提高分辨率2倍。最后输出图像的分辨率是输入的一半。
具体构造如表Ⅱ所示：

<div align="center"><img src="../../_resources/e58fe48e238c1cfbb3c7e6b4ef10ecd8.png" width="400" height="XXX" class="jop-noMdConv"></div>

**PoseNet Estimator**
PoseNet是一个回归网络，输入是相邻图像($I_s$和$I_t$)的concatenate。encoder是5层cnn(Conv1x1,ResNet-18)。decoder如表Ⅲ所示：

<div align="center"><img src="../../_resources/5466fef4db2ae4ec840f99f41754ad9d.png" width="400" height="XXX" class="jop-noMdConv"></div>

## 3.4 Overall Pipelines


DepthNet输入是源图像，输出是depth image。 PoseNet的输出是source image$I_s$和target image $I_t$的相对位置。模型的整体结构如图3所示：

<div align="center"><img src="../../_resources/a2919dab26ac724df525c3144a1a7208.png" width="500" height="XXX" class="jop-noMdConv"></div>

## 3.5 Geometry Models and Losses

作者采用3种损失函数：分别是:$L_{Pl}$(L1-norm+SSIM loss)，$L_{Rec}$(reconstruction loss, L1-norm)，$L_{Smooth}$(一阶导数和二阶导数，也就是边缘loss和曲率loss)
>1 reconstruction loss $L_{Rec}$ (一阶范数)
	起源于monodepth2，

$$
L_{Rec}=\sum_p|I_{Rec}(p)-I_{t}(p)|\tag{6}
$$
>2 combines the L1-norm and SSIM losses(为了处理遮挡帧)
$$
L_{Pl}=0.15\times \sum_p|I_{Rec}(p)-I_{t}(p)|+0.85\times \frac{1-SSIM(I_{rec},I_t)}{2}\tag{7}
$$

>3 Smooth loss
> 拓展知识：Lambertian shading function,Lambertian表面是指在一个固定的照明分布下从所有的视场方向上观测都具有相同亮度的表面，Lambertian表面不吸收任何入射光．Lambertian反射也叫散光反射，不管照明分布如何，Lambertian表面在所有的表面方向上接收并发散所有的入射照明，结果是每一个方向上都能看到相同数量的能量。
> 作者认为图像强度函数遵循Lambertian表面方程，因此为了解决遮挡、过度平滑和纹理区域而通常存在的深度不连续性，需要生成的深度图的损失来约束物体边缘，降低纹理效果。一阶二阶导数可以突出物体几何特征以及同质区域（相同分布的区域）。

**边缘保存的深度图生成损失$L_{Dis}$**，目的是给低纹理区域赋予较大的权重：

$$
L_{Dis}=\sum_p e^{-\lambda\nabla^1 I_s(p)}|\nabla^1D(p)|\tag{8}
$$
其中$D$表示每个像素点p的预测深度，$\nabla ^1$表示像素点p的一阶导数，$||\nabla^1||=\sqrt{I^2_x+I_y^2}$。$\lambda$权重因子，本文采用0.5。

**曲率 loss$L_{Cvt}$**，参考FeatDepth网络。目的是给纹理区域低的权重：
$$
L_{Cvt}=\sum_p e^{-\lambda\nabla^2 I_s(p)}|\nabla^2D(p)|\tag{9}
$$

两种方法进行融合，为smoothness loss的两部分：
$$
L_{Smooth}=\alpha L_{Dis}+\beta L_{Cvt}\tag{10}
$$
其中：$\alpha$=$\beta$=1e-3

>4 最终的损失函数
$$
L_{Final}=L_{Pl}+L_{Rec}+L_{smooth}\tag{11}
$$

## Implementation Detials

好消息，作者用GTX1080Ti *1就可以搞定，而且20个epochs。
# Experiments

> 作者如何评估自己的方法？实验的setup是什么样的？感兴趣实验数据和结果有哪些？有没有问题或者可以借鉴的地方？

<div align="center"><img src="../../_resources/3e5fb5a5a1bff10aefb8d24922e8f3f3.png" width="400" height="XXX" class="jop-noMdConv"></div>

<div align="center"><img src="../../_resources/d385271be5af9c9428037423c5d8594e.png" width="400" height="XXX" class="jop-noMdConv"></div>
<div align="center"><img src="../../_resources/b14cf78781994d82693eb7185aa94524.png" width="400" height="XXX" class="jop-noMdConv"></div>
<div align="center"><img src="../../_resources/88bbb57cd4cbce26fb61fb805a941f4f.png" width="400" height="XXX" class="jop-noMdConv"></div>
<div align="center"><img src="../../_resources/3458c8c2a6b83a10e1621e5740d1a12d.png" width="400" height="XXX" class="jop-noMdConv"></div>
<div align="center"><img src="../../_resources/7388988795f9ed8b285ec19ed9da453e.png" width="400" height="XXX" class="jop-noMdConv"></div>


# Conclusion

> 作者给出了哪些结论？哪些是strong conclusions, 哪些又是weak的conclusions（即作者并没有通过实验提供evidence，只在discussion中提到；或实验的数据并没有给出充分的evidence）?


# Notes(optional)

> 不在以上列表中，但需要特别记录的笔记。

# References(optional)

> 列出相关性高的文献，以便之后可以继续track下去。

# 单词句型句式

## 单词

> 记录单词，便于记忆

## 句型

> 常用句型，便于将来写作

## 长句

> 便于理解