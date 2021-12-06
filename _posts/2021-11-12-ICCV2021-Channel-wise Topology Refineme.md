---

layout:     post
title:      Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition
subtitle:   GCNs
date:       2021-11-12
author:     zztchios
header-img: img/post-bg-take.png
catalog: true
tags:
    - GCNs系列
---
<!--<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>-->

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


**基于骨架的动作识别的通道式拓扑增强图卷积**

>Author and Department
> Ruwen Bai et. al. 中科大自动化研究所&中科大人工智能学院&南昌航空大学； 发表在ICCV上，2021.
> **重点复现**



# Summary

> 写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。




# Abstract
>分为三个部分：1.background 2.motivation 3.method 4. conclusion

- **Background**:“In GCNs, graph topology dominates feature aggregation and therefore is the key to extracting representative features.”  图卷积中，图拓扑主导特征聚合，因此图卷积是提取特征表示的关键。

- **Motivation**: The topologies of graph convolutions for channel-wise have strict constraints (channel-wise图卷积的拓扑是严格约束，it is strict constraints,consequently, 需要给不同通道分配不同的图结构才算是合理，而且这些图结构必须共享才能体现出不同有效性)

- **Method**：the author propose a novel Channel-wise Topology Refinement Graph Convolution(CTR-GC)

   - CTR-GC dynamically learn different topologies and effictively aggregate joint features in different channels(CTR-GC动态学习不同的拓扑并有效地将节点特征聚合到不同的通道上)

   - “through learning a shared topology as a generic prior for all channels and refining it with channel-specific correlations for each channel.” (CTR-GC通过学习共享拓扑作为所有的通道的通用先验，并通过每个通道的特定关联来完善它，来建立通道的拓扑结构。)

   - The refinement method introduces few extra parameters and significantly reduces the difficulty of modeling channel-wise topologies(这个强化方法引入很少的额外参数，并显著的降低了建模通道级拓扑的难度)

   - “Furthermore, via reformulating graph convolutions into a unified form, we find that CTR-GC relaxes strict constraints of graph convolutions, leading to stronger representation capability.” 此外通过将图卷积重新表述为一个统一格式，作者发现CTR-GC减轻了图卷积严格的约束，拥有更强的表达能力

- **Conclusion**：notably outperforms state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120, and NW-UCLA datasets.


# Research Objective(s)/Motivation

> 作者的研究目标是什么？

“Instead of learning topologies of different channels independently, CTR-GC learns channel-wise topologies in a refinement way.”  CTR-GC不是独立的学习不同通道的拓扑，而是通过一种refinement方法学习通道间的拓扑。

“Specifically, CTR-GC learns a shared topology and channel-specific correlations simultaneously.” 具体来说，CTR-GC学习一种共享拓扑结构并同时进行通道特定关联性。

“The shared topology is a parameterized adjacency matrix that serves as topological priors for all channels and provides generic correlations between vertices.”  共享拓扑结构是一种参数化邻接矩阵，为所有通道提供拓扑先验，并且提供通用的顶点关联性。

“The channel-specific correlations are dynamically inferred for each sample and they capture subtle relationships between vertices within each channel.”  每个样本的特定通道相关性是动态推断出来的，它们捕捉到每个通道中顶点间微妙的关系。

通过特定通道的关联性完善共享拓扑结构，CTR-GC获取通道方面拓扑结构。

“Our refinement method avoids modeling the topology of each channel independently and introduces few extra parameters, which significantly reduces the difficulty of modeling channel-wise topologies. Moreover, through reformulating four categories of graph convolutions into a unified form, we verify the proposed CTR-GC essentially relaxes strict constraints of other categories of graph convolutions and improves the representation capability.” 

作者的refinement方法避免了在每个通道独立建模拓扑结构,并引入少量的额外参数，这样做显著减少了建模通道方向的拓扑结构的难度。此外，通过重新定义四类图卷积(是否共享channel，是否是动态地)为一个统一的形式，作者验证了提出的网络本质上减轻了其他类别图卷积的严格约束，并且提高了表达能力。

## Contribution

- 作者提出了一个channel-wise拓扑结构refinement graph convolution，采用refinement 方法动态建模通道方面的拓扑，从而实现灵活且有效的关系建模。

- 作者在数学上统一了基于骨架动作识别的现有图卷积的格式，发现CTR-GC减轻了其他图卷积的约束，提供了更强的图建模能力。

- 大量的实验结果强调了channel-wise拓扑和refinement方法的优势，并且获得了SOTA。


# Background / Problem Statement(Introduction)

>研究的背景以及问题陈述：作者需要解决的问题是什么？

<div align=center>
<img src="https://github.com/zztchios/zztchios.github.io/raw/master/img/daf50e37cc1474b8976300bca642314d.png" width="600" height="XXX" />
</div>

>Cheng et al. [3] set individual parameterized topologies for channel groups. However, the topologies of different groups are learned independently and the model becomes too heavy when setting channel-wise parameterized topologies, which increases the difficulty of optimization and hinders effective modeling of channel-wise topologies. Moreover, parameterized topologies remain the same for all samples, which is unable to model sample-dependent correlations.
作者主要是参考文章[Decoupling GCN with DropGraph Module for Skeleton-Based Action Recognition](https://github.com/zztchios/zztchios.github.io/raw/master/img/undefined), 这篇文章主要是为通道组设置独立的参数化拓扑结构，但是不同组的拓扑结构独立学习，当设置通道方面的参数拓扑的时候，模型变得too heavy,这样会增加优化难度，阻碍了通道方面有效的拓扑建模。Moreover, 参数拓扑对所有样本都是一样的，这样就不能建模依赖样本关联性。

The author describe many existing methods such as  pseudo-image, GCNs, 2s-AGCN etc.  However, [3] is the motivation and reference for publishing the paper:

作者将图卷积方法分为两类拓扑：1.根据推理中，是否动态调整拓扑，可以分为静态方法和动态方法；2.根据拓扑是否再不同通道之间被共享，可以分为拓扑共享性方法和无共享拓扑方法。

- **静态/动态方法**。**静态方法**指的是在推理过程中，图卷积的拓扑是否被固定。如：ST-GCN，multi-scale GCNs，都是按照人体结构固定拓扑结构。**动态方法**指的是the topologies of GCNs are dynamically inferred during inference。如2s-AGCNs, AS-GCN, SGN(通过自注意力机制)。

- **Topology-shared / Topology-non-shared Methods**。对于**Topology-shared方法**，the static or dynamic topologies are shared in all channels. These methods force GCNs to aggregate features in different channels with the same topology, 限制了模型性能的上限。 **Topology-non-shared methods**在不同通道或者通道组中采用不同的拓扑结构，这样做自然克服了拓扑共享方法的限制。Cheng等人[3]提出了一个DCGCN，它为不同的信道组设置了单独的参数化拓扑结构。然而，DC-GCN在设置通道方面的拓扑结构时，面临着参数过多造成的优化困难。作者是第一个而建立动态通道方面非共享拓扑的模型(dynamically topology-non-shared model)。
# Method(s)

>作者解决问题的方法/算法是什么？是否基于前人的方法？基于了哪些？


- Transform input features into high-level features (将输入特征转化为高阶特征)

- dynamically infer channel-wise topologies to capture pairwise correlations between input sample’s joints under different types of motion features, and aggregate features in each channel with corresponding topology to get the final output(动态推理通道拓扑结构去捕捉不同类型运动输入样本中关节点之间成对的关联性，采用相应拓扑聚合每个通道的特征以获得最终的结果)
   - Feature transformation which is done by transformation function $$\mathcal{T} (\cdot)$$;（$$\mathcal{T} (\cdot)$$是特征转换函数）

   - Channel-wise topology modeling which consists of correlation modeling function$$\mathcal{M}(\cdot)$$ and refinement function $$\mathcal{R}(\cdot)$$;（通道方面拓扑建模，包含了相关建模方程$$\mathcal{M}(\cdot)$$和refinement function $$\mathcal{R}(\cdot)$$;）

   - Channel-wise aggregation which is completed by aggregation function$$\mathcal{A}(\cdot)$$(通道方面的聚合，由聚合函数$$\mathcal{A}(\cdot)$$完成）
   $$\mathbf{Z}=\mathcal{A}(\mathcal{T(\mathbf{X}),\mathcal{R}(\mathcal{M}(\mathbf{X}),\mathbf{A})})\tag{2}$$

<div align=center>
<img src="https://github.com/zztchios/zztchios.github.io/raw/master/img/b951d2de473b508b47f36db8c7a07a17.png" width="800" height="XXX" />
</div>

**Feature Transformation**

feature transformation的目的是将输入特征转化为high level 表示。其中$$\mathcal{T(\mathbf{X})}$$为线性转换函数，它负责拓扑共享图卷积：

$$\widetilde{X}=\mathcal{T(\mathbf{X})}=\mathbf{X}\mathbf{W}\tag3$$

其中$$\widetilde{X}\in \mathbb{R}^{N \times C^\prime}$$是转换特征，$$\mathbf{W}\in \mathbb{R}^{C\times C^\prime}$$是权重矩阵。

<div align=center>
<img src="https://github.com/zztchios/zztchios.github.io/raw/master/img/19b55dd6f3b9c3406e73887b34eaa566.png" width="500" height="XXX" />
</div>

**Channel-wise Topology Modeling**

邻接矩阵为所有通道共享拓扑结构，通过反向传播学习参数。此外，作者学习特定通道的关系$\mathbf{Q}\in \mathbb{R}^{N \times N \times C^\prime}$去建模$C^\prime$通道顶点见的特定关系。之后通过优化共享拓扑$$\mathbf{A}$$和$$\mathbf{Q}$$去获取channel-wise拓扑$$R\in \mathbb{R}^{N \times N \times C^\prime}$$。

- 通过$$\mathcal{M}(\cdot)$$建模channel-wise顶点相关性。

- 再输入特征feed in $$\mathcal{M}(\cdot)$$之前，采用线性转换$$\phi$$和$$\varphi$$减少特征维度

- design two functions 寻找关联性。

   - $$\mathcal{M}_1(\varphi(x_i),\phi(x_j))=\sigma(\varphi(x_i)-\phi(x_j))\tag4$$ 
   $$\mathcal{M}_1(\cdot)$$本质上是计算沿着通道$$\varphi(x_i)$$和$$\phi(x_j)$$距离，并且利用这 些距离非线性转换作为特定通道拓扑结构节点之间关系
   - $$\mathcal{M}_2(\varphi(x_i),\phi(x_j))=MLP(\varphi(x_i)\|\|\phi(x_j))\tag5$$
   其中||是连接操作。

<div align=center>
<img src="https://github.com/zztchios/zztchios.github.io/raw/master/img/2e93803a0967d4b9fe3da42eeb6deec5.png" width="500" height="XXX" />
</div>

利用线性转换函数$$xi$$获取特定通道关系矩阵$$\mathbf{Q}\in \mathbb{R}^{N\times N \times C^\prime}$$，去提升通道维度：

$$\mathbf{q}_{ij}=\xi(\mathcal{M}(\varphi(x_i),\phi(x_j))),i,j\in {1,2,…,N}\tag6$$

$$\mathbf{q}_{ij}\in \mathbb{R}^{C^\prime}$$是$$\mathbf{Q}$$中的向量，反映了特定通道中$$v_i$$和$$v_j$$的拓扑关系。

通道拓扑$$\mathcal{R}\in\mathbb{R}^{N\times N\times C^\prime}$$由共享拓扑$$\mathbf{A}$$和特定通道关联$$\mathbf{Q}$$优化得来：

$$\mathbf{R}=\mathcal{R}(\mathbf{Q},\mathbf{A})=\mathbf{A}+\alpha\cdot\mathbf{Q}\tag7$$

**Channel-wise Aggregation**

经过上两步已经确定channel-wise拓扑结构和高阶特征X，CTR-GC采用channel-wise方式聚合特征。并将$$\mathbb{R}_c$$和X按照通道进行分层提取特征。每个channel-graph反映了节点之间的关系。

Consequently,得到feature aggregation ouput $$\mathbf{Z}$$

$$\mathbf{Z}=\mathcal{A}(\widetilde{X},\mathbf{R})=[\mathbf{R}_{1}\widetilde{x}_{:,1}||\mathbf{R}_{2}\widetilde{x}_{:,2}\cdots||\mathbf{R}_{C^\prime}\widetilde{x}_{:,C^\prime}]\tag8$$

## 3.3 Analysis of Graph Convolutions

作者分析了一下不同图卷积的表达能力，通过重新将其组合成一个统一的形式，然后将他们与动态卷积相比较。

动态卷积表达式：

$$\mathbf{z_i^k}=\sum_{p_j\in \mathcal{N}(p_i)}\mathbf{x_j^kW_j^k}\tag9$$

其中$$\mathbf{k}$$代表输入样本序列，$$\mathbf{x_j^k}$$和$$\mathbf{z_i^k}$$是$$p_i$$输入特征，$$p_i$$的输出特征为k-th样本。

由于不规则图结构，所以很难用传统卷积方法建立。因此，图卷积将权重退化为邻域权重，然后共享邻域权重。

然而共享权重限制了表达能力。作者为了分析不同图卷积和动态卷积的表达能力差距，将邻域权重和邻域共享权重整合为一个通用的权重矩阵$$\mathbf{E}_{ij}^k$$.


作者将所有的GCs用公式表达为$$\mathbf{z_i^k}=\sum_{v_j\in\mathcal{N}(v_i)}\mathbf{x_j^kE_{ij}^k}$$,其中$$\mathcal{E_{ij}^k}$$表示通用权重。作者将图卷积分为4类：
- **Static Topology-shared GCs**.就是指最原始的固定卷积
   $$\mathbf{z_i^k}=\sum_{v_j\in\mathcal{N}(v_i)}a_{ij}\mathbf{x_j^k}\mathbf{W}=\sum_{v_j\in\mathcal{N}(v_i)}\mathbf{x_j^k} a_{ij}\mathbf{W})\tag{10}$$
   其中$a_{ij}\mathcal{W}$是通用权重。静态拓扑共享GC的广义权重要受到如下约束：
   - Constraint 1: $$E^{k_1}_{ij}$$ and $$E^{k_2}_{ij}$$ are forced to be same.

   - Constraint 2: $$E^{k}_{ij_1}$$ and $$E^{k}_{ij_2}$$differ by a scaling factor.相差一个比例系数

	 $$k_{1}$$ and $$k_{2}$$ 是不同的样本指数，$$j_{1}$$ and $$j_{2}$$ 是不同的顶点序列。这些约束造成了static topology-shared GCs and dynamic convolutions表达能力的差距。
- **Dynamic topology-shared GCs**.其实就是2s-AGCN为代表的自注意力学习图拓扑结构

	相比于static共享拓扑GCs，动态的推理拓扑因此会有更好的泛化能力。数学表达式为：

	$$\mathbf{z_i^k}=\sum_{v_j\in\mathcal{N}(v_i)}a_{ij}^k\mathbf{x_j^k}\mathbf{W}=\sum_{v_j\in\mathcal{N}(v_i)}\mathbf{x_j^k} a_{ij}^k\mathbf{W})\tag{11}$$

	其中$$a_{ij}^k$$是节点$$i$$和$$j$$动态拓扑关联,取决于输入样本。可以看出，动态拓扑共享GC的广义权重仍然受制于约束条件2，但将约束条件1放宽为以下约束条件。

	- Constraint 3: $$E^{k_1}_{ij}$$ and $$E^{k_2}_{ij}$$differ by a scaling factor.相差一个比例系数
- **Static topology-non-shared GCs**.
	这种GCs利用不同的拓扑结构对不同的通道进行处理。

	$$\mathbf{z_i^k}=\sum_{v_j\in\mathcal{N}(v_i)}\mathbf{p_{ij}}\odot(\mathbf{x_j^k}\mathbf{W})\tag{12}$$

	$$=\sum_{v_j\in\mathcal{N}(v_i)}\mathbf{x_j^k}([p_{ij1}\mathbf{w_{:,1},\cdots,}p_{ijC^\prime}\mathbf{w_{:,C^\prime}}]) \tag{13}$$
	其中$$\odot$$表示元素级别的乘积，$$\mathbf{p_{ij}}\in\mathcal{R}^{C^\prime}$$是通道方面的拓扑关系，通过公式13可以发现，由于静态拓扑广义权重受限于约束条件1，但是放松了约束2在如下约束中：
	- Constraint 4: Different corresponding columns of $$E^{k}_{ij_1}$$ and $$E^{k}_{ij_2}$$ differ by different scaling factors.

- **Dynamic topology-non-shared GCs**
	静态拓扑-非共享GC和动态拓扑-非共享GC的唯一区别是，动态拓扑-非共享GC是动态推断非共享拓扑，因此动态拓扑-非共享GC可以表述为:

	$$\mathbf{z_i^k}=\sum_{v_j\in\mathcal{N}(v_i)}\mathbf{x_j^k}([r^k_{ij1}\mathbf{w_{:,1},\cdots,}r^k_{ijC^\prime}\mathbf{w_{:,C^\prime}}]) \tag{14}$$

	显然，动态拓扑-非共享图卷积的广义权重同时放松了约束条件1和2。具体来说，它将约束条件2放宽为约束条件4，并将约束条件1放宽为以下约束条件。
	- Constraint 5: Different corresponding columns of $$E^{k_1}_{ij}$$ and $$E^{k_2}_{ij}$$ differ by different scaling factors.
	

作者在表1中总结了不同类别的图卷和它们的约束。可以看出，动态拓扑-非共享GC的约束性最小。我们的CTRGC属于动态拓扑-非共享GC，方程8可以重新表述为方程14，表明理论上CTR-GC比以前的图卷积具有更强的表示能力。具体的重构方式见补充材料。

<div align=center>
<img src="https://github.com/zztchios/zztchios.github.io/raw/master/img/2aece8cce041d040e8171689c59f4aa6.png" width="600" height="XXX" />
</div>

## Model Architecture

作者将节点邻域设定为全部节点的连接，和2s-AGCN同样的思路。然后自适应所有的channel

**Spatial Modeling**

首先，包含三个CTR-GCs block去提取节点间的关系，

<div align=center>
<img src="https://github.com/zztchios/zztchios.github.io/raw/master/img/83a35926175c698e950e69bdce1f6d09.png" width="300" height="XXX" />
</div>

CTR-GC用来提取具有输入特征X的图的特征。然后将时间序列pool之后，用池化后的特征推断通道拓扑。Specifically,利用φ和ψ的下降率r提取紧凑表示。之后，时间池化被用作聚合时间特征。After that，CTR-GC建立成对的减法操作和激活操作。采用$$\xi$$提高激活的通道维度，以获得channel特定相关性，这些相关性用于完善共享拓扑$$A$$以获取通道方面的拓扑结构。Eventually, channel-wise 聚合在每个骨架图中应用去获取输出表达$$\mathbf{S}^\mathbf{o}$$


**Temporal Modeling** 这个和MS-G3D类似，做了一点点改进

multi-scale temporal modeling module. 作者采用更少的分支来应对太多分支减慢了推理速度。
<div align=center>
<img src="https://github.com/zztchios/zztchios.github.io/raw/master/img/6988afab7dbd1d7e81913ac4d6c31699.png" width="300" height="XXX" />
</div>

如上图， 这个模型包含了4个分支，每个分支包含$$1\times1$$卷积去减少通道维度。前三个分支包含两个不同膨胀的时间卷积和一个$$1\times1$$卷积后的Max-Pool。四分支结果最后进行concatenate。

# Experiments

>作者如何评估自己的方法？实验的setup是什么样的？感兴趣实验数据和结果有哪些？有没有问题或者可以借鉴的地方？



## Ablation Study

# Conclusion

>作者给出了哪些结论？哪些是strong conclusions, 哪些又是weak的conclusions（即作者并没有通过实验提供evidence，只在discussion中提到；或实验的数据并没有给出充分的evidence）?

# Notes(optional)

> 不在以上列表中，但需要特别记录的笔记。

# References(optional)

>列出相关性高的文献，以便之后可以继续track下去。


# 单词句型句式

## 单词
>记录单词，便于记忆

notably 值得注意的是

## 句型
>常用句型，便于将来写作


## 长句
>便于理解 



