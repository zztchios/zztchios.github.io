---

layout:     post
title:      Spatial Temporal Transformer Network for Skeleton-based Action Recognition
subtitle:   Transformer
date:       2021-11-10
author:     zztchios
header-img: img/post-bg-take.png
catalog: true
tags:
    - Attention系列
---
<!--<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>-->

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


>Author and Department
> Chiara et. al. 米兰理工大学,意大利； 发表在ICPR，2020.

论文有代码，但是复现不正确，之后跟踪继续。

# Summary

> 写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。

这篇文章是将Transformer应用于skeleton-based action Recognition.算是赶了一波热度，效果也还不错。作者首先将ST-GCN中的GCN和TCN分别用SSA和TSA进行替换，最后增强了时空的自注意力，从而增强了效果，调参过程中应用了DropAttention。**复现**还是有问题，将来希望能够解决。



# Abstract
>分为三个部分：1.background 2.motivation 3.method 4. conclusion

- **Background**: Skeleton data has been demonstrated to be robust to illumination changes(光线变化) etc. Nevertheless, an effective encoding of the latent information underlying the 3D skeleton is still an open problem(虽然骨架数据对于复杂环境鲁棒性较强，但是对于3D数据潜在信息的有效编码仍然是个问题)

- **Motivation**：I think rubbing Transformer’s hotness. In addition, The existing methods ignore the correlation between joint pairs.

- **Method**：Spatial-Temporal Transformer network(ST-TR)

   - Spatial Self-Attention module (SSA): Understand intra-frame interactions between different body parts;

   - Temporal Self-Attention module (TSA):model inter-frame correlations.

- **Conclusion**：A two-stream network which outperforms state-of-the-art models on both NTU-RGB+D 60 and NTU-RGB+D 120.



# Research Objective(s)/Motivation

> 作者的研究目标是什么？

&emsp;&emsp;作者目的是通过Spatial Self-Attention module (SSA) 和Temporal Self-Attention module (TSA) 提取自适应低层特征，建模人类行为中的交互。


## Contribution
- Author propose a novel two-stream Transformer-based model (both the Termporal and spatial dimensions)

- Spatial Self-Attention (SSA) & Temporal SelfAttention (TSA)

   - SSA module dynamically build links between skeleton joints, 该模块获取人体各部分之间的关系，与动作有关，而非完全遵守自然人体关节结构。

   - TSA study the dynamics of joints along time.


# Background / Problem Statement(Introduction)

>研究的背景以及问题陈述：作者需要解决的问题是什么？


## Problem Statement
1. The topology of the graph representing the human body is fixed for all layers and actions, preventing the extraction of rich representations(图表示人体的拓扑结构都是固定的，不能够提取丰富的表达)
2. 时空卷积都是基于2D卷积的，所以都受限于局部邻居的特征影响;
3. correlations between body joints not linked in the human skeleton(人体的关节点未连接的部分同样有关联性)。


# Method(s)

>作者解决问题的方法/算法是什么？是否基于前人的方法？基于了哪些？


![时空自注意力](https://github.com/zztchios/zztchios.github.io/raw/master/img/d96d9cea8760e8e0b9f05e17fad85bfe.png)
## Spatial Self-Attention (SSA)

&emsp;&emsp;如图1(a)所示, first calculate $$q_i^t\in \mathcal{R}^{dq}$$, $$k_i^t\in \mathcal{R}^{dq}$$ and $$v_i^t\in \mathcal{R}^{dq}$$;Then, 计算a query-key dot product 获取权重$\alpha_{i,j}^t\in matgh$(权重代表两个节点之间的关联性强度)。
a weighted sum is computed to obtain a new embedding for node $$i^t$$($$\sum$$的目的是为了获取节点新的嵌入)
$$a_{i.j}^t=\mathbf{q_i^t}\cdot \mathbf{k_j^t}^T,\forall{t}\in T, \mathbf{z}_i^t=\sum_jsoftmax_j(\frac{a_{i.j}^t}{\sqrt{d_k}})\mathbf{v}_j^t\tag{1}$$

&emsp;&emsp;Multi-head 自注意力经过重复H次嵌入提取过程，每次采用不同集合的学习参数。，从而获得节点嵌入$$z_{i_1}^t,…,z_{i_H}^t$$，所有参考$$i^t$$,如$$concat(z_{i_1}^t,…,z_{i_H}^t)\cdot W_O$$,并且构成SSA的输出特征。

&emsp;&emsp;==总结，这部分就是为了获取节点与其他节点在空间中的特征聚合==

&emsp;&emsp;因此，如图1a所示，节点的关系($$a_{i.j}^t$$ score)动态的预测；所有动作的关系结构并不是固定的，都是随着样本自适应改变。SSA操作和全连接的图卷积相似，但是核心values($$a_{i.j}^t$$ score)是基于骨架动作动态预测的。

## Temporal Self-Attention (TSA)

$$a_{i.j}^v=\mathbf{q_i^v}\cdot \mathbf{k_j^v},\forall{v}\in V, \mathbf{z}_i^v=\sum_jsoftmax_j(\frac{a_{i.j}^v}{\sqrt{d_k}})\mathbf{v}_j^v\tag{2}$$
$$i^v,j^v$$分别表示节点v在时刻i,j的情况。其他和SSA一样。

## Two-Stream Spatial Temporal Transformer Network
&emsp;&emsp;既然有了SSA和TSA，那么下一步就是为了合并。

![3a5b89dc02b240c2b2e618f2472086fa.png](https://github.com/zztchios/zztchios.github.io/raw/master/img/3a5b89dc02b240c2b2e618f2472086fa.png)

作者分别用SSA和TSA代替ST-GCN中的GCN和TCN

**Spatial Transformer Stream (S-TR)**
$$\mathbf{S-TR}(x)=Conv_{2D(1\times K_t)}(\mathbf{SSA}(x))$$. Following the original Transformer structure,Batch Normalization layer and skip connections are used。

**Temporal Transformer Stream (T-TR)**

$$\mathbf{T-TR}(x)=\mathbf{TSA}(GCN(x))$$. 

# Experiments

>作者如何评估自己的方法？实验的setup是什么样的？感兴趣实验数据和结果有哪些？有没有问题或者可以借鉴的地方？

++Datasets++:NTU RGB+D 60 and NTU RGB+D 120.


## Ablation Study

![a01ee82ff89a6afd0ae5402a49e5ae1a.png](https://github.com/zztchios/zztchios.github.io/raw/master/img/a01ee82ff89a6afd0ae5402a49e5ae1a.png)

STR stream achieves slightly better performance(+0.4%) than the T-TR stream. 原因：S-TR的SSA只有25个关节点，而时间维度相关需要大量的帧。并且在参数方面也是下降了的

![60fe8be7fd9a0fbc17d4a6ba57b5e42a.png](https://github.com/zztchios/zztchios.github.io/raw/master/img/60fe8be7fd9a0fbc17d4a6ba57b5e42a.png)

其中“playing with phone”,“typing”, and “cross hands” on S-TR 收益最大，上时间关联或者两个人的如：“hugging”, “point finger”, “pat on back”, on T-TR收益最大。

# Conclusion

>作者给出了哪些结论？哪些是strong conclusions, 哪些又是weak的conclusions（即作者并没有通过实验提供evidence，只在discussion中提到；或实验的数据并没有给出充分的evidence）?

# Notes(optional)

> 不在以上列表中，但需要特别记录的笔记。


# References(optional)

>列出相关性高的文献，以便之后可以继续track下去。

[[1]](https://arxiv.org/abs/1912.08435) Cho, S., Maqbool, M., Liu, F., Foroosh, H.: Self-attention network for skeletonbased human action recognition. In: The IEEE Winter Conference on Applications of Computer Vision. pp. 635–644 (2020)
[[2]](https://arxiv.org/abs/1907.11065)Zehui, L., Liu, P., Huang, L., Fu, J., Chen, J., Qiu, X., Huang, X.: Dropattention: A regularization method for fully-connected self-attention networks. arXiv preprint arXiv:1907.11065 (2019)


# 单词句型句式
## 单词
>记录单词，便于记忆

illumination 光照;启发;阐明

latent 潜在的

resolution 分辨率；解决；决心

literature 文献

proportionally 成比例地；适当地

tackle 应付；解决

conditionally 有条件地

outperform 胜过

depicted 描述

discriminant n. 可资辨别的因素；（数）判别式

slightly 稍微，轻微地

entire action 整个动作

surpassing 胜过的；卓越的

## 句型
>常用句型，便于将来写作


we face all these limitations by employing a modified Transformer self-attention operator.

我们面对这些局限性，通过采用一种改进的Transformer自注意力操作。

Along the temporal dimension, 沿着时间维度;按照i这个做下去

In our formulation，在我们的表述中

on the way 在。。。路上/过程中

in substitution to 以替代

In this case 在这种情况下

as far as  it concerns the SSA,  就SSA而言

in terms of 依据，在...方面

w.r.t. with respect to的缩写，意思是关于、谈到、涉及等。

make use of 利用

as we also did 我们也这样做了

## 长句
>便于理解

*conditionally on the action and independently from the natural human body structure（conditionally … and independently ）*

根据动作决定，而非取决于自然的人体结构。这句话主要是体现一个意境，不是为了一字一句翻译。

*capturing discriminant features that are not otherwise possible to capture with a standard convolution, being this limited by the kernel size.*

提取标准卷积无法提取的判别式特征，这个受限于卷积核size。

*As adding bones information demonstrated leading to better results in previous works*

在之前的工作中，已经证明加入骨骼信息可以获得更好的结果。

