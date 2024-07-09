# AIGC Tutorial
 
# 深度渲染（Neural Rendering）入门手册
个人理解深度生成应该属于计算机图形学方向的分支

## 1.基础机器学习
李航 《统计学习方法》看第一章熟悉概念

链接：https://pan.baidu.com/s/135q4YvqHrcCX9edXbwmkZQ 

提取码：1p2b 
周志华 《机器学习》

链接：https://pan.baidu.com/s/101jVuJvKcrSO_Z66XJqgCw 

提取码：616b 


## 2.操作系统环境

ubuntu16.04版本的基础使用

硬盘挂载mount、硬盘使用情况查看df, du, fdisk等



### nvidia显卡驱动安装教程 (nvidia驱动安装 卸载)：

根据显卡型号到nvidia的官网下载对应的```*.run```的安装文件，将该文件放置在待安装的机器中，切换到对应文件夹下
安装方法
<pre><code>
sudo apt-get purge nvidia*
sudo service lightdm stop
sudo ./<b>[replace_with_download_file_name]</b>.run
</code></pre>

### CUDA与cudnn安装
***杜绝使用apt-get install安装任何cuda相关的软件包！！！！！***
- 查看系统中是否安装nvidia显卡驱动（***显卡驱动也严格杜绝使用apt-get install安装！！！！***），如存在显卡驱动请使用nvidia-smi查看“CUDA Version”版本，该版本即为已安装驱动能支持的最高CUDA版本
- 下载目标版本的CUDA与Cudnn，[CUDA下载地址](https://developer.nvidia.com/cuda-downloads)，[Cudnn下载地址](https://developer.nvidia.com/cudnn)需要注册账号。***注意看版本号，如果目标版本不是最新版本号，CUDA需要通过"Archive of Previous CUDA Releases"选择目标版本***。CUDA下载和Cudnn都可以通过wget直接下载。
- 安装CUDA，下载得到的```*.run```文件，首先需要赋予其执行权限，即通过
<pre><code>
sudo chomod +x <b>[replace_with_download_file_name]</b>.run
</code></pre>
来实现，然后通过
<pre><code>
sudo ./<b>[replace_with_download_file_name]</b>.run
</code></pre>
安装CUDA，请注意如果已有显卡驱动满足版本需求即不需重复安装显卡驱动，example、doc都不需要安装。
- 在目录```/usr/local/```建立软连接```cuda```，将其链接到上一步骤安装的cuda目录上，***注意：这个软链接一般在安装CUDA时候会提示是否自动创建，如已经创建，此步骤跳过***，具体命令为
<pre><code>
sudo ln -s /path/to/installed_cuda /usr/local/cuda
</code></pre>
这样可以通过该软链接切换不同cuda的安装版本。
- 安装Cudnn，下载得到压缩包```cudnn-linux-x86_64-******-archive.tar.xz```，将其解压
<pre><code>
tar xf cudnn-linux-x86_64-******-archive.tar.xz
</code></pre>
解压后在当前目录得到文件夹```cudnn-linux-x86_64-******-archive/```。将相关文件拷贝到cuda安装目录中，顺序执行以下三条命令：
<pre><code>
  sudo cp cudnn-linux-x86_64-******-archive/include/cudnn.h /usr/local/cuda/include
  sudo cp cudnn-linux-x86_64-******-archive/lib/libcudnn* /usr/local/cuda/lib64
  sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
</code></pre>
  
- 设置环境路径：
- 在```~/.bashrc```
<pre><code>
sudo vim ~/.bashrc
</code></pre>  
中加入:
<pre><code>
export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64/"
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64"
</code></pre>  
按```ctrl+wq```保存并退出，然后执行:
<pre><code>
source ~/.bashrc
</code></pre> 
使环境路径设置生效。
- 验证是否安装成功```nvcc -V```:
<pre><code>
nvcc -V
</code></pre> 

nvidia-smi命令使用

gcc如何安装，多版本切换

ssh服务开启及远程使用

docker容器安装与使用 nvidia-docker安装与使用


## 3.语言要求

### python

要求掌握python基本语法：分支（if elif）、循环（while for）、python的独有缩进方式、异常处理机制（try exception）、class（类的编写）

系统功能掌握：文件读取（file，注意资源的打开和关闭养成良好习惯）、JSON文件读取、CSV文件读取、loging模块掌握、os模块掌握（能熟练的对文件、目录增删改查）、pathlib、multiprocessing多进程模块的使用、定时器Timer的使用

业务相关库掌握：PIL（图像处理库）、opencv(cv2图像处理库掌握)、numpy（矩阵预算库掌握）、scipy（矩阵运算库掌握）、argparse库掌握（脚本参数化输入输出）、prettytable（表格化输出）

## 4.深度学习基础

### 深度学习工具：

pytorch 1.0及以上版本

tensorflow 2.0以前版本，1.12版

anaconda python虚拟环境使用

visual studio code 代码编辑环境

机器远程连接软件：MobaXterm

## 5.重要数据及课程

斯坦福大学CS231n网络课程

**Deep Learning (对于希望系统性了解整个深度学习理论架构并且已经具备初阶深度学习基础的同学)**

https://github.com/exacity/deeplearningbook-chinese/releases/download/v0.5-beta/dlbook_cn_v0.5-beta.pdf
nndl-ebook "Neural Networks and Deep Learning"

链接：https://pan.baidu.com/s/1RdhbExDBGly50kjTuET1ig 

提取码：khon 

***冲着读深度生成（现在称AIGC）方向的博士***
请把***Pattern Recognition and Machine Learning***(PRML)倒背如流！

[英文PDF](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

## 6.计算机领域论文搜索

CVF Open access （包括ICCV CVPR ECCV近年的论文，可以找到论文附带的补充材料） https://openaccess.thecvf.com/menu

arxiv (预印版，为了快速发布抢占Idea的归属权，此数据库不属于发表，一般论文会为了增加自己的引用可能都会放出预印版)  https://arxiv.org/

google

针对计算机图形学领域：直接google siggraph (asia) paper list既可以搜到近些年的官方给出的paper list

## 7.需要了解的基本概念：

了解SGD（随机梯度下降）原理

了解mini-batch概念，为什么要用mini-batch，有什么作用

了解权重更新策略：AdaGrad算法、RMSProp算法、AdaDelta算法、Adam算法、radam算法

VGG神经网络，原论文：https://arxiv.org/abs/1409.1556

ResNet神经网络，原论文：https://arxiv.org/abs/1512.03385

Batch Normalization特征空间正规化加速网络训练，建议查阅博客深入理解后有兴趣可查阅论文，原论文：https://arxiv.org/abs/1502.03167

Instance Normalization

白化whitening 与coloring

conditional batch normalization

adaptive instance normalization https://arxiv.org/pdf/1703.06868.pdf

conditional instance normalization


## 8.深度生成相关知识：

理解GAN，请先查阅相关博客资料，原论文：https://arxiv.org/abs/1406.2661

DCGAN 首先采样卷积上采样实现图像生成，原论文：http://arxiv.org/abs/1511.06434

WGAN 请查阅相关博客，原论文推导复杂

WGAN-GP 请查阅相关博客，原论文推导复杂

SNGAN https://arxiv.org/pdf/1802.05957.pdf

SAGAN https://arxiv.org/pdf/1805.08318.pdf

**StyleGAN**(重点掌握) https://arxiv.org/pdf/1812.04948.pdf

**StyleGAN V2**(重点掌握) https://arxiv.org/abs/1912.04958

MSG-GAN https://arxiv.org/abs/1903.06048

了解TTUR机制

了解Hinge loss

了解checkerboard artifacets（棋盘伪影）及其解决方法 https://distill.pub/2016/deconv-checkerboard/

对于transpose conv的棋盘伪影从频谱角度的解决方案 https://arxiv.org/pdf/2003.01826.pdf


### Conditional Generation：

**ACGAN** https://arxiv.org/pdf/1610.09585.pdf

**cGANs with projection Discriminator** https://arxiv.org/pdf/1802.05637.pdf 请查阅相关博客，尽量理解，未理解可以找我讨论

**BigGAN** https://arxiv.org/pdf/1809.11096.pdf 大batch size版的cGANs with projection Discriminator

conditional coloring https://arxiv.org/pdf/1806.00420.pdf

Multi-Hinge https://arxiv.org/pdf/1912.04216.pdf

**我们自己的face attributes editing** https://link.springer.com/chapter/10.1007/978-3-030-58621-8_39

**我们自己的换脸** https://github.com/neuralchen/SimSwap


### Image translation:

StarGAN 改进版ACGAN https://arxiv.org/pdf/1711.09020.pdf

StarGAN v2 https://arxiv.org/pdf/1912.01865.pdf

cycle GAN https://arxiv.org/pdf/1703.10593.pdf

pixel2pixel https://arxiv.org/pdf/1611.07004.pdf

pixel2pixelHD  https://arxiv.org/pdf/1711.11585.pdf

### Style Transfer:

Nerual style transfer review https://arxiv.org/pdf/1705.04058v4.pdf

adaptive instance normalization https://arxiv.org/pdf/1703.06868.pdf

Patch Swap https://arxiv.org/pdf/1612.04337.pdf

WCT https://arxiv.org/pdf/1705.08086.pdf

Perceptual Loss https://arxiv.org/pdf/1603.08155.pdf

Style Aware(效果SOTA)https://openaccess.thecvf.com/content_ECCV_2018/papers/Artsiom_Sanakoyeu_A_Style-aware_Content_ECCV_2018_paper.pdf

**我们自己的论文** https://dl.acm.org/doi/abs/10.1145/3394171.3413770

### Neural Rendering综述
**非常关键** https://arxiv.org/abs/2004.03805


## 9. 计算机图形学基础

课程网站 https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html

课程视频  https://www.bilibili.com/video/BV1X7411F744
## 10.科学上网

https://github.com/shadowsocks/shadowsocks-windows

## 11.杂项

一个下书的丧心病狂的网站 http://gen.lib.rus.ec/
