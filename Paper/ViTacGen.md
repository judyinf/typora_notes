# ViTacGen: Robotic Pushing with Vision-to-Touch Generation

## Paper

### Related Works

#### Visual-Tactile Representation Learning

触觉反馈提高抓取(grasp)稳定性

visual-tactile fusion 联合表征学习(注意力机制)

#### Robotic Pushing

**末端执行器变形**用于接触参数估计：接触形状，面积，位置，力，表面纹理

物理属性推断：硬度，粗糙度，弹性

带有触觉感知的sim-to-real RL policy

- 传感器容易出现噪音、磨损和物理损坏；不同传感器制造上不一致，生成**接触深度图**来代替触觉传感器



## Overview

### 仿真

用预训练的RL策略从专家轨迹中收集**视觉-触觉（gt）**图片对

> **M2curl** ("Sample-efficient multimodal reinforcement learning via self-supervised representation learning for robotic manipulation") 

专家轨迹是在 Tactile Gym 2.0 仿真环境中生成的，具体生成逻辑如下：


   任务定义：推挤任务要求机器人将物体沿着一条预设的 2D 轨迹移动并通过一系列目标点 。


   路径生成：具体的目标轨迹是使用 OpenSimplex 噪声随机生成的，以确保轨迹的多样性和复杂性 。

   执行过程：在每一条生成的路径上，运行上述预训练的 RL 专家网络 。专家网络能够同时接收**视觉观测**、**真实触觉观测**（Tactile Ground Truth）以及**本体感知**（Proprioception，如机器人TCP坐标）信息控制机器人。

   运行专家策略：在仿真中，让预训练的 RL 专家网络执行推挤任务（共执行 1,000 个操纵序列）。

   **同步录制数据**


   视觉数据：记录**机器人视角的 RGB 图像序列**（Visual Image Sequence）。


   触觉数据：同步记录仿真器生成的**接触深度图（Contact Depth Images）**作为触觉真值 。这是通过计算传感器接触时的深度图与未接触时的参考深度图之差得到的 。

   数据处理：图像尺寸：将收集到的视觉和触觉图像都调整为 128×128 像素 。

   数据集划分：将收集到的 1,000 条序列按 7:2:1 的比例划分为训练集、验证集和测试集

### 训练VT-Gen

用视觉-触觉图片对训练VT-Gen（输入：机器人视角的 RGB 图像，输出：接触深图，用ground truth进行监督训练，损失函数：VGG loss）

**模型架构 **

输入 (Input): 视觉图像序列 $\mathcal{V}=\{v_{1},...,v_{N}\}$ 。

**粗粒度编码器** (Coarse Encoder $\mathcal{E}_{coarse}$): 用于提取初始的视觉特征图 $f_{coarse}^{v}$ 。

**跨模态注意力模块** (Cross-modal Attention $\mathcal{A}_{cm}$): 引入可学习的位置嵌入（Positional Embedding）$p$，通过注意力机制处理粗粒度特征：$\mathcal{A}_{cm}(f_{coarse}^{v}, p)$。注意力头数设置为 8 。

**细化编码器** (Refine Encoder $\mathcal{E}_{refine}$): 进一步处理特征以生成更深层、更精细的特征图 $f_{refine}^{v}$ 。

残差块 (Residual Blocks): 一系列相同的残差块用于增强特征表示并保留空间信息 。

**分层解码器** (Hierarchical Decoder): 由转置卷积层（Transposed Convolutional Layers）和上采样操作组成，用于恢复空间分辨率 。

输出 (Output): 生成的触觉接触深度图 $c^{gen}$ 。

PSNR, SSIM and LPIPS

### 训练RL policy VT-Con 

用冻结的VT-Gen(输入：机器人的 TCP 坐标、视觉和生成的触觉观察)

VT-Con: RL with Visual-Tactile Contrastive Learning

仿真场景搭建 Tactile Gym 2

外置Intel RealSense D435 RGB摄像头，视场角(FOV)为42°，安装在距机器人底座中心1米处，向下30°视角，摄像头参数在相应的模拟训练中保持一致。

350 time-steps per episode

sim-to-real randomization 物体质量，表面摩擦系数，视觉域随机化，包括摄像机视图、照明、背景和颜色变化

推动轨迹是使用 OpenSimplex 噪声随机生成的。



## Implementation 

VT-gen 1,000操作序列，batch_size=64

VT-Con 采用 Stable Baselines 3  中的 Soft Actor-Critic (SAC)  算法，

time_steps=1,000,000// batch_size=64// buffer_size=20,000

成功终止标准 物体中心与目标中心的距离 <2.5cm

优化器 Adam lr=1e-4, epsilon=1e-8

- fusion module: addition, concatenation, and **attention**

- contrastive learning module: SimCLR, **MoCo**

#### 真机实验

UR5e机械臂，500Hz，350 time-steps per episode，xy 平面内 800 × 600 毫米，固定高度为桌面上方 2 厘米

外置Intel RealSense D435 RGB摄像头，视场角（FOV）为42°，安装在距离机器人底座中心1米处，向下30°视角，并且摄像头参数在相应的模拟训练中保持一致。

#### 仿真

域随机化：物体质量、表面摩擦系数和视觉域随机化，包括摄像机视图、照明、背景和颜色变化

推动轨迹是使用 OpenSimplex 噪声随机生成的



### Pretrain expert policy                       

Pretrain visual-tactile expert policy (VT-Con) for data collection:

```bash
python tactile_gym/sb3_helpers/train_agent.py --exp_name vtcon -A mvitac_sac --features_extractor_class VisualTactileCMCL --seed 0 --learning_rate 1e-4
```

### Collect visual-tactile data pairs

Collect paired visual-tactile data with expert policy:

```bash
python tactile_gym/sb3_helpers/data_collection.py # Replace saved_model_dir with the path to your pretrained model
```

### Train vision-to-touch generation (VT-Gen)

Train VT-Gen for vision-to-touch generation:

### Train policy with vision-to-touch generation (ViTacGen)

Train visual-only ViTacGen policy:

```bash
python tactile_gym/sb3_helpers/train_agent.py --exp_name vitacgen -A mvitac_sac --features_extractor_class VisualCMCL_atten --seed 0 --learning_rate 1e-4
```





