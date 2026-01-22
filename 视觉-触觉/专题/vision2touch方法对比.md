# vision2touch方法对比

| 论文                                                         | 输入                             | 输出                     | 特点             | 缺点                       | 适配任务                            |
| ------------------------------------------------------------ | -------------------------------- | ------------------------ | ---------------- | -------------------------- | ----------------------------------- |
| Generating Visual Scenes from Touch 2023 http://arxiv.org/abs/2309.15117 //UniTouch 2024 | 按压物体表面的图片(有手？)       | 视触觉图                 | 反应物体物理属性 | 无法反映接触状态           | 传感器有关                          |
| TactGen https://ieeexplore.ieee.org/document/10815063/media#media | 环绕物体一周的视频               | 视触觉图                 | 查询特定接触位置 | 输入难以获取               | 接触位置定位                        |
| ViTacGen http://arxiv.org/abs/2510.14117                     | 第三视角全局相机记录物体运动视频 | 对应接触深度图（灰度图） | 反映动态接触深度 | 缺乏细粒度表面纹理和力分布 | 场景是Pushing，适合推理阶段实时生成 |
| Visuo-Tactile Cross Generation ObjectFolder Real https://github.com/objectfolder/visuo-tactile-cross-generation 2023 | 局部图片                         | 视触觉图                 | 位置和视角一致性 |                            | 传感器有关                          |



### 触觉信息

物理属性：硬度，粗糙度，弹性，材料（塑料、瓷器etc）

接触状态：接触位置，面积，形状，接触力，穿透深度



### 效果验证

1. 有GT的数据集测试生成图片结构相似度等
2. 利用生成的触觉图进行仿真/真机实验，需要额外确定推理模型架构和策略 



### 数据集

规模：总条数/时长，机器人本体类型，操作任务，物体类别

采集主体：人类（是否遥操）/机器人（机械臂，夹爪，灵巧手等）/仿真/合成

采集设备：传感器，相机类型（第一视角/第三视角/腕部/顶部/头部等）

采集流程：环境、物体；操作任务：接触/按压/滑动等

数据类型：图像/视频（时长、fps）；视触觉图片（是否包含marker，是否接触深度灰度图）；包含什么过程；是否有标签

适用任务：



| 数据集                    | 规模 | 采集主体 | 采集流程 | 采集设备 | 数据类型 |      |
| ------------------------- | ---- | -------- | -------- | -------- | -------- | ---- |
| Open X-Embodiment         |      |          |          |          |          |      |
| ObjectFolder 1.0          |      |          |          |          |          |      |
| ObjectFolder 2.0          |      |          |          |          |          |      |
| ObjectFolder Real         |      |          |          |          |          |      |
| Touch100k                 |      |          |          |          |          |      |
| SSVTP                     |      |          |          |          |          |      |
| TVL                       |      |          |          |          |          |      |
| Visuo-Tactile Video (VTV) |      |          |          |          |          |      |
| ControlTac                |      |          |          |          |          |      |
| Touch and Go              |      |          |          |          |          |      |
| ObjTac                    |      |          |          |          |          |      |
| YCB-Slide                 |      |          |          |          |          |      |
| The Feeling of Success    |      |          |          |          |          |      |
| VTDexManip                |      |          |          |          |          |      |
| FreeTacMan                |      |          |          |          |          |      |



### 任务

#### Object Manipulation

- Grasp-Stability Prediction 抓取稳定性
- Contact Refinement 接触位置调整
- Surface Traversal 表面遍历
- Dynamic Pushing 推物体

#### Object Reconstruction

- 3D Shape Reconstruction
- Sound Generation of Dynamic Objects
- Visuo-Tactile **Cross-Generation**

#### Object Recognition

- Cross-Sensory Retrieval
- Contact Localization **接触位置定位**
- Material Classification

![](/home/temp2/TrigYei/Project/demo_figure/object_manipulation.png)

![object_reconstruction](/home/temp2/TrigYei/Project/demo_figure/object_reconstruction.png)

![object_recognition](/home/temp2/TrigYei/Project/demo_figure/object_recognition.png)
