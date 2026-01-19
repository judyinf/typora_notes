# Paper Reading: Embodied AI 0

> Review: Embodied AI; Tactile Robotics

[TOC]

### Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning  

https://arxiv.org/abs/2508.10399v1



### Tactile Robotics: An Outlook

https://arxiv.org/pdf/2508.11261



### When Vision Meets Touch: A Contemporary  Review for Visuotactile Sensors from the Signal  Processing Perspective

http://arxiv.org/abs/2406.12226





### Multi-Modal Perception with Vision, Language, and Touch for Robot Manipulation

**机器人感知，推理，行动**

**性能：** **泛化性**，适应性，**安全性**，精度，**高效性**  generalization, adaptability, safety, precision, efficiency

**指标：** 成功率，规划时间，泛化能力

**关键词：**运动控制、运动规划、模仿学习、机械搜索、丰富的接触操作和多模态对齐

**挑战** 

- 视觉空间（例如照明、物体放置或背景）的变化可能会导致经过训练的机器人策略的**分布变化**(distribution shift)，从而导致策略性能下降。
- 现实世界的场景通常很混乱，需要机器人**运动规划**(motion planning)以避免碰撞，而传统方法需要完整的状态信息，并且很难快速规划实时操作。
- 现实环境中对象和任务的多样性要求机器人能够有效地进行**泛化**(generalize)，而无需针对每个新场景进行大量微调。

**模态提供的信息和局限性**

视觉：物体位置，几何形状，纹理

> 在操作任务期间受到遮挡，并且难以估计重量或摩擦力等物理特性。

本体感觉：机器人关节位置，速度和力的反馈

> 无法获知接触点位以外的信息。

文本：从细节层面设定任务目标、给出任务指令，推理物体可操作性(Object Affordances)、任务层次结构(Task Hierarchies)和上下文约束( Contextual Constraints)，理解物体相互关系和规则，生成高级规划

> 无法提供环境上下文 。

触觉：细粒度的接触信息

> 仅限于局部区域

**结合思路**

1. 视觉+本体感觉  Vision and Proprioception in Safe and  Generalizable Robot System Design
2. 视觉+语言 Vision and Language in Efficient and  Generalizable Robot System Design
3. 视觉+触觉 Vision and Touch for Precise Robot Manipulation
4. 视觉+语言+触觉 Aligning Vision, Language and Touch

Vision and Proprioception in **Safe** and  **Generalizable** Robot System Design

> 1. Conformal policy learning for sensorimotor control under distribution shifts 检测并根据运动控制中的**分布变化**调整优化策略，实现安全性和性能的权衡。
>
> 2. Diffusionseeder: Seeding motion optimization with diffusion for rapid motion planning 在杂乱的环境下，利用深度图像生成无碰撞多模态轨迹进行**运动规划**；速度优化。
> 3. In-Context Imitation Learning via Next-Token Prediction 让机器人通过观察一些演示来适应新的操作任务。小样本泛化能力。

Vision and Language in **Efficient** and  **Generalizable** Robot System Design

> VLMs and LLMs explicitly for mechanical search (VLMs for scene understanding and LLMs for affinity generation)
>
> 1. 几何推理+机械搜索策略，依次推动遮挡对象以显示隐藏目标，减少搜索时间
>
> 2. 使用VLM进行场景理解，LLM基于对象的亲和/关系、生成语义遮挡分布
> 3. Otter VLA 提取与语言指令对应的视觉特征，结合下游策略网络精细控制

Vision and Touch for **Precise** Robot Manipulation

> 触觉信号是局部性的，需要与其它模态结合；探索**自监督**的方法
>
> 1. 将插入任务分为使用基于触觉的抓取姿势估计的**对齐阶段**和由基于视觉的策略引导的**插入阶段**。
> 2. multi-task 预训练：视觉和触觉的对比损失

Aligning Vision, Language and Touch

> TVL dataset

**接触丰富的任务场景**

工业装配（刚性物体）

- **极高的精度要求 (High Precision):** 间隙往往只有 0.1mm 甚至更小（称为 Tight Tolerance），超过了普通工业机器人的重复定位精度。
- **接触力控制 (Contact-Rich Interaction):** 一旦发生接触，微小的位置误差会导致巨大的反作用力，可能损坏零件或机器人。
- **视觉盲区:** 当零件接近孔位时，机器人手臂或零件本身会遮挡相机视野。

衣物/柔性物体处理

- **无限自由度 (Infinite Degrees of Freedom):** 刚体只有6个自由度（位置+旋转），而衣物可以任意折叠、弯曲，状态空间几乎是无限的。
- **自遮挡 (Self-Occlusion):** 衣服堆在一起时，机器人看不清它的全貌，也不知道抓起一个角后整件衣服会变成什么样。
- **欠驱动特性 (Underactuated):** 你只能控制抓取点，但无法直接控制衣服的其他部分（比如你抓着衣领，衣角会因重力甩动，难以预测）。



### Scholarpedia of Touch