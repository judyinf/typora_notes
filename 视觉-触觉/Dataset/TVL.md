# TVL dataset

| **元数据**                        | **详细说明**                                                 |
| --------------------------------- | ------------------------------------------------------------ |
| **名称 (Name)**                   | **TVL Dataset** (Touch-Vision-Language Dataset)              |
| **采集设备 (Acquisition Device)** | **触觉传感器**：**DIGIT**（一种低成本、开源的高分辨率视触觉传感器，基于GelSight原理）。   **视觉传感器**：**Logitech BRIO** 网络摄像头（用于捕捉宏观视觉图像）。  **采集装置**：定制的 **3D打印手持采集支架**，将DIGIT和摄像头固定在一起，确保触觉接触点位于摄像头的视野中心，并实现时空同步。 |
| **采集流程 (Collection Process)** | 数据集包含两部分来源：  1. **机器人采集 (Robot-Collected)**：使用UR5机械臂在实验室环境下自动采集（基于之前的SSVTP数据集）。  2. **人工采集 (Human-Collected, HCT)**：研究人员手持上述采集装置，在“野外”（非实验室受控环境）对各种日常物体表面进行按压和滑动，捕捉触觉和视觉数据。  **标注流程**：采用“人机协作”标注。约10%的数据由人类标注员根据触觉感受撰写自然语言描述（如 "rough", "bumpy"），剩余90%利用 **GPT-4V** 根据视觉图像生成伪标签（Pseudo-labels）。 |
| **规模与类型 (Scale & Type)**     | **规模**：总计约 **44,000 (44K)** 组对齐的三模态数据对。  **类型**：  - **Vision**: RGB图像（宏观视角）。  - **Touch**: RGB触觉图像（微观纹理/形变）。  - **Language**: 英语自然语言描述（描述触觉感受）。 |
| **数据组织结构 (Organization)**   | 数据集主要分为两个子集：  1. **SSVTP (Self-Supervised Visuo-Tactile Pretraining)**：包含约 4,587 对由机器人采集的数据。  2. **HCT (Human Collected Tactile)**：包含约 39,154 对由人类在野外采集的数据。  目录下通常包含 `image/` (视觉图), `tactile/` (触觉图), 和 `metadata.csv` (包含对应的文本描述和对齐索引)。 |
| **适用任务 (Applicable Tasks)**   | 1. **开放词汇触觉分类 (Open-Vocabulary Tactile Classification)**：根据文本描述识别触觉纹理。  2. **视触觉-语言对齐 (Vision-Touch-Language Alignment)**：训练多模态编码器（如Tactile-CLIP），学习三者在同一特征空间的表达。  3. **跨模态生成 (Cross-Modal Generation)**：输入视觉图像，生成描述其触觉感受的文本（即“看图知感”）。 |



