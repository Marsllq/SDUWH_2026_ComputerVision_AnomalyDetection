# CV大作业 — 工业装配件缺失检测

## 项目架构
- tools/         工具脚本（ROI标定、帧提取）
- src/           核心代码
  - config.py       配置加载
  - preprocessing.py 视频解析 + ROI裁剪
  - tracker.py       运动检测 + 按件分组（帧差法 + 拉普拉斯门控）
  - feature_extractor.py  DINOv2特征提取
  - memory_bank.py  PatchCore记忆库 + kNN评分
  - detection.py    异常检测流程
  - visualization.py 仪表盘可视化
  - main.py         主流程编排
- config.yaml    共享配置

## 核心算法
- DINOv2 ViT-S/14 (timm, forward_features, 取patch tokens)
- PatchCore: 正常patch特征记忆库 → kNN余弦距离 → 阈值判定
- 按件检测: 运动状态分组帧为"件单位"，每件输出一个OK/NG

## 关键参数
- 跳过前60s摄像机抖动
- 训练取1-2分钟稳定段，step=30（≈1fps）
- 运动阈值25.0，模糊阈值80.0
- Coreset比例0.10，kNN k=3，阈值系数3.0
- 4096x2160 → 224x224输入

## 代码规范
- Python 3.9+, type hints
- 函数有docstring
- OpenCV BGR格式，matplotlib RGB格式
