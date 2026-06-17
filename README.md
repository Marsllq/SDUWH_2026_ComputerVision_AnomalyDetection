# 🔧 基于计算机视觉的工业装配件缺失检测

> **计算机视觉课程大作业** · Task A & Task B 工业流水线装配质检

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 项目概述

本项目完成 Task A 与 Task B 两段工业流水线视频的装配件缺失检测。核心思路是将"盖子是否装配到位"转化为**正常模式建模问题**：训练阶段只使用视频前段正常样本建立特征库，推理阶段按稳定工件输出 `OK` / `NG`，并生成带 ROI、分数和状态面板的演示视频。

### ✨ 主要特性

- 🧠 **无监督异常检测** — 仅需正常样本训练，无需异常标注
- 🎯 **按件检测** — 自动识别工件稳定到位后才进行质检，避免半进入/运动误报
- 🔍 **动态 ROI 定位** — Task A 自适应端面对齐，解决位置漂移
- 🧩 **多帽位独立建模** — Task B 四个帽位各自独立 PatchCore，避免语义混淆
- 🎬 **可视化演示** — 自动生成带状态面板、ROI 框和 OK/NG 标注的演示视频

---

## 🔬 方法概述

| 模块 | 方法 | 说明 |
|:---|:---|:---|
| 特征提取 | **DINOv2 ViT-S/14** | 自监督视觉表征，提取 patch token 局部特征 |
| 异常检测 | **PatchCore + kNN** | 正常特征记忆库 + 最近邻距离评分 |
| Task A 策略 | 动态定位 + 单 ROI PatchCore | 圆形端面逐帧对齐后检测 |
| Task B 策略 | 4 ROI 独立 PatchCore | 蓝/白帽位分别建模，presence 辅助门控 |
| 分件策略 | 运动/前景/模糊门控 | 跳过前 60s 抖动，只在工件稳定时检测 |

---

## 📁 目录结构

```
.
├── src/                          # 核心算法代码
│   ├── config.py                 # 配置加载
│   ├── preprocessing.py          # 视频读取、ROI 裁剪、patch 提取
│   ├── tracker.py                # 运动门控与工件分件
│   ├── locator.py                # Task A 动态端面定位
│   ├── feature_extractor.py      # DINOv2 特征提取
│   ├── memory_bank.py            # PatchCore memory bank + kNN 评分
│   ├── detection.py              # 单/多 ROI 检测器
│   └── visualization.py          # 仪表盘可视化
├── tools/
│   └── run_demo.py               # 训练、推理和演示视频生成入口
├── config.yaml                   # Task A / Task B 全局与任务级配置
├── dataset/                      # 原始视频和参考图
├── results/                      # 演示视频和中间可视化结果
├── report/
│   ├── cv_final_report.tex       # 实验报告 LaTeX 源码
│   └── cv_final_report.pdf       # 最终实验报告 PDF
├── guide.md                      # 作业要求说明
└── requirements.txt              # Python 依赖
```

---

## 🚀 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 生成 Task A 演示视频（15 秒）

```bash
python tools/run_demo.py taskA 360 30 2.2
```

### 生成 Task B 演示视频（15 秒）

```bash
python tools/run_demo.py taskB 1050 30 2.2
```

### Task B 长片段回归（可选）

```bash
python tools/run_demo.py taskB 1040 50 2
python tools/run_demo.py taskB 250 50 2
python tools/run_demo.py taskB 930 50 2
```

---

## 📊 实验结果速览

| 任务 | 演示片段 | 输出工件 | OK | NG | 关键异常点 |
|:---|:---|:---:|:---:|:---:|:---|
| Task A | 360s–390s | 5 | 2 | 3 | 374s 起异常段 |
| Task B | 1050s–1080s | 7 | 5 | 2 | 1050s, 1072s |

---

## 📦 最终产物

| 产物 | 路径 | 说明 |
|:---|:---|:---|
| 代码仓库 | 本项目目录 | 预处理、特征提取、训练、推理和可视化 |
| Task A 演示 | `results/taskA_15s_demo.mp4` | 15 秒短视频，含正常/异常对照 |
| Task B 演示 | `results/taskB_15s_demo.mp4` | 15 秒短视频，含正常/异常对照 |
| 实验报告 | `report/cv_final_report.pdf` | 完整方法、结果、分析与改进讨论 |

---

## 📝 引用

```bibtex
@inproceedings{roth2022patchcore,
  title     = {Towards Total Recall in Industrial Anomaly Detection},
  author    = {Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Schölkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2022}
}

@article{oquab2023dinov2,
  title   = {DINOv2: Learning Robust Visual Features without Supervision},
  author  = {Oquab, Maxime and Darcet, Timothée and Moutakanni, Théo and others},
  journal = {arXiv preprint arXiv:2304.07193},
  year    = {2023}
}
```
