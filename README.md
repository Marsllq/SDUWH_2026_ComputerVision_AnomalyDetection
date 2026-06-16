# 基于计算机视觉的工业装配件缺失检测

本项目完成 Task A 与 Task B 两段工业流水线视频的装配件缺失检测。训练阶段只使用视频前段正常样本，测试阶段按稳定工件输出 `OK` / `NG`，并生成带 ROI、分数和状态面板的演示视频。

## 方法概述

- 特征提取：DINOv2 ViT-S/14 patch tokens
- 异常检测：PatchCore memory bank + kNN 距离评分
- Task A：动态定位圆形端面后进行单 ROI PatchCore 检测
- Task B：4 个帽位分别训练 per-ROI PatchCore，并用蓝/白帽位 presence 作为辅助门控
- 分件策略：跳过视频前 60 秒抖动，只在工件稳定到位时检测；空场景和运动过程显示为等待/移动，不输出 OK/NG

## 目录结构

- `src/`：核心算法代码
- `tools/run_demo.py`：训练、推理和演示视频生成入口
- `config.yaml`：Task A / Task B 配置
- `dataset/`：原始视频和参考图
- `results/`：演示视频和中间可视化结果
- `report/cv_final_report.pdf`：最终实验报告

## 环境安装

```bash
pip install -r requirements.txt
```

## 运行演示

生成 Task A 15 秒演示：

```bash
python tools/run_demo.py taskA 360 30 2
```

生成 Task B 15 秒演示：

```bash
python tools/run_demo.py taskB 1045 30 2
```

输出文件默认写入：

- `results/taskA_demo_result.mp4`
- `results/taskB_demo_result.mp4`

已整理的提交短视频副本：

- `results/taskA_15s_demo.mp4`
- `results/taskB_15s_demo.mp4`

## 最终产物

- 代码仓库：包含预处理、特征提取、训练、推理和可视化代码
- 演示视频：`results/taskA_15s_demo.mp4`、`results/taskB_15s_demo.mp4`
- 实验报告：`report/cv_final_report.pdf`
