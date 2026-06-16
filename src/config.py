"""配置加载模块"""
import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config(task_name: str = None):
    """加载配置文件，可选指定特定task的配置"""
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    if task_name:
        task_cfg = cfg.get("tasks", {}).get(task_name, {})
        cfg.update(task_cfg)
        cfg["task_name"] = task_name

    # 路径解析
    cfg["project_root"] = PROJECT_ROOT
    for key in ["video_path", "output_path"]:
        if key in cfg and cfg[key] and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(PROJECT_ROOT, cfg[key])

    return cfg
