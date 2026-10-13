"""tools/manifest.py —— experiment_manifest.yaml 的唯一读取入口

干什么：
    所有 Tool 需要引用论文数字（如 MAE 质量口径）时，一律从这里读，
    禁止在代码里硬编码数字（开发文档 §2 数字口径纪律）。

为什么单独一个模块：
    manifest 的加载/解析只有一处，将来加缓存、加 schema 校验、
    换存储格式（yaml→db）都只改这一个文件，6 个工具零改动。
"""

from __future__ import annotations

from pathlib import Path

import yaml

# 项目根目录 = tools/ 的上一级；manifest 路径固定，属于项目约定
_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "experiment_manifest.yaml"


def load_manifest() -> dict | None:
    """读取 manifest；文件缺失或 YAML 解析失败时返回 None（由调用方转为 DATA_MISSING）。

    返回 None 而不是抛异常：工具层的纪律是"不抛异常给 LLM"，
    加载失败只是 DATA_MISSING 的一种，走统一错误协议。
    """
    try:
        return yaml.safe_load(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
