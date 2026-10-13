"""tools/predict_traffic_speed.py —— Tool 1：区域速度预测（当前为 mock 模式）

干什么：
    预测指定区域未来 N 步平均车速，是六层能力中的 Perception（感知）层——
    调度决策之前，先观察未来的交通状态。

为什么这样设计：
    1. mock 优先（冻结决策 §0.3-2）：论文 STGCN 接入前，用确定性伪数据
       （sha256 哈希生成，同输入必得同输出）；论文代码到达后只替换
       _load_speeds() 内部实现，本文件对外签名一行不动。
    2. quality_reference 直接读 manifest：把"模型测试集 MAE"作为参考口径
       随结果返回，并显式标注"测试集均值，非单次预测置信度"——
       防止 LLM 拿着 2.78 去回答"这次预测准不准"这类问题。
    3. time_slice 参数是对开发文档 §5.1 的已批准偏差（2026-09-30）：
       "未来 N 步"必须锚定一个起始时刻，否则预测没有时间起点，
       mock 也无法确定性复现。

对应面试点：Function Calling 的 Schema 即合同（含负向约束）；
            provenance/质量口径随结果返回，防止数字被错误引用。
"""

from __future__ import annotations

import hashlib
import time

from typing import Annotated

from pydantic import BaseModel, Field, ValidationError

from .base import ErrorCode, ToolResult
from .manifest import load_manifest

# ---------- 口径常量（来源：experiment_manifest.yaml 的 verified 项） ----------
N_REGIONS = 156            # SZ_TAXI 区域数（verified）
MAX_HORIZON_STEPS = 12     # STGCN output_window=12（verified：日志 exp_id=87576）
SPEED_CAP_KMH = 86.43      # 论文 dispatch_thesis_v9_fast.py 的速度上限口径

# ---------- mock 生成器参数 ----------
MOCK_VERSION = "mock-v1"
MOCK_SPEED_MIN, MOCK_SPEED_MAX = 5.0, 60.0  # km/h，覆盖拥堵到畅通的合理区间


class PredictTrafficSpeedInput(BaseModel):
    """输入 Schema = LLM 与本工具之间的合同。description 会进入
    Function Calling 的 parameters，负向约束写在这里防乱调。"""

    # 注意：取值范围必须用 Annotated 约束声明（机器可执行），
    # description 里的"严禁越界"只是给 LLM 看的文档，没有强制力。
    region_ids: list[Annotated[int, Field(ge=0, le=N_REGIONS - 1)]] = Field(
        min_length=1,
        description=(
            f"区域索引列表，取值 0–{N_REGIONS - 1}（矩阵索引，非原始 geo_id）。"
            f"严禁传入越界 ID 或空列表。"
        ),
    )
    horizon_steps: int = Field(
        default=MAX_HORIZON_STEPS, ge=1, le=MAX_HORIZON_STEPS,
        description=(
            f"预测步数（每步 15 分钟），默认 {MAX_HORIZON_STEPS}。"
            f"严禁 >{MAX_HORIZON_STEPS}：超出训练 output_window 是不可靠外推。"
        ),
    )
    time_slice: int = Field(
        default=0, ge=0,
        description="预测锚定的起始时间片索引（0 起）。预测的是该时刻之后的 N 步。",
    )
    use_cache: bool = Field(
        default=True,
        description="是否优先读预计算缓存。论文接入前此参数不影响 mock 行为。",
    )


def _mock_speed(region_id: int, time_slice: int, step: int) -> float:
    """确定性伪速度：sha256(region|time_slice|step) → [MOCK_SPEED_MIN, MOCK_SPEED_MAX]。

    同输入必得同输出（测试不变量 #3 依赖这一点）。
    TODO(论文接入)：整个函数替换为从 STGCN 预计算 npz 缓存读取
    （predictions[time_slice, step, region]，注意 npz 末 49 行为
    pad_with_last_sample 填充，消费时按 manifest.test_slices=591 截断）。
    """
    key = f"{MOCK_VERSION}|{region_id}|{time_slice}|{step}".encode()
    frac = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return round(MOCK_SPEED_MIN + frac * (MOCK_SPEED_MAX - MOCK_SPEED_MIN), 4)


def _format_validation_error(exc: ValidationError) -> str:
    """把 Pydantic 校验错误翻译成给人/LLM 看的一句话：哪个字段、什么值、为什么不行。"""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        parts.append(f"字段 {loc}: {err['msg']}（输入值: {err.get('input')!r}）")
    return "参数校验失败——" + "；".join(parts)


def predict_traffic_speed(params: dict) -> ToolResult:
    """工具入口。编排层只传 dict、只收 ToolResult，永不抛异常。"""
    start = time.perf_counter()

    # ── 1. 参数校验：非法参数是调用方的问题，不重试 ──────────────────
    try:
        inp = PredictTrafficSpeedInput.model_validate(params)
    except ValidationError as exc:
        return ToolResult.fail(ErrorCode.INVALID_PARAM, _format_validation_error(exc))

    # ── 2. 质量口径来自 manifest；manifest 缺失时拒绝返回预测 ────────
    #    （否则 LLM 会在没有质量参考的情况下引用预测值，违反数字口径纪律）
    manifest = load_manifest()
    if manifest is None:
        return ToolResult.fail(
            ErrorCode.DATA_MISSING,
            "data/experiment_manifest.yaml 缺失或不可读；无法附带质量口径，拒绝返回预测",
        )

    # ── 3. 生成预测（mock 模式）──────────────────────────────────────
    predictions: dict[str, list[float]] = {}
    for rid in inp.region_ids:
        predictions[str(rid)] = [
            _mock_speed(rid, inp.time_slice, step)
            for step in range(1, inp.horizon_steps + 1)
        ]

    # ── 4. 组装输出：预测值 + 质量口径 + provenance ──────────────────
    mae = manifest["experiments"]["speed_prediction"]["mae"]
    data = {
        "predictions": predictions,
        "time_slice": inp.time_slice,
        "horizon_steps": inp.horizon_steps,
        "source": "mock",            # 论文接入后为 cache | model
        "model_version": MOCK_VERSION,
        "quality_reference": {
            "mae_test_mean": mae["value"],
            "mae_std": mae["std"],
            "unit": mae["unit"],
            "aggregation": mae["aggregation"],
            "note": "测试集统计均值，严禁当作单次预测的置信度",
            "status": mae["status"],
        },
        "provenance": {
            "tool": "predict_traffic_speed",
            "implementation": "mock",
            "manifest_version": manifest.get("manifest_version"),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    }
    return ToolResult.success(data)

# 考点 1：Function Calling 中，如何防止大模型胡乱编造参数（比如传超界的值）？
#
#     满分回答：
#
#         “不能仅仅依赖 Prompt 的软约束（如 description），必须使用结构化 Schema 与强类型校验（如 Pydantic 的 Field 和 Annotated 边界约束）。在工具入口处进行 model_validate，一旦 LLM 越界，立刻通过 ToolResult.fail(ErrorCode.INVALID_PARAM) 把结构化错误带回给 LLM，让它在下一轮 Agent Loop 中自主纠正参数。”
#
# 考点 2：大模型容易对预测类数据产生“盲目自信（过度解读置信度）”，在架构设计上如何治理？
#
#     满分回答：
#
#         “我们引入了 Data-Provenance（数据血统）与 Quality Reference（质量口径绑定）机制。在工具返回的数据结构中，不仅带结果，还强制附带模型的全局指标（如测试集 MAE 均值）以及严格的防幻觉提示说明。这迫使大模型在生成回复时必须遵循客观统计口径，不能把宏观误差当成单次预测的实时置信度。”
#
# 考点 3：在多模态或复杂底层模型（如交通图神经网络 STGCN）尚未开发完成时，如何进行 Agent 的联调与测试？
#
#     满分回答：
#
#         “采用 Deterministic Mock（确定性模拟） 策略。通过哈希算法（如 sha256）基于输入参数生成固定的伪数据，做到‘同输入必得同输出’。这保证了在上下游系统缺失时，Agent 的集成测试具有完全的可复现性，且未来切换真实模型时只需替换底层加载函数，契约零修改。”

#
# 痛点：大模型（LLM）天生没有“真实世界的时间和空间感”。它看到一个数字（比如 45.2），它只觉得是个数字，根本不知道这个数字是从哪来的、是用旧模型算出来的还是临时用 mock 随机捏造的。
#
# Provenance 的作用：当 Provenance 数据和结果一起喂给大模型时，等于是给了大模型一个“锚点”。大模型在组织语言回答用户时，能够感知到：“哦，这个数据是来自 mock 版本的 STGCN 模型，基于 manifest v1 算出来的”。这极大地减少了它盲目向用户吹嘘“这是绝对实时精准预测”的幻觉。