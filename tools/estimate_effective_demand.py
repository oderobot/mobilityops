"""tools/estimate_effective_demand.py —— Tool 2：有效需求估计（mock 链路中的确定性管道）

干什么：
    把"原始需求"按速度相关的容量衰减折算为"有效需求"：
        d_eff = d_bar / max(rho(v), 0.01)
    物理含义：车速 v 越低（越拥堵），每辆车的服务能力 rho(v) 越差，
    服务同样的乘客就需要更多车——所以有效需求上升。

为什么这样设计：
    1. 代码层工具组合（冻结决策 §0.3-3）：本工具内部直接调用 Tool 1
       获取速度，确定性管道不经 LLM 编排；Tool 1 失败时错误原样透传，
       retryable 标记保留，由编排层决定重试/降级。
    2. rho(v) 分段函数逐行来自论文 CongestionAwareDemand
       （dispatch_thesis_v9_fast.py L56-71），不改一个系数。
    3. 速度取预测的第 1 步：与论文端到端实验口径一致
       （dispatch 只消费 prediction[:, 0, :]，见 manifest.dispatch）。

对应面试点：代码层组合 vs LLM 编排的取舍（确定性管道为何不经模型）；
            上游错误透传而非吞掉。
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field, ValidationError

from .base import ErrorCode, ToolResult
from .predict_traffic_speed import (
    N_REGIONS,
    PredictTrafficSpeedInput,
    _format_validation_error,
    predict_traffic_speed,
)

# 论文口径常量（dispatch_thesis_v9_fast.py：BASE_DEMAND / CongestionAwareDemand）
DEFAULT_BASE_DEMAND = 0.5   # 每区域每时间片的合成原始需求
RHO_FLOOR = 0.01            # rho 下限，防除零
FREE_FLOW_SPEED = 40.0      # v ≥ 40 km/h 视为畅通，rho = 1
MIN_CAPACITY_RATIO = 0.3    # 严重拥堵（v < 5）时每辆车只剩 30% 服务能力


class EstimateDemandInput(BaseModel):
    region_ids: list[int] = Field(min_length=1, description="区域索引列表（0–155），与 Tool 1 同口径")
    time_slice: int = Field(default=0, ge=0, description="时间片索引；速度取该时刻预测的第 1 步")
    raw_demand: dict[str, float] | None = Field(
        default=None,
        description=(
            "可选：{区域索引: 原始需求}。缺省时所有区域用论文合成基准 "
            f"{DEFAULT_BASE_DEMAND}。严禁传负值——需求为负没有物理意义。"
        ),
    )


def capacity_ratio(speed: float) -> float:
    """rho(v)：速度 → 单车服务能力比率。逐行对应论文 piecewise 五段。

    分段内单调不减，值域 [0.3, 1.0]。
    ⚠️ 论文实现的已知瑕疵（2026-09-30 不变量测试发现，经所有者确认保留原样）：
       v=20 处 ρ 从 0.6 向下跳到 0.5（第 3 段终点与第 4 段起点系数未对齐）；
       v=40 处从 0.7 向上跳到 1.0。面试口径：测试发现 → 记录 → 忠实复现。
    """
    v = max(0.0, float(speed))
    if v < 5:
        return MIN_CAPACITY_RATIO
    if v < 10:
        return MIN_CAPACITY_RATIO + 0.1 * (v - 5) / 5
    if v < 20:
        return 0.4 + 0.2 * (v - 10) / 10
    if v < FREE_FLOW_SPEED:
        return 0.5 + 0.2 * (v - 20) / 20
    return 1.0


def effective_demand(raw: float, speed: float) -> float:
    """d_eff = d_bar / max(rho(v), RHO_FLOOR)。纯函数，便于单测。"""
    return raw / max(capacity_ratio(speed), RHO_FLOOR)


def estimate_effective_demand(params: dict) -> ToolResult:
    """工具入口：dict 进，ToolResult 出，永不抛异常。"""
    start = time.perf_counter()

    # ── 1. 参数校验 ──────────────────────────────────────────────────
    try:
        inp = EstimateDemandInput.model_validate(params)
    except ValidationError as exc:
        return ToolResult.fail(ErrorCode.INVALID_PARAM, _format_validation_error(exc))

    # 需求为负：Schema 管不到 dict 内部值，业务层校验（负向约束的代码实现）
    if inp.raw_demand is not None:
        bad = {k: v for k, v in inp.raw_demand.items() if v < 0}
        if bad:
            return ToolResult.fail(
                ErrorCode.INVALID_PARAM, f"raw_demand 存在负值 {bad}，需求为负没有物理意义"
            )

    # ── 2. 代码层调用 Tool 1 获取速度（冻结决策 §0.3-3）───────────────
    #    只取第 1 步预测，与论文端到端实验口径一致
    speed_result = predict_traffic_speed({
        "region_ids": inp.region_ids, "horizon_steps": 1, "time_slice": inp.time_slice,
    })
    if not speed_result.ok:
        return speed_result  # 上游错误原样透传，保留 code 与 retryable

    # ── 3. 容量衰减折算，逐区域保留 rho 明细（供 Agent 解释因果链）────
    details: dict[str, dict] = {}
    eff: dict[str, float] = {}
    for rid in inp.region_ids:
        key = str(rid)
        v = speed_result.data["predictions"][key][0]
        raw = inp.raw_demand.get(key, DEFAULT_BASE_DEMAND) if inp.raw_demand else DEFAULT_BASE_DEMAND
        rho = capacity_ratio(v)
        eff[key] = round(raw / max(rho, RHO_FLOOR), 6)
        details[key] = {"speed_kmh": v, "rho": round(rho, 4), "raw_demand": raw}

    return ToolResult.success({
        "effective_demand": eff,
        "capacity_detail": details,      # speed → rho → d_eff 的完整链条
        "time_slice": inp.time_slice,
        "provenance": {
            "tool": "estimate_effective_demand",
            "formula": "d_eff = d_bar / max(rho(v), 0.01)（论文 CongestionAwareDemand）",
            "upstream": speed_result.data["provenance"],  # 溯源链不断
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    })

# 路径 A（编排层直接调 Tool 1）：
#   用户："预测 3 号区未来车速"
#   LLM → Tool 1 → ToolResult 直接回编排层
#
# 路径 B（Tool 2 内部代码层调 Tool 1）：
#   用户："估计一下各区域有效需求"
#   LLM → Tool 2 ──import 调用──> Tool 1 → ToolResult
#          ↑________ 内部管道，不过 LLM ________|
#   Tool 2 收到 Tool 1 的 ToolResult 后：
#     - ok=True  → 取出速度数据，继续算有效需求
#     - ok=False → 原样透传（整个 ToolResult 对象原封不动返回给编排层）

# 考点 1：在多步骤 Agent 工作流中，多个 Tool 之间需要串联调用（例如 Tool B 依赖 Tool 2 的输出），你是让 LLM 来编排（ReAct/Plan-and-Solve），还是在代码里硬编码？为什么？
#
#     满分回答：
#
#         “对于确定性的业务逻辑、数学计算和严密的流水线（Pipeline），我们采用代码层组合（Code-Level Composition），而不是交给大模型动态编排。正如我们在需求估计工具中直接调用速度预测工具一样：第一，避免了 LLM 进行复杂浮点计算和分段函数时的幻觉；第二，省去了多轮大模型推理的 Token 成本和延迟；第三，能完美实现上游错误和状态（如 retryable）的原样透传。大模型只应该保留在战略决策和意图识别层。”
#
# 考点 2：当底层工具链发生级联报错时（例如 Tool A 报错导致 Tool B 无法执行），在架构上如何优雅地处理错误？
#
#     满分回答：
#
#         “采用**错误原样透传（Upstream Error Propagation）**机制。上游工具在捕获到结构化的 ToolResult.fail() 后，绝不吞掉错误，也不盲目包装，而是直接将带有一致性错误码和 retryable 标记的 ToolResult 向上返回。这确保了编排层的状态机在收到报错时，能够准确无误地根据底层的错误属性来决定是触发自动重试、参数修正（Reflection）还是全局重规划（Replan）。”