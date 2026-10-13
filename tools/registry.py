"""tools/registry.py —— 工具注册中心：工具层与编排层之间的"窄腰"

设计定位（面试考点）：
1. 注册即声明：每个工具 = 一条 ToolSpec（名字/入口/Input 模型/描述）。
   新增工具只加一条 Spec，编排层（raw_loop / LangGraph）零改动。
2. Schema 同源生成：给 LLM 的 JSON Schema 由 Input 模型 model_json_schema()
   自动生成，全项目不存在第二份手写参数描述 —— 约束与描述物理上不可能不一致
   （这是 Tool 1 "description 不是约束" 教训的正面解）。
3. 三层防线收敛到 execute()：白名单 → JSON 解析 → 工具内 Pydantic 校验，
   任何失败都是结构化 ToolResult，永不抛异常。
4. to_observation() 把 ToolResult 序列化成喂给 LLM 的文本：
   失败时携带 error_code + retryable，LLM 可据此决定重试还是换方案。

对应 raw_loop.py 的升级关系：
    raw_loop 的 TOOL_REGISTRY + TOOLS_DEFINITION + _execute_tool 三处，
    在本模块统一收口；编排层只调用本模块的三个公开函数。
"""

import json
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from tools.base import ErrorCode, ToolResult
from tools.compare_experiments import CompareInput, compare_experiments
from tools.estimate_effective_demand import EstimateDemandInput, estimate_effective_demand
from tools.optimize_rebalancing import OptimizeRebalancingInput, optimize_rebalancing
from tools.predict_traffic_speed import PredictTrafficSpeedInput, predict_traffic_speed
from tools.run_baseline import RunBaselineInput, run_baseline
from tools.search_project_docs import SearchDocsInput, search_project_docs


# ─────────────────────────────────────────────
# 1. 注册表：工具 = 声明式 Spec
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class ToolSpec:
    """一个工具的全部注册信息。frozen：注册表运行期不可变。"""
    name: str
    entry: Callable[[dict], ToolResult]  # 工具入口契约：只收 dict、只收 ToolResult、永不抛异常
    input_model: type[BaseModel]          # Schema 同源生成的唯一来源 调用工具时输入必须是该模型的子类实例
    description: str                      # 给 LLM 看的能力描述（含负向约束）


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="predict_traffic_speed",
        entry=predict_traffic_speed,
        input_model=PredictTrafficSpeedInput,
        description=(
            "预测指定区域未来若干时间片的交通车速（km/h）。"
            "返回附带模型质量参考（测试集 MAE），该值为统计均值，严禁当作单次预测的置信度。"
            "不要用于区域数超过 156 或预测步长超过 12 的请求。"
        ),
    ),
    ToolSpec(
        name="estimate_effective_demand",
        entry=estimate_effective_demand,
        input_model=EstimateDemandInput,
        description=(
            "估计考虑道路通行能力折减后的有效需求：原始需求 / 通行能力比 ρ(v)。"
            "内部会调用 predict_traffic_speed 获取车速。不要直接传入你自己的车速估计。"
        ),
    ),
    ToolSpec(
        name="optimize_rebalancing",
        entry=optimize_rebalancing,
        input_model=OptimizeRebalancingInput,
        description=(
            "在给定车队规模下做区域间车辆再平衡调度（贪心最近邻启发式，无最优性保证）。"
            "内部会调用 estimate_effective_demand。不要用于车队规模超过 1000 的场景。"
        ),
    ),
    ToolSpec(
        name="run_baseline",
        entry=run_baseline,
        input_model=RunBaselineInput,
        description=(
            "运行基线调度策略（no_dispatch/random/nearest/demand_only）并返回 OSR 等对比指标。"
            "congestion_aware 不是合法策略名；论文数值请从实验对比中引用。"
        ),
    ),
    ToolSpec(
        name="compare_experiments",
        entry=compare_experiments,
        input_model=CompareInput,
        description=(
            "对比多个实验的指标，给出优势/短板画像与 Pareto 候选集。"
            "二选一输入：experiment_ids（查持久化记录）或 inline_results（内联数据），不能同时给。"
        ),
    ),
    ToolSpec(
        name="search_project_docs",
        entry=search_project_docs,
        input_model=SearchDocsInput,
        description=(
            "在项目文档语料中做混合检索（BM25 + 向量 + RRF 融合）。"
            "用于回答论文口径、实验设置、指标定义类问题；查不到会明确说'文档中未找到'，不要强行编造。"
        ),
    ),
)
#相当于生成了一个字典，字典key为工具名，value为ToolSpec对象 此乃全局注册表，编排层只需调用 execute() 即可，不必关心具体工具实现
_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}

# 注册表完整性自检（模块加载时执行，防拼写错误导致名称与函数不匹配）
for _spec in TOOL_SPECS:
    assert _spec.entry.__name__ == _spec.name, f"Spec 名称与入口函数不一致: {_spec.name}"


# ─────────────────────────────────────────────
# 2. Schema 同源生成：Pydantic 模型 → OpenAI function definition
# ─────────────────────────────────────────────

def get_openai_tools() -> list[dict]:
    """生成给 LLM 的工具描述列表（OpenAI function calling 格式）。

    parameters 直接取自 Input 模型的 JSON Schema —— 改 Pydantic 字段，
    LLM 看到的 schema 自动同步，不存在"描述与校验脱节"的可能。
    """
    tools = []
    for spec in TOOL_SPECS:
        schema = spec.input_model.model_json_schema()
        tools.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                    # 嵌套模型的 $defs 一并带上（如 CompareInput 内的 ExperimentRef）
                    **({"$defs": schema["$defs"]} if "$defs" in schema else {}),
                },
            },
        })
    return tools


def list_tool_names() -> list[str]:
    """可用工具名列表（白名单全集）。"""
    return list(_REGISTRY)


# ─────────────────────────────────────────────
# 3. execute：三层防线统一执行入口
# ─────────────────────────────────────────────

def execute(name: str, arguments: str | dict) -> ToolResult:
    """执行工具的唯一入口。任何失败都返回结构化 ToolResult，永不抛异常。

    防线一：白名单校验函数名（LLM 幻觉调用的兜底）。
    防线二：JSON 参数解析（模型吐非法 JSON 是常态 → LLM_INVALID_OUTPUT）。
    防线三：业务执行异常兜底（理论上工具入口已自校验，这里是双保险 → INTERNAL）。
    参数本身的合法性（值域/类型）由工具内部的 Pydantic 模型把关 → INVALID_PARAM。
    """
    # ── 防线一：白名单 ──
    spec = _REGISTRY.get(name)
    if spec is None:
        return ToolResult.fail(
            ErrorCode.INVALID_PARAM,
            f"工具 '{name}' 不存在。可用工具：{', '.join(list_tool_names())}",
        )

    # ── 防线二：JSON 解析（raw_loop 修正点 1 的正式版）──
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return ToolResult.fail(
                ErrorCode.LLM_INVALID_OUTPUT,
                f"工具参数不是合法 JSON（{exc}），请修正参数格式后重新调用。",
            )
    if not isinstance(arguments, dict):
        return ToolResult.fail(
            ErrorCode.INVALID_PARAM,
            f"工具参数必须是 JSON 对象，实际收到 {type(arguments).__name__}。",
        )

    # ── 防线三：执行兜底（工具内部契约是永不抛异常，这里防契约被破坏）──
    try:
        return spec.entry(arguments)
    except Exception as exc:  # noqa: BLE001 —— 窄腰处必须兜住一切
        return ToolResult.fail(
            ErrorCode.INTERNAL,
            f"工具 '{name}' 执行时发生未预期异常：{exc}",
        )


# ─────────────────────────────────────────────
# 4. to_observation：ToolResult → 喂给 LLM 的观测文本
# ─────────────────────────────────────────────

def to_observation(result: ToolResult) -> str:
    """把 ToolResult 序列化成写回 LLM 上下文的观测文本。

    成功：直接给 data 的 JSON（LLM 只应消费成功结果的数据）。
    失败：给结构化的错误描述，显式携带 error_code 和 retryable ——
        retryable 由 ErrorCode 集中派生（冻结决策），LLM 看到 true 可重试，
        看到 false 应换方案或向用户报告，而不是盲目重复调用。
    """
    if result.ok:
        return json.dumps(result.data, ensure_ascii=False, default=str)

    err = result.error
    return (
        f"工具调用失败 [error_code={err.code.value}] "
        f"[retryable={str(err.retryable).lower()}] {err.message}"
    )
