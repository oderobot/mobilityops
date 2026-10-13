"""tools/optimize_rebalancing.py —— Tool 3：空驶调度优化（贪心版，SCIP 待论文接入）

干什么：给定车队规模与时间片，求解空驶调度方案（从盈余区调车到缺口区）。
管道：代码层调用 Tool 2 拿有效需求（冻结决策 §0.3-3）→ 贪心求解 → 评估预期 OSR。
为什么贪心先行：零依赖、毫秒级、逻辑可讲清；接口预留 SCIP 替换点
    （TODO 处），论文 OR-Tools 代码到达后只换内部求解器，签名不动。
面试点：solver_status 四分类（OPTIMAL/FEASIBLE/INFEASIBLE/TIMEOUT）——
    贪心版恒为 FEASIBLE 并显式声明"启发式无最优性保证"；INFEASIBLE 不重试应降级。
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field, ValidationError

from .base import ErrorCode, ToolResult
from .estimate_effective_demand import estimate_effective_demand
from .predict_traffic_speed import N_REGIONS, _format_validation_error
from .simulation import greedy_rebalance, init_supply, service_rate


class OptimizeRebalancingInput(BaseModel):
    fleet_size: int = Field(
        ge=1, le=1000,
        description="车队规模（辆）。严禁 ≤0 或 >1000：超出范围没有业务意义。",
    )
    time_slice: int = Field(default=0, ge=0, description="时间片索引")
    seed: int = Field(default=42, description="初始车辆分布种子（确定性复现用）")
    time_limit_sec: int = Field(
        default=60, ge=1,
        description="求解时间预算（秒）。贪心版瞬时完成，该参数为 MIP 接入预留。",
    )


def optimize_rebalancing(params: dict) -> ToolResult:
    """工具入口：dict 进，ToolResult 出，永不抛异常。"""
    start = time.perf_counter()

    # ── 1. 参数校验 ──────────────────────────────────────────────────
    try:
        inp = OptimizeRebalancingInput.model_validate(params)
    except ValidationError as exc:
        return ToolResult.fail(ErrorCode.INVALID_PARAM, _format_validation_error(exc))

    # ── 2. 代码层调用 Tool 2 获取有效需求；上游错误原样透传 ──────────
    demand_result = estimate_effective_demand({
        "region_ids": list(range(N_REGIONS)), "time_slice": inp.time_slice,
    })
    if not demand_result.ok:
        return demand_result  # 透传，保留 code 与 retryable
    demand = [demand_result.data["effective_demand"][str(i)] for i in range(N_REGIONS)]

    # ── 3. 贪心求解（TODO(论文接入)：替换为 CongestionAwareIPOptimizer/SCIP，
    #        并将 solver_status 改为求解器真实状态）─────────────────────
    supply = init_supply(inp.fleet_size, inp.seed)
    new_supply, plan, moved, dist = greedy_rebalance(supply, demand)
    osr = service_rate([float(x) for x in new_supply], demand)

    # 结构性不可行判定：有缺口但无车可调（本贪心版仅理论触发，MIP 版为真实路径）
    if sum(demand) > 1e-6 and inp.fleet_size == 0:  # fleet_size≥1 已拦住，防御性保留
        return ToolResult.fail(
            ErrorCode.SOLVER_INFEASIBLE, "需求存在但可用车辆为 0，结构性不可行，建议降级跑基线"
        )

    plan_summary = [
        {"from": i, "to": j, "vehicles": m} for i, j, m in sorted(plan, key=lambda x: -x[2])[:10]
    ]
    return ToolResult.success({
        "plan_summary": plan_summary,          # Top-10 调度动作（完整计划留 trace 层）
        "total_moves": len(plan),
        "total_moved_vehicles": moved,
        "reposition_distance": round(dist, 2),
        "expected_osr": round(osr, 4),
        "solver_status": "FEASIBLE",           # 贪心恒可行；MIP 接入后为真实四分类
        "solver": "greedy-nearest（启发式，无最优性保证）",
        "solver_gap": None,                    # 启发式无对偶间隙
        "fleet_size": inp.fleet_size,
        "time_slice": inp.time_slice,
        "provenance": {
            "tool": "optimize_rebalancing",
            "implementation": "greedy（TODO(论文接入): SCIP）",
            "upstream": demand_result.data["provenance"],
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    })



# 考点 1：当 Agent 需要处理复杂的数学优化任务（如车辆调度、排班、背包问题）时，你是让大模型来写代码求解，还是调用后端优化求解器？如何设计两者的边界？
#
#     满分回答：
#
#         “绝对不让大模型直接参与矩阵运算或求解。大模型只负责意图识别和参数配置（例如决定车队规模 fleet_size、选择优化目标）。真正的求解必须下沉到确定的后端工具中。我们在设计工具时，会采用‘接口标准化’策略：即便现阶段是用轻量级的贪心算法（Heuristic）实现，我们也必须在返回结构中显式暴露出高阶求解器的元数据（如 solver_status、solver_gap）。这不仅保证了当前阶段的稳定交付，也为后续无缝热插拔企业级求解器（如 SCIP / Gurobi）留出了标准的架构通道。”
#
# 考点 2：如果后端优化求解器返回了 INFEASIBLE（无解）或 TIMEOUT（超时），Agent 的上层状态机应该如何处理？
#
#     满分回答：
#
#         “绝不进行盲目的自动重试。因为 INFEASIBLE 属于业务或约束层面的结构性无解。编排层的状态机在捕获到这个结构化错误后，应当触发降级策略（Graceful Degradation）：例如自动切换至启发式基线方案，或者由 Agent 归纳当前的约束冲突，生成一份‘为什么无解’的诊断报告反馈给人类操作员，而不是让 Agent 陷入无限死循环。”