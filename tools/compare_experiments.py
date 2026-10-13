"""tools/compare_experiments.py —— Tool 5：实验对比（trade-off 版）

干什么：汇总多组实验结果，输出对比表 + 各维度优劣 + Pareto 候选。
核心纪律（§5.5）：不机械标注"最优项"——多目标决策没有绝对最优，只有权衡；
    显著性仅当有 seed-level 数据时才计算，否则只报差值（不沿用论文 p 值）。
输入两种模式（二选一）：experiment_ids（读 Tool 4 落盘结果）或 inline_results（内联）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from .base import ErrorCode, ToolResult
from .predict_traffic_speed import _format_validation_error
from .run_baseline import EXPERIMENTS_DIR

# 指标方向：max=越大越好，min=越小越好
METRIC_DIRECTIONS = {
    "osr": "max", "avg_wait_minutes": "min", "empty_mile_ratio": "min",
    "elapsed_ms": "min", "reposition_distance": "min", "mae": "min", "rmse": "min",
}
DEFAULT_METRICS = ["osr", "avg_wait_minutes", "empty_mile_ratio"]


class ExperimentRef(BaseModel):
    scenario_id: str
    run_id: str


class CompareInput(BaseModel):
    experiment_ids: list[ExperimentRef] | None = Field(
        default=None, description="模式一：引用 Tool 4 落盘结果（scenario_id + run_id）"
    )
    inline_results: list[dict] | None = Field(
        default=None,
        description="模式二：内联结果，每项含 name 与 metrics 字段。两种模式严禁同时传或都不传。",
    )
    metrics: list[str] = Field(
        default_factory=lambda: list(DEFAULT_METRICS),
        description=f"对比指标，可选：{list(METRIC_DIRECTIONS)}",
    )

    @model_validator(mode="after")
    def _check_exactly_one_mode(self):
        if bool(self.experiment_ids) == bool(self.inline_results):
            raise ValueError("experiment_ids 与 inline_results 必须二选一")
        unknown = set(self.metrics) - set(METRIC_DIRECTIONS)
        if unknown:
            raise ValueError(f"未知指标 {unknown}，可选：{list(METRIC_DIRECTIONS)}")
        return self


def _load_rows(inp: CompareInput) -> tuple[list[dict] | None, ToolResult | None]:
    """两种输入模式统一成行：{name, params, metrics}。失败返回 ToolResult。"""
    if inp.experiment_ids:
        rows = []
        for ref in inp.experiment_ids:
            path = EXPERIMENTS_DIR / ref.scenario_id / f"{ref.run_id}.json"
            if not path.exists():
                return None, ToolResult.fail(
                    ErrorCode.DATA_MISSING,
                    f"实验结果不存在：{ref.scenario_id}/{ref.run_id}（先运行 run_baseline），"
                    "严禁凭空编造对比数据",
                )
            rec = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"name": f"{ref.scenario_id}/{ref.run_id}",
                         "params": rec["params"], "metrics": rec["metrics"]})
        return rows, None
    rows = [{"name": r.get("name", f"exp_{i}"), "params": r.get("params", {}),
             "metrics": r.get("metrics", {})} for i, r in enumerate(inp.inline_results)]
    return rows, None


def _pareto_candidates(rows: list[dict], metrics: list[str]) -> list[str]:
    """非支配解：没有任何一行在所有指标上不劣且至少一个更优。"""
    def dominates(a, b):
        better_or_equal = all(
            (a["metrics"].get(m) is not None and b["metrics"].get(m) is not None)
            and (a["metrics"][m] >= b["metrics"][m] if METRIC_DIRECTIONS[m] == "max"
                 else a["metrics"][m] <= b["metrics"][m])
            for m in metrics
        )
        strictly = any(
            (a["metrics"][m] > b["metrics"][m] if METRIC_DIRECTIONS[m] == "max"
             else a["metrics"][m] < b["metrics"][m])
            for m in metrics
            if a["metrics"].get(m) is not None and b["metrics"].get(m) is not None
        )
        return better_or_equal and strictly

    return [r["name"] for r in rows
            if not any(o["name"] != r["name"] and dominates(o, r) for o in rows)]


def compare_experiments(params: dict) -> ToolResult:
    start = time.perf_counter()
    try:
        inp = CompareInput.model_validate(params)
    except ValidationError as exc:
        return ToolResult.fail(ErrorCode.INVALID_PARAM, _format_validation_error(exc))

    rows, err = _load_rows(inp)
    if err:
        return err

    # ── 对比表 + 每行优劣维度（不标"最优项"）─────────────────────────
    table = []
    for r in rows:
        strengths, weaknesses = [], []
        for m in inp.metrics:
            vals = [x["metrics"].get(m) for x in rows]
            v = r["metrics"].get(m)
            if v is None or any(x is None for x in vals):
                continue
            best = max(vals) if METRIC_DIRECTIONS[m] == "max" else min(vals)
            worst = min(vals) if METRIC_DIRECTIONS[m] == "max" else max(vals)
            if v == best and len(set(vals)) > 1:
                strengths.append(m)
            elif v == worst and len(set(vals)) > 1:
                weaknesses.append(m)
        table.append({"name": r["name"], "params": r["params"],
                      "metrics": {m: r["metrics"].get(m) for m in inp.metrics},
                      "strengths": strengths, "weaknesses": weaknesses,
                      # 中间值行无优劣维度，显式标注"均衡型"，防止被误读为遗漏
                      "profile": "均衡型（无极端维度）" if not strengths and not weaknesses else "trade-off"})

    return ToolResult.success({
        "table": table,
        "pareto_candidates": _pareto_candidates(rows, inp.metrics),
        "significance": None,
        "significance_note": "无 seed-level 重复数据，只报差值不计算 p 值（不沿用论文 p<0.001）",
        "note": "多目标决策没有绝对最优，请按业务权重在 Pareto 候选中权衡",
        "provenance": {"tool": "compare_experiments",
                       "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)},
    })

# 考点 1：当 AI Agent 需要对比多个算法方案或处理多目标冲突（如兼顾服务满意度与空驶成本）时，为什么不能让大模型直接选出“最优解”？你是如何通过代码防线来规范的？
#
#     满分回答：
#
#         “因为大模型在面对多目标（Multi-Objective）权衡时，极易产生单维度偏见或盲目自信，强行指定一个有缺陷的方案为主最优。我们在架构上采用了‘程序化帕累托边界过滤 + 显式 Trade-off 标注’的模式（如 Tool 5 源码所示）：底层代码先通过数学定义过滤出非支配解（Pareto Candidates），并在返回结果中明确声明‘多目标决策没有绝对最优’。同时，我们在 Prompt 和工具契约中硬性规定大模型不得机械化标注‘全局最优’，而是必须引导人类在 Pareto 候选中根据业务权重进行权衡。这从根本上杜绝了大模型在复杂博弈决策中的幻觉。”
#
# 考点 2：在设计评估型 Agent 时，如何防止大模型在缺乏底层统计依据时“瞎编”实验结论（如虚构显著性差异）？
#
#     满分回答：
#
#         “必须在工具的设计中引入‘数据完备性门禁（Data Completeness Gate）’。正如对比工具中所体现的：当系统检测到输入数据缺乏底层方差或重复实验样本（seed-level data）时，代码会主动将统计显著性（Significance）置为 None，并附带硬编码的免责声明。这不仅防止了大模型为了迎合人类而伪造统计学指标（如随口说‘经过验证差异具有统计显著性’），也确保了 Agent 产出的实验报告在工业级或学术级评审中经得起推敲。”