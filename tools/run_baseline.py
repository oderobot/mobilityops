"""tools/run_baseline.py —— Tool 4：基线策略执行

干什么：运行 4 种基线调度策略（no_dispatch / random / nearest / demand_only），
    输出评估指标并落盘，为 Tool 3（优化）和 Tool 5（对比）提供对照组。
为什么需要基线（面试点）：没有对照组的优化数字无意义。
口径：
    - 需求来自代码层调用 Tool 2（有效需求，全链路口径一致，冻结决策 §0.3-3 同款）；
    - demand_only 按论文逻辑用均匀基准需求（0.5）决策、用有效需求评估；
    - avg_wait_minutes / empty_mile_ratio 论文未计算，为 mock 派生代理指标
    - （公式见输出 note，TODO(论文接入)）；
    - 结果落盘 data/experiments/{scenario_id}/{run_id}.json 供 Tool 5 复现；
      同 run_id 不同参数 → INVALID_PARAM（防污染对比结果池）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .base import ErrorCode, ToolResult
from .estimate_effective_demand import DEFAULT_BASE_DEMAND, estimate_effective_demand
from .predict_traffic_speed import N_REGIONS, _format_validation_error
from .simulation import greedy_rebalance, init_supply, random_rebalance, service_rate

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "data" / "experiments"
TIME_GRANULARITY_MIN = 15   # ✅ verified：dyna 时间戳 900s
AVG_TRIP_LEN_HOPS = 5.0     # mock 派生常数：估计服务里程用（TODO(论文接入)）

STRATEGIES = ("no_dispatch", "random", "nearest", "demand_only")


class RunBaselineInput(BaseModel):
    strategy: Literal["no_dispatch", "random", "nearest", "demand_only"] = Field(
        description=(
            "基线策略。严禁传入 congestion_aware / ca_mip——那是优化工具 "
            "optimize_rebalancing 的职责，本工具只跑基线。"
        ),
    )
    fleet_size: int = Field(ge=1, le=1000, description="车队规模，严禁 ≤0 或 >1000")
    time_slice: int = Field(default=0, ge=0, description="时间片索引")
    scenario_id: str = Field(min_length=1, description="实验分组标识（必填）")
    run_id: str | None = Field(
        default=None,
        description="运行标识；缺省时由参数哈希确定性生成。严禁同 run_id 复用于不同参数。",
    )
    seed: int = Field(default=42, description="随机性唯一来源（仅 random 策略消费）")


def _make_run_id(inp: RunBaselineInput) -> str:
    """同参数必得同 run_id（可复现性根基，测试不变量 #4）。"""
    key = f"{inp.scenario_id}|{inp.strategy}|{inp.fleet_size}|{inp.time_slice}|{inp.seed}"
    return "run_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _params_fingerprint(inp: RunBaselineInput) -> dict:
    return {"strategy": inp.strategy, "fleet_size": inp.fleet_size,
            "time_slice": inp.time_slice, "seed": inp.seed}


def run_baseline(params: dict) -> ToolResult:
    start = time.perf_counter()

    # ── 1. 参数校验 ──────────────────────────────────────────────────
    try:
        inp = RunBaselineInput.model_validate(params)
    except ValidationError as exc:
        return ToolResult.fail(ErrorCode.INVALID_PARAM, _format_validation_error(exc))
    run_id = inp.run_id or _make_run_id(inp)

    # ── 2. run_id 冲突检查：同 id 不同参数 → 拒绝（防污染对比结果池）──
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / inp.scenario_id / f"{run_id}.json"
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ToolResult.fail(ErrorCode.DATA_MISSING, f"结果文件损坏：{path}")
        if old.get("params") != _params_fingerprint(inp):
            return ToolResult.fail(
                ErrorCode.INVALID_PARAM,
                f"run_id={run_id} 已存在且参数不一致（{old.get('params')}），"
                "严禁复用 run_id 覆盖不同实验，请更换 run_id",
            )

    # ── 3. 代码层调用 Tool 2 获取评估需求；上游错误原样透传 ──────────
    demand_result = estimate_effective_demand({
        "region_ids": list(range(N_REGIONS)), "time_slice": inp.time_slice,
    })
    if not demand_result.ok:
        return demand_result
    demand = [demand_result.data["effective_demand"][str(i)] for i in range(N_REGIONS)]

    # ── 4. 执行策略（论文 DispatchStrategies 对应实现）────────────────
    supply = init_supply(inp.fleet_size, inp.seed)
    if inp.strategy == "no_dispatch":
        after, plan, moved, dist = list(supply), [], 0, 0.0
    elif inp.strategy == "random":
        after, plan, moved, dist = random_rebalance(supply, demand, inp.seed)
    elif inp.strategy == "nearest":
        after, plan, moved, dist = greedy_rebalance(supply, demand)
    else:  # demand_only：按均匀基准需求决策（论文口径），用有效需求评估
        uniform = [DEFAULT_BASE_DEMAND] * N_REGIONS
        after, plan, moved, dist = greedy_rebalance(supply, uniform)

    # ── 5. 指标计算（OSR 为论文口径；等待/空驶占比为 mock 派生）──────
    osr = service_rate([float(x) for x in after], demand)
    served = sum(min(float(a), d) for a, d in zip(after, demand))
    avg_wait = round(TIME_GRANULARITY_MIN * (1 - osr), 2)          # 派生：等待∝未满足比例
    service_miles = served * AVG_TRIP_LEN_HOPS
    empty_ratio = round(dist / (dist + service_miles), 4) if dist + service_miles > 0 else 0.0

    record = {
        "tool": "run_baseline",
        "scenario_id": inp.scenario_id,
        "run_id": run_id,
        "params": _params_fingerprint(inp),
        "metrics": {
            "osr": round(osr, 4),
            "avg_wait_minutes": avg_wait,
            "empty_mile_ratio": empty_ratio,
            "total_moved": moved,
            "reposition_distance": round(dist, 2),
        },
        "metrics_note": "osr 为论文口径；avg_wait=15min×(1-OSR) 与 empty_mile_ratio 为 mock 派生代理，TODO(论文接入)",
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        "provenance": {"implementation": "mock",
                       "upstream": demand_result.data["provenance"]},
    }

    # ── 6. 落盘（供 Tool 5 复现）─────────────────────────────────────
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return ToolResult.success({
        "metrics": record["metrics"],
        "strategy": inp.strategy,
        "fleet_size": inp.fleet_size,
        "time_slice": inp.time_slice,
        "scenario_id": inp.scenario_id,
        "run_id": run_id,
        "result_ref": str(path),
        "metrics_note": record["metrics_note"],
        "provenance": record["provenance"],
    })

#
# 考点 1：在设计评估型或决策型 Agent 时，如何防止大模型在反复迭代中产生“幻觉数据污染（Data Hallucination / Overwriting）”？
#
#     满分回答：
#
#         “必须在底层工具中引入内容寻址与参数指纹校验机制。正如基线工具（Tool 4）所设计的：每一个实验运行不仅依赖显式的 scenario_id 和 run_id，系统还会通过参数哈希生成不可篡改的指纹。当检测到相同的 run_id 但输入参数发生漂移时，系统会主动拦截并报错（INVALID_PARAM），绝不允许静默覆盖旧文件。这确保了多轮对话和复杂 Agent 工作流中的所有实验产物（Artifacts）具有 100% 的可追溯性与绝对隔离性。”
#
# 考点 2：在多 Agent 协同系统中，不同工具之间需要共享历史实验数据或评测指标时，应当采用哪种架构模式？
#
#     满分回答：
#
#         “采用基于文件系统或对象存储的 Artifact 契约模式。上游工具（如 Tool 4）不仅在内存中返回精简的摘要给大模型，更将包含完整元数据、派生指标和血统追踪（Provenance）的结构化 JSON 规范落盘，并向编排层返回受控的文件引用路径（result_ref）。下游工具（如 Tool 5 的对比工具）通过引用直接安全读取。这种‘轻量摘要上行，大块数据落盘’的设计，既避免了塞满大模型的 Context Window（上下文窗口），又保障了多工具之间数据交换的强类型与可靠性。”

# 二、 什么是“参数指纹（Parameter Fingerprint）”？
# 1. 概念通俗解：
#
# “参数指纹”就是给一组输入参数拍一张唯一的“身份证照”（哈希值）。
# 在 run_baseline.py 里，我们把 scenario_id、strategy、fleet_size、time_slice 和 seed 拼成一个字符串，然后用 SHA-256 砸碎成一段短哈希（比如 run_a1b2c3d4）。只要这堆参数里有任何一个数字变了（哪怕 fleet_size 从 200 变成 201），指纹就会彻底改变。
# 2. 在后端和 MLOps 里的祖宗：
#
#     缓存中的 Cache Key / Git Commit Hash：这和 Java 里的 hashCode()、Redis 的缓存键生成、或者是 Git 用 SHA-1 标记代码版本是同一个逻辑。
#
#     MLflow / DVC 的实验追踪：在算法工程中，每次训练模型前，都要对超参数（Hyperparameters）打指纹。如果参数没变，直接读缓存结果，绝不重复跑计算。
#
# 3. 为什么 Agent 需要它？
#
# 防止大模型“张冠李戴”。当 Agent 在多轮对话中要求重新跑实验时，如果大模型偷偷改了参数却复用了旧的实验 ID（run_id），参数指纹校验就会立刻报警，拒绝覆盖，从而保护实验数据的绝对纯洁性。
# 三、 什么是“Artifact 契约模式（Artifact Contract Pattern）”？
# 1. 概念通俗解：
#说白了就是“别让大模型在对话里瞎传大文件，让程序乖乖把结果写成 JSON 文件存好”。
#     Artifact（制品 / 产物）：程序运行后吐出来的沉淀文件，比如一个结构化的 JSON 报告、一张图表、一个训练好的模型文件（在代码里就是落盘到 data/experiments/... 的 JSON 文件）。
#
#     Contract（契约）：上游和下游之间关于这个文件的“强类型格式约定”（比如必须包含 metrics.osr 字段，类型必须是 float）。
#
# 2. 为什么不用大模型直接在对话里传数据？（痛点）
#
# 如果没有 Artifact 契约，传统的 Agent 喜欢让大模型把所有计算出的密密麻麻的数字（156个区域的车流、矩阵、坐标）全塞进对话的上下文（Context Window）里，让大模型口头传给下一个工具。
#
#     灾难后果：大模型既贵、又容易看漏、还会产生严重的数字幻觉（Floating-point hallucination），而且会瞬间把 Token 窗口撑爆。
#
# 3. 契约模式是怎么解的？
#
#     轻量摘要上行：大模型和上层编排层只拿到一个极简的引用凭证（如 result_ref: "data/experiments/scen_1/run_abc.json"）。
#
#     重度数据落盘：完整、庞大的计算结果被严格按照 Pydantic 契约规范写入磁盘或对象存储（Artifact Store）。
#
#     安全交接：下游工具（如评测对比工具 Tool 5）通过路径契约直接去读这个 JSON 文件。
#
# 4. 在后端和大数据里的祖宗：
#
# 这完全就是 RESTful API 的 OpenAPI/Swagger 契约 加上 AWS S3 / 传统文件服务器的对象存储规范。微服务之间从不把几个 G 的大对象直接在 HTTP 报文里瞎传，而是传一个文件路径或对象存储的 URL（Pre-signed URL），双方基于强类型的 DTO（Data Transfer Object）契约进行通信。
# 💡 核心总结
#
# 你看到的这些高级词汇：
#
#     参数指纹 = 用哈希锁死输入，防篡改、防覆盖（分布式缓存与版本控制思想）。
#
#     Artifact 契约模式 = 大块数据乖乖落盘、通过强类型契约交互，绝不让大模型在对话里瞎传大对象（微服务与数据平台架构思想）。
#
# 它们全部来自于成熟的后端分布式系统和 MLOps。 现代优秀 Agent 的工程落地，拼到最后其实就是把传统软件工程里最严苛的“防御性编程”和“数据治理”原封不动地搬回来。