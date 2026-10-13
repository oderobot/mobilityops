"""tools/simulation.py —— 调度仿真原语（Tool 3/4 共享）

干什么：初始车辆分布、mock 距离、服务率、贪心/随机调度的纯函数实现。
为什么单独抽出来：Tool 3（优化）和 Tool 4（基线）共用同一套仿真物理，
    否则两份实现漂移，对比实验就失去公平性（对照组大忌）。
口径来源：service_rate 与论文 DispatchStrategies.calculate_service_rate 一致
    min(s,d).sum()/d.sum()；贪心调度对应论文 nearest_dispatch 逻辑。
全部确定性：同输入必得同输出（mock 铁律），随机性只来自显式 seed。
TODO(论文接入)：mock_distance 替换为 sz_taxi_dist_matrix_weighted.npy 真实路网距离。
"""

from __future__ import annotations

import hashlib
import random

N_REGIONS = 156  # verified: SZ_TAXI.geo


def mock_distance(i: int, j: int) -> float:
    """确定性对称距离（跳数口径，1.0–21.0）。i==j 为 0。"""
    if i == j:
        return 0.0
    a, b = (i, j) if i < j else (j, i)  # 排序保证对称
    h = int.from_bytes(hashlib.sha256(f"dist|{a}|{b}".encode()).digest()[:8], "big")
    return 1.0 + (h % 2000) / 100.0


def init_supply(fleet_size: int, seed: int, n_regions: int = N_REGIONS) -> list[int]:
    """确定性初始分布：fleet_size 辆车按 seed 撒到各区域（对应论文 _init_supply）。"""
    rng = random.Random(f"supply|{seed}|{fleet_size}")
    supply = [0] * n_regions
    for _ in range(fleet_size):
        supply[rng.randrange(n_regions)] += 1
    return supply


def service_rate(supply: list[float], demand: list[float]) -> float:
    """OSR = Σmin(s,d)/Σd（论文口径）。需求为 0 时定义为 1.0。"""
    served = sum(min(s, d) for s, d in zip(supply, demand))
    total = sum(demand)
    return served / total if total > 1e-6 else 1.0


def greedy_rebalance(
    supply: list[int], demand: list[float], dist_fn=mock_distance
) -> tuple[list[int], list[tuple[int, int, int]], int, float]:
    """就近贪心调度：盈余区按盈余量降序，向最近缺口区调车（论文 nearest_dispatch 逻辑）。

    返回：调度后 supply、调度计划 [(i,j,m)]、总调车数、总空驶距离。
    不变量：车辆守恒（只搬不造）。
    """
    n = len(supply)
    s = [float(x) for x in supply]
    surplus = [max(s[i] - demand[i], 0.0) for i in range(n)]
    deficit = [max(demand[i] - s[i], 0.0) for i in range(n)]
    out = list(supply)
    plan: list[tuple[int, int, int]] = []
    moved, dist = 0, 0.0
    for i in sorted(range(n), key=lambda x: -surplus[x]):
        if surplus[i] < 0.01:
            continue
        # 候选缺口按距离升序
        cands = sorted((j for j in range(n) if deficit[j] > 0.01), key=lambda j: dist_fn(i, j))
        for j in cands:
            if surplus[i] < 0.01:
                break
            m = min(int(surplus[i]), int(-(-deficit[j] // 1)))  # ceil(deficit)
            if m <= 0:
                continue
            out[i] -= m
            out[j] += m
            surplus[i] -= m
            deficit[j] -= m
            moved += m
            dist += m * dist_fn(i, j)
            plan.append((i, j, m))
    return out, plan, moved, dist


def random_rebalance(
    supply: list[int], demand: list[float], seed: int, dist_fn=mock_distance
) -> tuple[list[int], list[tuple[int, int, int]], int, float]:
    """随机调度基线：盈余区随机配对缺口区（论文 random_dispatch 逻辑，seed 显式）。"""
    n = len(supply)
    rng = random.Random(f"rand|{seed}")
    surplus = [max(supply[i] - demand[i], 0.0) for i in range(n)]
    deficit = [max(demand[i] - supply[i], 0.0) for i in range(n)]
    out = list(supply)
    plan: list[tuple[int, int, int]] = []
    moved, dist = 0, 0.0
    s_idx = [i for i in range(n) if surplus[i] > 0.01]
    rng.shuffle(s_idx)
    d_idx = [j for j in range(n) if deficit[j] > 0.01]
    for i in s_idx:
        if surplus[i] < 0.01 or not d_idx:
            break
        j = rng.choice(d_idx)
        m = min(int(surplus[i]), int(-(-deficit[j] // 1)))
        if m <= 0:
            continue
        out[i] -= m
        out[j] += m
        surplus[i] -= m
        deficit[j] -= m
        moved += m
        dist += m * dist_fn(i, j)
        plan.append((i, j, m))
        if deficit[j] < 0.01:
            d_idx.remove(j)
    return out, plan, moved, dist


# 考点 1：在开发需要进行策略对比或优化的 Agent 时（例如评估两种不同的调度策略），如何保证 Agent 评估结果的公正性与可复现性？
#
#     满分回答：
#
#         “必须在底层构建一个确定性的仿真世界（Deterministic Simulation Environment）。我们将底层的距离计算、车辆初始化、服务满意度指标以及调度算法（如贪心与随机基线）剥离为共享的纯函数原语，并通过显式种子（Explicit Seed）和哈希确定性（如 sha256）锁死所有计算。这确保了对照组和实验组共享同一套物理定律，彻底消除了代码漂移，使得 Agent 的多轮策略评估具备 100% 的可复现性。”
#
# 考点 2：面对复杂的业务仿真和组合优化任务（如车队调度重平衡），你是如何划分大模型（LLM）与后端代码（Code）的职责边界的？
#
#     满分回答：
#
#         “遵循**‘确定性归代码，策略性归大模型’的黄金法则。像车辆守恒计算、贪心匹配、距离矩阵这类对数学精度、计算性能和逻辑严密性要求极高的任务，必须用 Python 纯函数在底层实现。大模型只负责高层的意图识别、参数配置（如决定调配规模、选择优化策略）以及对底层仿真输出的数据结果进行人类可读的业务解释**。绝不让大模型参与任何矩阵运算和组合优化。”