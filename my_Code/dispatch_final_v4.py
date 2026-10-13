"""
============================================================
📦 拥堵感知前置调度实验系统 —— 最终学术版 v4.0
============================================================
核心修复：
1. MIP约束从 floor 改为 ceil，允许整数车辆填补浮点缺口
2. 目标函数改为：最大化总调度量 - λ×总成本（确保调度发生）
3. 增加参数敏感性分析工具
"""

import os
import numpy as np
import pandas as pd
import warnings
import scipy.sparse.csgraph as csg
import scipy.sparse as sp
from ortools.linear_solver import pywraplp
from scipy import stats

warnings.filterwarnings('ignore')

# ==================== 配置区 ====================
PROJECT_ROOT = "/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity"
REL_PATH = os.path.join(PROJECT_ROOT, "raw_data/SZ_TAXI/SZ_TAXI.rel")
PRED_PATH = os.path.join(
    PROJECT_ROOT,
    "libcity/cache/STGCN-ROUND1/evaluate_cache/2026_06_02_19_52_43_STGCN_SZ_TAXI_predictions.npz"
)

DISTANCE_MODE = "weighted"  # "uniform" 或 "weighted"
TOTAL_CARS = 260
BASE_DEMAND = 0.5
SPEED_CAP = 86.43

# MIP成本惩罚系数（越大越重视成本，越小越重视调度量）
LAMBDA_COST = 0.3  # 每单位距离的成本权重

N_SEEDS = 15
RANDOM_SEEDS = [42, 100, 2025, 7, 19, 88, 123, 456, 789, 1024,
                2048, 333, 777, 999, 1111]


# ==================== 0. 拓扑分析 ====================
def analyze_rel_topology(rel_path):
    rel = pd.read_csv(rel_path)
    print("\n" + "="*70)
    print("📊 .rel 文件拓扑分析")
    print("="*70)
    print(f"总行数: {len(rel)}")
    print(f"link_weight分布: \n{rel['link_weight'].value_counts().sort_index()}")

    non_self = rel[rel['origin_id'] != rel['destination_id']]
    n_nodes = len(set(rel['origin_id']) | set(rel['destination_id']))
    out_degrees = non_self.groupby('origin_id').size()

    print(f"\n节点数: {n_nodes}")
    print(f"平均出度: {out_degrees.mean():.1f} (全连接期望: {n_nodes-1})")
    print(f"是否全连接: {'是' if out_degrees.mean() >= n_nodes-2 else '否'}")

    lw1 = non_self[non_self.link_weight == 1]
    if len(lw1) > 0:
        pairs = set()
        for _, row in lw1.iterrows():
            pairs.add((min(row['origin_id'], row['destination_id']),
                      max(row['origin_id'], row['destination_id'])))
        print(f"\nlink_weight=1的边: {len(lw1)}条, 去重无向对: {len(pairs)}对")

    print("\n" + "="*70)
    print("📝 论文声明模板")
    print("="*70)
    print('"""')
    print("SZ_TAXI数据集提供的.rel文件包含156个区域节点，共计24,336条关系记录。")
    print("经分析，该文件构成全连接图（每个节点与其他155个节点均有连接），")
    print("其中link_weight=1的边532条（266对无向边），link_weight=0的边23,648条。")
    print("由于.geo文件坐标字段为空，无法构建基于真实路网的地理距离矩阵。")
    print("因此，本研究采用以下距离代理策略：")
    if DISTANCE_MODE == "uniform":
        print("  - 所有非自环边统一视为单位距离（跳数=1）")
    else:
        print("  - link_weight=1的边距离设为1（强化连接）")
        print("  - link_weight=0的边距离设为2（普通连接）")
    print("该处理与LibCity框架中其他基于SZ_TAXI的研究保持一致。")
    print('"""')
    print("="*70)
    return rel


# ==================== 1. 容量衰减模型 ====================
class CongestionAwareDemand:
    def __init__(self, free_flow=40.0, min_ratio=0.3):
        self.free_flow = free_flow
        self.min_ratio = min_ratio

    def get_capacity_ratio(self, speed):
        speed = np.asarray(speed, dtype=float)
        ratio = np.piecewise(speed,
            [speed < 5,
             (speed >= 5) & (speed < 10),
             (speed >= 10) & (speed < 20),
             (speed >= 20) & (speed < 40),
             speed >= 40],
            [self.min_ratio,
             lambda v: self.min_ratio + 0.1 * (v - 5) / 5,
             lambda v: 0.4 + 0.2 * (v - 10) / 10,
             lambda v: 0.5 + 0.2 * (v - 20) / 20,
             1.0])
        return ratio

    def get_effective_demand(self, speed, base_demand_per_region=1.0):
        ratio = self.get_capacity_ratio(speed)
        ratio = np.maximum(ratio, 0.01)
        return base_demand_per_region / ratio


# ==================== 2. MIP优化器（核心修复） ====================
class CongestionAwareIPOptimizer:
    def __init__(self, num_regions, distance_matrix, speed_predictions, demand_model):
        self.num_regions = num_regions
        self.distance_matrix = distance_matrix.astype(float)
        self.speed_predictions = speed_predictions
        self.demand_model = demand_model

    def solve_rebalancing(self, current_supply, future_time_idx=0):
        future_speed = self.speed_predictions[future_time_idx].flatten()
        future_speed = np.clip(future_speed, 0, SPEED_CAP)

        effective_demand = self.demand_model.get_effective_demand(future_speed, BASE_DEMAND)
        current_supply_flat = np.array(current_supply).ravel().astype(float)

        surplus = np.maximum(0, current_supply_flat - effective_demand)
        deficit = np.maximum(0, effective_demand - current_supply_flat)

        print(f"   [MIP] 速度均值{future_speed.mean():.1f} | 总需求{effective_demand.sum():.1f}")
        print(f"   [MIP] 盈余: sum={surplus.sum():.2f}, >0.01={np.sum(surplus>0.01)}区")
        print(f"   [MIP] 缺口: sum={deficit.sum():.2f}, >0.01={np.sum(deficit>0.01)}区")

        THRESHOLD = 0.01
        if surplus.sum() < THRESHOLD or deficit.sum() < THRESHOLD:
            print(f"   [MIP] ⚠️ 无需调度")
            return [], 0.0

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise RuntimeError("SCIP求解器初始化失败")

        surplus_idx = np.where(surplus > THRESHOLD)[0]
        deficit_idx = np.where(deficit > THRESHOLD)[0]

        # 决策变量 X_ij
        X = {}
        for i in surplus_idx:
            for j in deficit_idx:
                if self.distance_matrix[i, j] < 500:
                    X[(i, j)] = solver.IntVar(0, int(np.ceil(surplus[i])), f'X_{i}_{j}')

        print(f"   [MIP] 决策变量数: {len(X)}")
        if len(X) == 0:
            return [], 0.0

        # 🔴 核心修复：目标函数改为 最大化(调度量 - λ×成本)
        # 这样solver有动力调度车辆，因为每调度1辆车获得收益1
        obj = solver.Objective()
        for (i, j), var in X.items():
            # 收益 = 1 - λ × 距离
            coeff = 1.0 - LAMBDA_COST * float(self.distance_matrix[i, j])
            obj.SetCoefficient(var, coeff)
        obj.SetMaximization()

        # 🔴 修复：约束用 ceil 而不是 floor
        # 约束1：流出 <= ceil(盈余)
        for i in surplus_idx:
            relevant = [X[(i, j)] for j in deficit_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.ceil(surplus[i])))

        # 约束2：流入 <= ceil(缺口)
        for j in deficit_idx:
            relevant = [X[(i, j)] for i in surplus_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.ceil(deficit[j])))

        print(f"   [MIP] 求解中 (λ={LAMBDA_COST})...")
        status = solver.Solve()
        plan = []
        cost = 0.0
        total_moved = 0

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            for (i, j), var in X.items():
                val = int(var.solution_value())
                if val > 0:
                    plan.append((i, j, val))
                    cost += val * self.distance_matrix[i, j]
                    total_moved += val
            plan.sort(key=lambda x: -x[2])
            print(f"   [MIP] ✅ 求解成功({status}), {len(plan)}条路线, "
                  f"调度{total_moved}辆, 成本{cost:.1f}")
        else:
            print(f"   [MIP] ❌ 求解失败({status}), 回退贪心")
            return self._greedy_fallback(surplus, deficit)

        return plan, cost

    def _greedy_fallback(self, surplus, deficit):
        surplus_regions = np.where(surplus > 0.01)[0]
        deficit_regions = np.where(deficit > 0.01)[0]
        plan = []
        cost = 0.0
        s_list = surplus.copy()
        d_list = deficit.copy()

        for i in sorted(surplus_regions, key=lambda x: -s_list[x]):
            if s_list[i] < 0.01:
                continue
            candidates = [(j, self.distance_matrix[i, j])
                          for j in deficit_regions if d_list[j] >= 0.01]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])

            for j, dist in candidates:
                if s_list[i] < 0.01 or d_list[j] < 0.01:
                    continue
                move = int(min(np.ceil(s_list[i]), np.ceil(d_list[j])))
                if move > 0:
                    plan.append((i, j, move))
                    cost += move * dist
                    s_list[i] -= move
                    d_list[j] -= move
        return plan, cost


# ==================== 3. 路网 ====================
class RoadNetwork:
    def __init__(self, rel_path: str, dist_matrix_path: str = None, mode="weighted"):
        rel = pd.read_csv(rel_path)

        all_ids = sorted(set(rel["origin_id"]) | set(rel["destination_id"]))
        self.nodes = all_ids
        self.n = len(all_ids)
        self.id2idx = {nid: i for i, nid in enumerate(all_ids)}

        print(f"✅ 路网加载: {self.n} 个区域")

        edges = rel[rel["origin_id"] != rel["destination_id"]]
        print(f"   非自环边: {len(edges)} (lw=0: {len(edges[edges.link_weight==0])}, "
              f"lw=1: {len(edges[edges.link_weight==1])})")

        if dist_matrix_path is None:
            suffix = "_weighted" if mode == "weighted" else "_uniform"
            dist_matrix_path = os.path.join(os.path.dirname(rel_path), f"sz_taxi_dist_matrix{suffix}.npy")

        if os.path.exists(dist_matrix_path):
            self._dist_matrix = np.load(dist_matrix_path).astype(np.float64)
            print(f"✅ 距离矩阵已加载: {dist_matrix_path}")
        else:
            print(f"⚠️  计算距离矩阵 (模式={mode})...")
            self._dist_matrix = self._compute_distance_matrix(edges, mode)
            np.save(dist_matrix_path, self._dist_matrix)
            print(f"✅ 已保存: {dist_matrix_path}")

    def _compute_distance_matrix(self, edges, mode):
        n = self.n
        adj = np.zeros((n, n), dtype=np.float32)

        if mode == "uniform":
            edge_pairs = set()
            for _, row in edges.iterrows():
                i, j = row["origin_id"], row["destination_id"]
                edge_pairs.add((min(i, j), max(i, j)))
            for i_id, j_id in edge_pairs:
                i = self.id2idx[i_id]
                j = self.id2idx[j_id]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        else:
            for _, row in edges.iterrows():
                i = self.id2idx[row["origin_id"]]
                j = self.id2idx[row["destination_id"]]
                weight = 2.0 if row["link_weight"] == 0 else 1.0
                if adj[i, j] == 0 or weight < adj[i, j]:
                    adj[i, j] = weight
                    adj[j, i] = weight

        sparse_adj = sp.csr_matrix(adj)

        n_comp, labels = csg.connected_components(sparse_adj, directed=False)
        print(f"   连通分量: {n_comp}")
        if n_comp > 1:
            sizes = np.bincount(labels)
            print(f"   分量大小: {sorted(sizes, reverse=True)}")

        dist = csg.shortest_path(sparse_adj, method="D", directed=False, unweighted=False)

        finite_max = dist[np.isfinite(dist)].max() if np.any(np.isfinite(dist)) else 10
        penalty = int(finite_max * 2) + 10
        dist[np.isinf(dist)] = penalty
        np.fill_diagonal(dist, 0)

        finite_dist = dist[np.isfinite(dist) & (dist > 0)]
        if len(finite_dist) > 0:
            print(f"   距离分布: min={finite_dist.min():.1f}, max={finite_dist.max():.1f}, "
                  f"mean={finite_dist.mean():.2f}")
        print(f"   不可达填充: {penalty}, 不可达对: {np.sum(dist >= penalty)}")

        return dist.astype(np.float64)

    def generate_distance_matrix(self, seed=None):
        return self._dist_matrix.copy()


# ==================== 4. 调度策略 ====================
class DispatchStrategies:
    @staticmethod
    def calculate_service_rate(supply, demand):
        s = np.array(supply).ravel().astype(float)
        d = np.array(demand).ravel().astype(float)
        matched = np.minimum(s, d).sum()
        total = d.sum()
        return matched / total if total > 1e-6 else 1.0

    @staticmethod
    def no_dispatch(supply, demand):
        s = np.array(supply).ravel().astype(float)
        return s, 0, 0.0, DispatchStrategies.calculate_service_rate(s, demand)

    @staticmethod
    def random_dispatch(supply, demand, distance_matrix, seed):
        np.random.seed(seed)
        s = np.array(supply).ravel().astype(float)
        d = np.array(demand).ravel().astype(float)
        surplus = np.maximum(s - d, 0)
        deficit = np.maximum(d - s, 0)

        total_move = 0
        total_dist = 0.0
        s_regions = list(np.where(surplus > 0.01)[0])
        d_regions = list(np.where(deficit > 0.01)[0])
        np.random.shuffle(s_regions)

        for i in s_regions:
            if surplus[i] < 0.01 or not d_regions:
                break
            j = np.random.choice(d_regions)
            move = min(int(np.ceil(surplus[i])), int(np.ceil(deficit[j])))
            if move > 0:
                total_move += move
                total_dist += move * distance_matrix[i, j]
                surplus[i] -= move
                deficit[j] -= move
                s[i] -= move
                s[j] += move
                if deficit[j] < 0.01:
                    d_regions.remove(j)

        return s, total_move, total_dist, DispatchStrategies.calculate_service_rate(s, d)

    @staticmethod
    def nearest_dispatch(supply, demand, distance_matrix):
        s = np.array(supply).ravel().astype(float)
        d = np.array(demand).ravel().astype(float)
        surplus = np.maximum(s - d, 0)
        deficit = np.maximum(d - s, 0)

        total_move = 0
        total_dist = 0.0

        for i in np.argsort(-surplus):
            if surplus[i] < 0.01:
                continue
            candidates = [(j, distance_matrix[i, j]) for j in range(len(s)) if deficit[j] > 0.01]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])

            for j, dist in candidates:
                if surplus[i] < 0.01 or deficit[j] < 0.01:
                    continue
                move = min(int(np.ceil(surplus[i])), int(np.ceil(deficit[j])))
                if move > 0:
                    total_move += move
                    total_dist += move * dist
                    surplus[i] -= move
                    deficit[j] -= move
                    s[i] -= move
                    s[j] += move

        return s, total_move, total_dist, DispatchStrategies.calculate_service_rate(s, d)

    @staticmethod
    def demand_only_dispatch(supply, base_demand, distance_matrix):
        s = np.array(supply).ravel().astype(float)
        d = np.full_like(s, base_demand)
        surplus = np.maximum(s - d, 0)
        deficit = np.maximum(d - s, 0)

        total_move = 0
        total_dist = 0.0

        for i in np.argsort(-surplus):
            if surplus[i] < 0.01:
                continue
            candidates = [(j, distance_matrix[i, j]) for j in range(len(s)) if deficit[j] > 0.01]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])

            for j, dist in candidates:
                if surplus[i] < 0.01 or deficit[j] < 0.01:
                    continue
                move = min(int(np.ceil(surplus[i])), int(np.ceil(deficit[j])))
                if move > 0:
                    total_move += move
                    total_dist += move * dist
                    surplus[i] -= move
                    deficit[j] -= move
                    s[i] -= move
                    s[j] += move

        return s, total_move, total_dist


# ==================== 5. 仿真器 ====================
class CongestionAwareSimulator:
    def __init__(self, road_network, speed_predictions, total_cars=TOTAL_CARS, base_demand=BASE_DEMAND):
        self.road = road_network
        self.speed_predictions = speed_predictions
        self.total_cars = total_cars
        self.n_regions = road_network.n
        self.base_demand = base_demand
        self.demand_model = CongestionAwareDemand()
        self._precompute_demands()

    def _precompute_demands(self):
        n_time = self.speed_predictions.shape[0]
        self.demands = np.zeros((n_time, self.n_regions))
        for t in range(n_time):
            speeds = self.speed_predictions[t].flatten()
            speeds = np.clip(speeds, 0, SPEED_CAP)
            self.demands[t] = self.demand_model.get_effective_demand(speeds, self.base_demand)
        print(f"✅ 预计算完成: {n_time}个时间片, 平均需求{self.demands.mean():.2f}/区域")

    def _init_supply(self, seed):
        np.random.seed(seed)
        supply = np.zeros(self.n_regions)
        positions = np.random.choice(self.n_regions, self.total_cars, replace=True)
        for pos in positions:
            supply[pos] += 1
        return supply

    def run_single_seed(self, seed, time_idx=0):
        supply = self._init_supply(seed)
        distance_matrix = self.road.generate_distance_matrix(seed)
        congestion_demand = self.demands[time_idx].copy()
        speeds = np.clip(self.speed_predictions[time_idx].flatten(), 0, SPEED_CAP)

        ceiling = self.total_cars / congestion_demand.sum()
        print(f"\n🌱 种子{seed} | 速度均值{speeds.mean():.1f} | "
              f"总需求{congestion_demand.sum():.1f} | 天花板{ceiling:.1%}")

        results = {}

        _, _, _, sr_no = DispatchStrategies.no_dispatch(supply, congestion_demand)
        results['no_dispatch'] = {'service_rate': sr_no, 'move': 0, 'distance': 0}

        _, move_r, dist_r, sr_r = DispatchStrategies.random_dispatch(supply, congestion_demand, distance_matrix, seed)
        results['random'] = {'service_rate': sr_r, 'move': move_r, 'distance': dist_r}

        _, move_n, dist_n, sr_n = DispatchStrategies.nearest_dispatch(supply, congestion_demand, distance_matrix)
        results['nearest'] = {'service_rate': sr_n, 'move': move_n, 'distance': dist_n}

        s_do, move_do, dist_do = DispatchStrategies.demand_only_dispatch(supply, self.base_demand, distance_matrix)
        sr_do = DispatchStrategies.calculate_service_rate(s_do, congestion_demand)
        results['demand_only'] = {'service_rate': sr_do, 'move': move_do, 'distance': dist_do}

        ip = CongestionAwareIPOptimizer(self.n_regions, distance_matrix, self.speed_predictions, self.demand_model)
        plan, cost = ip.solve_rebalancing(supply, time_idx)

        s_ca = np.array(supply).ravel().astype(float)
        move_ca = 0
        for i, j, amt in plan:
            move_ca += amt
            s_ca[i] -= amt
            s_ca[j] += amt
        sr_ca = DispatchStrategies.calculate_service_rate(s_ca, congestion_demand)
        results['congestion_aware'] = {'service_rate': sr_ca, 'move': move_ca, 'distance': cost}

        print(f"  结果: 无调度{sr_no:.1%} | 随机{sr_r:.1%} | 就近{sr_n:.1%} | "
              f"DemandOnly{sr_do:.1%} | MIP{sr_ca:.1%}")
        print(f"  MIP提升: vs随机+{sr_ca-sr_r:.1%} vs就近+{sr_ca-sr_n:.1%} vsDemandOnly+{sr_ca-sr_do:.1%}")

        return results

    def run_multiple_seeds(self, seeds, time_idx=0):
        all_results = []
        for seed in seeds:
            all_results.append(self.run_single_seed(seed, time_idx))

        print(f"\n{'='*80}")
        print(f"📊 统计结果 ({len(seeds)}个种子)")
        print(f"{'='*80}")

        strategies = ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']
        names = ['无调度', '随机调度', '就近调度', 'Demand-Only(消融)', '拥堵感知MIP']

        summary = {}
        for strat, name in zip(strategies, names):
            rates = [r[strat]['service_rate'] for r in all_results]
            moves = [r[strat]['move'] for r in all_results]
            summary[strat] = {'mean_sr': np.mean(rates), 'std_sr': np.std(rates)}

            print(f"\n{name}:")
            print(f"  服务率: {np.mean(rates):.1%} ± {np.std(rates):.1%}")
            print(f"  调度车辆: {np.mean(moves):.1f} ± {np.std(moves):.1f}")

        # 统计检验
        print(f"\n{'='*80}")
        print(f"📈 统计显著性检验")
        print(f"{'='*80}")

        ca_rates = [r['congestion_aware']['service_rate'] for r in all_results]

        for baseline in ['random', 'nearest', 'demand_only']:
            base_rates = [r[baseline]['service_rate'] for r in all_results]
            diff = np.array(ca_rates) - np.array(base_rates)

            t_stat, p_val = stats.ttest_rel(ca_rates, base_rates)
            cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
            ci_lo = diff.mean() - 1.96 * diff.std() / np.sqrt(len(seeds))
            ci_hi = diff.mean() + 1.96 * diff.std() / np.sqrt(len(seeds))
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            print(f"\n拥堵感知MIP vs {baseline}:")
            print(f"  提升: {diff.mean():.2%} [{ci_lo:.2%}, {ci_hi:.2%}]")
            print(f"  t({len(seeds)-1})={t_stat:.3f}, p={p_val:.4f} {sig}")
            print(f"  Cohen's d={cohens_d:.3f}")

        print(f"\n{'='*80}")
        print(f"🎉 核心结论: 拥堵感知MIP = {summary['congestion_aware']['mean_sr']:.1%}")
        print(f"{'='*80}")

        return all_results, summary


# ==================== 6. 主函数 ====================
def main():
    print("="*80)
    print("🚀 拥堵感知前置调度 —— 最终学术版 v4.0")
    print(f"   距离模式: {DISTANCE_MODE} | λ={LAMBDA_COST}")
    print("="*80)

    analyze_rel_topology(REL_PATH)
    road = RoadNetwork(REL_PATH, mode=DISTANCE_MODE)

    pred_data = np.load(PRED_PATH)
    raw_pred = pred_data["prediction"][:, 0, :]
    if len(raw_pred.shape) == 3:
        raw_pred = raw_pred.squeeze()

    print(f"\n📊 STGCN原始: 范围[{raw_pred.min():.2f}, {raw_pred.max():.2f}], 均值{raw_pred.mean():.2f}")
    speed_predictions = np.clip(raw_pred, 0, SPEED_CAP)
    print(f"📊 后处理后: 范围[{speed_predictions.min():.2f}, {speed_predictions.max():.2f}], 均值{speed_predictions.mean():.2f}")

    simulator = CongestionAwareSimulator(road, speed_predictions, total_cars=TOTAL_CARS, base_demand=BASE_DEMAND)
    all_results, summary = simulator.run_multiple_seeds(RANDOM_SEEDS, time_idx=0)

    print("\n✅ 实验完成！")
    return all_results, summary


if __name__ == "__main__":
    main()
