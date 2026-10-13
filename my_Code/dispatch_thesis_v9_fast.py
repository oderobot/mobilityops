"""
============================================================
📦 拥堵感知前置调度实验系统 —— 快速版 v9.0
============================================================
修复：去掉Y_j连续变量（导致SCIP求解慢），改用带源区域惩罚的调度量最大化

核心思想：
- 从大盈余区域调度：惩罚=0（鼓励）
- 从小盈余区域调度：惩罚>0（抑制）
- 这样solver自然优先从"调度无损失"的区域调车

求解速度：毫秒级（与v1-v7相同）
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
DYNAPATH = os.path.join(PROJECT_ROOT, "raw_data/SZ_TAXI/SZ_TAXI.dyna")

STGCN_PRED = os.path.join(PROJECT_ROOT,
    "libcity/cache/STGCN-ROUND1/evaluate_cache/2026_06_02_19_52_43_STGCN_SZ_TAXI_predictions.npz")
RNN_PRED = os.path.join(PROJECT_ROOT,
    "libcity/cache/RNN-ROUND1/evaluate_cache/2026_06_02_20_15_08_RNN_SZ_TAXI_predictions.npz")

TOTAL_CARS = 260
BASE_DEMAND = 0.5
SPEED_CAP = 86.43
N_SEEDS = 10
RANDOM_SEEDS = [42, 100, 2025, 7, 19, 88, 123, 456, 789, 1024]
N_TIME_SLICES = 10

SOURCE_PENALTY_WEIGHT = 3.0  # 源区域调度惩罚权重


# ==================== 0. 加载真实速度 ====================
def load_true_speeds(dyna_path, n_regions=156):
    df = pd.read_csv(dyna_path)
    df = df.sort_values(['time', 'entity_id'])
    speeds = df['traffic_speed'].values
    n_time = len(speeds) // n_regions
    return speeds[:n_time * n_regions].reshape(n_time, n_regions)


# ==================== 1. 容量衰减模型 ====================
class CongestionAwareDemand:
    def __init__(self, free_flow=40.0, min_ratio=0.3):
        self.free_flow = free_flow
        self.min_ratio = min_ratio

    def get_capacity_ratio(self, speed):
        speed = np.asarray(speed, dtype=float)
        return np.piecewise(speed,
            [speed < 5, (speed >= 5) & (speed < 10), (speed >= 10) & (speed < 20),
             (speed >= 20) & (speed < 40), speed >= 40],
            [self.min_ratio, lambda v: self.min_ratio + 0.1*(v-5)/5,
             lambda v: 0.4 + 0.2*(v-10)/10, lambda v: 0.5 + 0.2*(v-20)/20, 1.0])

    def get_effective_demand(self, speed, base_demand_per_region=1.0):
        ratio = self.get_capacity_ratio(speed)
        return base_demand_per_region / np.maximum(ratio, 0.01)


# ==================== 2. MIP优化器（快速版） ====================
class CongestionAwareIPOptimizer:
    def __init__(self, num_regions, distance_matrix, speed_predictions, demand_model, source_penalty=3.0):
        self.num_regions = num_regions
        self.distance_matrix = distance_matrix.astype(float)
        self.speed_predictions = speed_predictions
        self.demand_model = demand_model
        self.source_penalty = source_penalty

    def solve_rebalancing(self, current_supply, future_time_idx=0):
        future_speed = np.clip(self.speed_predictions[future_time_idx].flatten(), 0, SPEED_CAP)
        effective_demand = self.demand_model.get_effective_demand(future_speed, BASE_DEMAND)
        current_supply_flat = np.array(current_supply).ravel().astype(float)

        surplus = np.maximum(0, current_supply_flat - effective_demand)
        deficit = np.maximum(0, effective_demand - current_supply_flat)

        if surplus.sum() < 0.01 or deficit.sum() < 0.01:
            return [], 0.0, 0

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise RuntimeError("SCIP求解器初始化失败")

        surplus_idx = np.where(surplus > 0.01)[0]
        deficit_idx = np.where(deficit > 0.01)[0]

        X = {}
        for i in surplus_idx:
            for j in deficit_idx:
                if self.distance_matrix[i, j] < 500:
                    X[(i, j)] = solver.IntVar(0, int(np.ceil(surplus[i])), f'X_{i}_{j}')

        if len(X) == 0:
            return [], 0.0, 0

        # 快速版目标：最大化(调度量 - 源区域惩罚)
        # 惩罚 = max(0, 1.0 - surplus_i) * weight
        # 大盈余区域(surplus>=1)：惩罚=0，鼓励调度
        # 小盈余区域(surplus<1)：惩罚>0，抑制调度
        obj = solver.Objective()
        for (i, j), var in X.items():
            # 基础收益
            reward = 1.0
            # 源区域损失惩罚
            penalty = max(0.0, 1.0 - surplus[i]) * self.source_penalty
            # 净收益
            net = reward - penalty
            obj.SetCoefficient(var, net)
        obj.SetMaximization()

        # 约束
        for i in surplus_idx:
            relevant = [X[(i, j)] for j in deficit_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.ceil(surplus[i])))
        for j in deficit_idx:
            relevant = [X[(i, j)] for i in surplus_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.ceil(deficit[j])))

        status = solver.Solve()
        plan, cost, total_moved = [], 0.0, 0

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            for (i, j), var in X.items():
                val = int(var.solution_value())
                if val > 0:
                    plan.append((i, j, val))
                    cost += val * self.distance_matrix[i, j]
                    total_moved += val
            plan.sort(key=lambda x: -x[2])

        return plan, cost, total_moved


# ==================== 3. 路网 ====================
class RoadNetwork:
    def __init__(self, rel_path, dist_matrix_path=None, mode="weighted"):
        rel = pd.read_csv(rel_path)
        all_ids = sorted(set(rel["origin_id"]) | set(rel["destination_id"]))
        self.nodes, self.n, self.id2idx = all_ids, len(all_ids), {nid: i for i, nid in enumerate(all_ids)}
        edges = rel[rel["origin_id"] != rel["destination_id"]]
        if dist_matrix_path is None:
            suffix = "_weighted" if mode == "weighted" else "_uniform"
            dist_matrix_path = os.path.join(os.path.dirname(rel_path), f"sz_taxi_dist_matrix{suffix}.npy")
        if os.path.exists(dist_matrix_path):
            self._dist_matrix = np.load(dist_matrix_path).astype(np.float64)
        else:
            self._dist_matrix = self._compute_distance_matrix(edges, mode)
            np.save(dist_matrix_path, self._dist_matrix)

    def _compute_distance_matrix(self, edges, mode):
        n, adj = self.n, np.zeros((self.n, self.n), dtype=np.float32)
        if mode == "uniform":
            pairs = set((min(r["origin_id"], r["destination_id"]), max(r["origin_id"], r["destination_id"]))
                       for _, r in edges.iterrows())
            for i_id, j_id in pairs:
                i, j = self.id2idx[i_id], self.id2idx[j_id]
                adj[i, j] = adj[j, i] = 1.0
        else:
            for _, r in edges.iterrows():
                i, j = self.id2idx[r["origin_id"]], self.id2idx[r["destination_id"]]
                w = 2.0 if r["link_weight"] == 0 else 1.0
                if adj[i, j] == 0 or w < adj[i, j]: adj[i, j] = adj[j, i] = w
        dist = csg.shortest_path(sp.csr_matrix(adj), method="D", directed=False, unweighted=False)
        dist[np.isinf(dist)] = dist[np.isfinite(dist)].max() * 2 + 10 if np.any(np.isfinite(dist)) else 10
        np.fill_diagonal(dist, 0)
        return dist.astype(np.float64)

    def generate_distance_matrix(self, seed=None):
        return self._dist_matrix.copy()


# ==================== 4. 调度策略 ====================
class DispatchStrategies:
    @staticmethod
    def calculate_service_rate(supply, demand):
        s, d = np.array(supply).ravel().astype(float), np.array(demand).ravel().astype(float)
        return np.minimum(s, d).sum() / d.sum() if d.sum() > 1e-6 else 1.0

    @staticmethod
    def no_dispatch(supply, demand):
        s = np.array(supply).ravel().astype(float)
        return s, 0, 0.0, DispatchStrategies.calculate_service_rate(s, demand)

    @staticmethod
    def random_dispatch(supply, demand, dm, seed):
        np.random.seed(seed)
        s, d = np.array(supply).ravel().astype(float), np.array(demand).ravel().astype(float)
        surplus, deficit = np.maximum(s - d, 0), np.maximum(d - s, 0)
        move, dist = 0, 0.0
        s_reg = list(np.where(surplus > 0.01)[0])
        d_reg = list(np.where(deficit > 0.01)[0])
        np.random.shuffle(s_reg)
        for i in s_reg:
            if surplus[i] < 0.01 or not d_reg: break
            j = np.random.choice(d_reg)
            m = min(int(np.ceil(surplus[i])), int(np.ceil(deficit[j])))
            if m > 0:
                move += m; dist += m * dm[i, j]
                surplus[i] -= m; deficit[j] -= m; s[i] -= m; s[j] += m
                if deficit[j] < 0.01: d_reg.remove(j)
        return s, move, dist, DispatchStrategies.calculate_service_rate(s, d)

    @staticmethod
    def nearest_dispatch(supply, demand, dm):
        s, d = np.array(supply).ravel().astype(float), np.array(demand).ravel().astype(float)
        surplus, deficit = np.maximum(s - d, 0), np.maximum(d - s, 0)
        move, dist = 0, 0.0
        for i in np.argsort(-surplus):
            if surplus[i] < 0.01: continue
            cand = [(j, dm[i, j]) for j in range(len(s)) if deficit[j] > 0.01]
            if not cand: break
            for j, dst in sorted(cand, key=lambda x: x[1]):
                if surplus[i] < 0.01 or deficit[j] < 0.01: continue
                m = min(int(np.ceil(surplus[i])), int(np.ceil(deficit[j])))
                if m > 0:
                    move += m; dist += m * dst
                    surplus[i] -= m; deficit[j] -= m; s[i] -= m; s[j] += m
        return s, move, dist, DispatchStrategies.calculate_service_rate(s, d)

    @staticmethod
    def demand_only_dispatch(supply, base_demand, dm):
        s, d = np.array(supply).ravel().astype(float), np.full_like(np.array(supply).ravel().astype(float), base_demand)
        surplus, deficit = np.maximum(s - d, 0), np.maximum(d - s, 0)
        move, dist = 0, 0.0
        for i in np.argsort(-surplus):
            if surplus[i] < 0.01: continue
            cand = [(j, dm[i, j]) for j in range(len(s)) if deficit[j] > 0.01]
            if not cand: break
            for j, dst in sorted(cand, key=lambda x: x[1]):
                if surplus[i] < 0.01 or deficit[j] < 0.01: continue
                m = min(int(np.ceil(surplus[i])), int(np.ceil(deficit[j])))
                if m > 0:
                    move += m; dist += m * dst
                    surplus[i] -= m; deficit[j] -= m; s[i] -= m; s[j] += m
        return s, move, dist


# ==================== 5. 仿真器 ====================
class CongestionAwareSimulator:
    def __init__(self, road_network, pred_speeds, true_speeds, total_cars=260, base_demand=0.5, source_penalty=3.0):
        self.road = road_network
        self.pred_speeds = pred_speeds
        self.true_speeds = true_speeds
        self.total_cars = total_cars
        self.n_regions = road_network.n
        self.base_demand = base_demand
        self.demand_model = CongestionAwareDemand()
        self.source_penalty = source_penalty
        self._precompute_demands()

    def _precompute_demands(self):
        n_time = self.pred_speeds.shape[0]
        self.pred_demands = np.zeros((n_time, self.n_regions))
        self.true_demands = np.zeros((n_time, self.n_regions))
        for t in range(n_time):
            ps = np.clip(self.pred_speeds[t].flatten(), 0, SPEED_CAP)
            ts = np.clip(self.true_speeds[t].flatten(), 0, SPEED_CAP)
            self.pred_demands[t] = self.demand_model.get_effective_demand(ps, self.base_demand)
            self.true_demands[t] = self.demand_model.get_effective_demand(ts, self.base_demand)

    def _init_supply(self, seed):
        np.random.seed(seed)
        supply = np.zeros(self.n_regions)
        for pos in np.random.choice(self.n_regions, self.total_cars, replace=True):
            supply[pos] += 1
        return supply

    def run_single_experiment(self, seed, time_idx):
        supply = self._init_supply(seed)
        dm = self.road.generate_distance_matrix(seed)
        pred_demand = self.pred_demands[time_idx]
        true_demand = self.true_demands[time_idx]

        results = {}

        _, _, _, sr_no = DispatchStrategies.no_dispatch(supply, true_demand)
        results['no_dispatch'] = sr_no

        _, _, _, sr_rand = DispatchStrategies.random_dispatch(supply, true_demand, dm, seed)
        results['random'] = sr_rand

        _, _, _, sr_nn = DispatchStrategies.nearest_dispatch(supply, true_demand, dm)
        results['nearest'] = sr_nn

        s_do, _, _ = DispatchStrategies.demand_only_dispatch(supply, self.base_demand, dm)
        results['demand_only'] = DispatchStrategies.calculate_service_rate(s_do, true_demand)

        ip = CongestionAwareIPOptimizer(self.n_regions, dm, self.pred_speeds, self.demand_model, self.source_penalty)
        plan, cost, moved = ip.solve_rebalancing(supply, time_idx)
        s_ca = np.array(supply).ravel().astype(float)
        for i, j, amt in plan:
            s_ca[i] -= amt; s_ca[j] += amt
        results['congestion_aware'] = DispatchStrategies.calculate_service_rate(s_ca, true_demand)

        return results

    def run_multi_seed_multi_time(self, seeds, time_indices):
        all_results = []
        for seed in seeds:
            for t in time_indices:
                all_results.append(self.run_single_experiment(seed, t))

        strategies = ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']
        summary = {}
        for strat in strategies:
            rates = [r[strat] for r in all_results]
            summary[strat] = {'mean': np.mean(rates), 'std': np.std(rates), 'n': len(rates)}

        return all_results, summary


# ==================== 6. 主实验 ====================
def main():
    print("="*80)
    print("🚀 拥堵感知前置调度 —— 快速版 v9.0")
    print(f"   源区域惩罚: {SOURCE_PENALTY_WEIGHT} | 预计运行时间: 1-2分钟")
    print("="*80)

    road = RoadNetwork(REL_PATH, mode="weighted")
    true_speeds = load_true_speeds(DYNAPATH)

    stgcn_raw = np.load(STGCN_PRED)["prediction"][:, 0, :]
    rnn_raw = np.load(RNN_PRED)["prediction"][:, 0, :]
    if len(stgcn_raw.shape) == 3: stgcn_raw = stgcn_raw.squeeze()
    if len(rnn_raw.shape) == 3: rnn_raw = rnn_raw.squeeze()

    n_pred = stgcn_raw.shape[0]
    n_true = true_speeds.shape[0]
    start_idx = max(0, n_true - n_pred)
    true_aligned = true_speeds[start_idx:start_idx + n_pred]
    min_len = min(n_pred, true_aligned.shape[0])
    stgcn_pred = np.clip(stgcn_raw[:min_len], 0, SPEED_CAP)
    rnn_pred = np.clip(rnn_raw[:min_len], 0, SPEED_CAP)
    true_aligned = true_aligned[:min_len]
    time_indices = list(range(min(N_TIME_SLICES, min_len)))

    print(f"\n📊 STGCN: {stgcn_pred.shape}, MAE≈2.62")
    print(f"📊 RNN: {rnn_pred.shape}, MAE≈3.05")
    print(f"📊 实验规模: {len(time_indices)}时间片 × {N_SEEDS}种子 = {len(time_indices)*N_SEEDS}次")

    # 实验1: STGCN驱动
    print(f"\n{'='*80}")
    print("📊 实验1: STGCN预测驱动")
    print(f"{'='*80}")
    sim_stgcn = CongestionAwareSimulator(road, stgcn_pred, true_aligned, 
                                         total_cars=TOTAL_CARS, source_penalty=SOURCE_PENALTY_WEIGHT)
    _, summary_stgcn = sim_stgcn.run_multi_seed_multi_time(RANDOM_SEEDS, time_indices)

    for strat in ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']:
        s = summary_stgcn[strat]
        print(f"  {strat}: {s['mean']:.1%} ± {s['std']:.1%} (n={s['n']})")

    # 实验2: RNN驱动
    print(f"\n{'='*80}")
    print("📊 实验2: RNN预测驱动")
    print(f"{'='*80}")
    sim_rnn = CongestionAwareSimulator(road, rnn_pred, true_aligned, 
                                       total_cars=TOTAL_CARS, source_penalty=SOURCE_PENALTY_WEIGHT)
    _, summary_rnn = sim_rnn.run_multi_seed_multi_time(RANDOM_SEEDS, time_indices)

    for strat in ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']:
        s = summary_rnn[strat]
        print(f"  {strat}: {s['mean']:.1%} ± {s['std']:.1%} (n={s['n']})")

    # 对比表
    print(f"\n{'='*80}")
    print("📊 STGCN vs RNN 端到端对比")
    print(f"{'='*80}")
    print(f"{'策略':<20} {'STGCN':<15} {'RNN':<15} {'差距':<15}")
    print("-"*65)
    for strat in ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']:
        stgcn_val = summary_stgcn[strat]['mean']
        rnn_val = summary_rnn[strat]['mean']
        diff = stgcn_val - rnn_val
        print(f"{strat:<20} {stgcn_val:<14.1%} {rnn_val:<14.1%} {diff:+.1%}")

    # 核心结论
    ca_stgcn = summary_stgcn['congestion_aware']['mean']
    ca_rnn = summary_rnn['congestion_aware']['mean']
    do_stgcn = summary_stgcn['demand_only']['mean']
    nn_stgcn = summary_stgcn['nearest']['mean']

    print(f"\n{'='*80}")
    print("🎉 核心结论")
    print(f"{'='*80}")
    print(f"  STGCN-MIP: {ca_stgcn:.1%}")
    print(f"  RNN-MIP:   {ca_rnn:.1%}")
    print(f"  Demand-Only: {do_stgcn:.1%}")
    print(f"  就近调度:    {nn_stgcn:.1%}")
    print(f"  拥堵感知 vs Demand-Only: {ca_stgcn - do_stgcn:+.1%}")
    print(f"  拥堵感知 vs 就近: {ca_stgcn - nn_stgcn:+.1%}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
