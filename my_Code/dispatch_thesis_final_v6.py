"""
============================================================
📦 拥堵感知前置调度实验系统 —— 硕士论文完整版 v6.0
============================================================
包含：
1. 多时间片滚动仿真（10个时间片）
2. STGCN vs RNN vs Oracle 三驱动对比
3. 参数敏感性分析（车辆数: 200, 260, 320）
4. 多目标MIP（β调参: 1, 5, 10）
5. 完整统计检验与论文图表数据输出
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

DISTANCE_MODE = "weighted"
BASE_DEMAND = 0.5
SPEED_CAP = 86.43
BETA = 5.0  # 未满足需求惩罚系数

N_SEEDS = 10
RANDOM_SEEDS = [42, 100, 2025, 7, 19, 88, 123, 456, 789, 1024]

# 实验配置开关
RUN_MULTITIME = True      # 多时间片滚动
RUN_MODEL_COMPARISON = True  # STGCN vs RNN vs Oracle
RUN_SENSITIVITY = True    # 车辆数敏感性
RUN_BETA_SWEEP = True     # β敏感性

N_TIME_SLICES = 10        # 滚动时间片数
TOTAL_CARS_LIST = [200, 260, 320]  # 敏感性分析车辆数
BETA_LIST = [1, 5, 10]    # β敏感性


# ==================== 0. 加载真实速度（用于Oracle） ====================
def load_true_speeds(dyna_path, n_regions=156):
    """从.dyna加载真实速度，reshape为 (n_time, n_regions)"""
    df = pd.read_csv(dyna_path)
    # 按时间排序，提取速度
    df = df.sort_values(['time', 'entity_id'])
    speeds = df['traffic_speed'].values
    # 计算时间片数
    n_total = len(speeds)
    n_time = n_total // n_regions
    speeds = speeds[:n_time * n_regions].reshape(n_time, n_regions)
    print(f"✅ 真实速度加载: {n_time}个时间片, {n_regions}个区域")
    print(f"   范围: [{speeds.min():.2f}, {speeds.max():.2f}], 均值{speeds.mean():.2f}")
    return speeds


# ==================== 1. 容量衰减模型 ====================
class CongestionAwareDemand:
    def __init__(self, free_flow=40.0, min_ratio=0.3):
        self.free_flow = free_flow
        self.min_ratio = min_ratio

    def get_capacity_ratio(self, speed):
        speed = np.asarray(speed, dtype=float)
        ratio = np.piecewise(speed,
            [speed < 5, (speed >= 5) & (speed < 10), (speed >= 10) & (speed < 20),
             (speed >= 20) & (speed < 40), speed >= 40],
            [self.min_ratio, lambda v: self.min_ratio + 0.1*(v-5)/5,
             lambda v: 0.4 + 0.2*(v-10)/10, lambda v: 0.5 + 0.2*(v-20)/20, 1.0])
        return ratio

    def get_effective_demand(self, speed, base_demand_per_region=1.0):
        ratio = self.get_capacity_ratio(speed)
        ratio = np.maximum(ratio, 0.01)
        return base_demand_per_region / ratio


# ==================== 2. MIP优化器（加权多目标） ====================
class CongestionAwareIPOptimizer:
    def __init__(self, num_regions, distance_matrix, speed_predictions, demand_model, beta=5.0):
        self.num_regions = num_regions
        self.distance_matrix = distance_matrix.astype(float)
        self.speed_predictions = speed_predictions
        self.demand_model = demand_model
        self.beta = beta

    def solve_rebalancing(self, current_supply, future_time_idx=0):
        future_speed = self.speed_predictions[future_time_idx].flatten()
        future_speed = np.clip(future_speed, 0, SPEED_CAP)

        effective_demand = self.demand_model.get_effective_demand(future_speed, BASE_DEMAND)
        current_supply_flat = np.array(current_supply).ravel().astype(float)

        surplus = np.maximum(0, current_supply_flat - effective_demand)
        deficit = np.maximum(0, effective_demand - current_supply_flat)

        THRESHOLD = 0.01
        if surplus.sum() < THRESHOLD or deficit.sum() < THRESHOLD:
            return [], 0.0, 0

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise RuntimeError("SCIP求解器初始化失败")

        surplus_idx = np.where(surplus > THRESHOLD)[0]
        deficit_idx = np.where(deficit > THRESHOLD)[0]

        X = {}
        for i in surplus_idx:
            for j in deficit_idx:
                if self.distance_matrix[i, j] < 500:
                    X[(i, j)] = solver.IntVar(0, int(np.ceil(surplus[i])), f'X_{i}_{j}')

        if len(X) == 0:
            return [], 0.0, 0

        # 🔴 加权多目标：最小化 [成本 + β × 未满足需求]
        # 未满足需求 = deficit_j - sum_i X_ij
        obj = solver.Objective()
        for (i, j), var in X.items():
            # 成本项
            obj.SetCoefficient(var, float(self.distance_matrix[i, j]))
        # 未满足需求惩罚：对每个缺口区域j，未满足部分 = deficit_j - sum_i X_ij
        # 在目标函数中体现为：-β * sum_i X_ij （因为 deficit_j 是常数）
        for j in deficit_idx:
            relevant = [X[(i, j)] for i in surplus_idx if (i, j) in X]
            if relevant:
                for var in relevant:
                    obj.SetCoefficient(var, obj.GetCoefficient(var) - self.beta)
        obj.SetMinimization()

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
        plan = []
        cost = 0.0
        total_moved = 0
        unsatisfied = deficit.sum()

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            for (i, j), var in X.items():
                val = int(var.solution_value())
                if val > 0:
                    plan.append((i, j, val))
                    cost += val * self.distance_matrix[i, j]
                    total_moved += val
                    unsatisfied -= val
            plan.sort(key=lambda x: -x[2])
        else:
            return self._greedy_fallback(surplus, deficit)

        return plan, cost, total_moved

    def _greedy_fallback(self, surplus, deficit):
        s_list, d_list = surplus.copy(), deficit.copy()
        plan, cost = [], 0.0
        for i in np.where(s_list > 0.01)[0]:
            if s_list[i] < 0.01: continue
            for j, dist in sorted([(j, self.distance_matrix[i, j]) 
                                    for j in np.where(d_list > 0.01)[0]], key=lambda x: x[1]):
                if s_list[i] < 0.01 or d_list[j] < 0.01: continue
                move = int(min(np.ceil(s_list[i]), np.ceil(d_list[j])))
                if move > 0:
                    plan.append((i, j, move)); cost += move * dist
                    s_list[i] -= move; d_list[j] -= move
        return plan, cost, sum(x[2] for x in plan)


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


# ==================== 5. 核心仿真引擎 ====================
class CongestionAwareSimulator:
    def __init__(self, road_network, pred_speeds, true_speeds, total_cars=260, base_demand=0.5, beta=5.0):
        self.road = road_network
        self.pred_speeds = pred_speeds
        self.true_speeds = true_speeds
        self.total_cars = total_cars
        self.n_regions = road_network.n
        self.base_demand = base_demand
        self.demand_model = CongestionAwareDemand()
        self.beta = beta
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

    def run_single_experiment(self, seed, time_idx, use_true_speed_for_pred=False):
        """
        use_true_speed_for_pred=True: Oracle模式（用真实速度做预测输入）
        use_true_speed_for_pred=False: 正常模式（用预测速度做输入）
        """
        supply = self._init_supply(seed)
        dm = self.road.generate_distance_matrix(seed)

        # 预测输入（决定调度方案）
        pred_demand = self.true_demands[time_idx] if use_true_speed_for_pred else self.pred_demands[time_idx]
        # 真实需求（用于评估服务率）
        true_demand = self.true_demands[time_idx]

        results = {}

        # 1. 无调度
        _, _, _, sr_no = DispatchStrategies.no_dispatch(supply, true_demand)
        results['no_dispatch'] = sr_no

        # 2. 随机（基于真实需求调度）
        _, _, _, sr_rand = DispatchStrategies.random_dispatch(supply, true_demand, dm, seed)
        results['random'] = sr_rand

        # 3. 就近（基于真实需求调度）
        _, _, _, sr_nn = DispatchStrategies.nearest_dispatch(supply, true_demand, dm)
        results['nearest'] = sr_nn

        # 4. Demand-Only（基于原始需求调度，用真实需求评估）
        s_do, _, _ = DispatchStrategies.demand_only_dispatch(supply, self.base_demand, dm)
        results['demand_only'] = DispatchStrategies.calculate_service_rate(s_do, true_demand)

        # 5. 拥堵感知MIP（基于预测需求调度，用真实需求评估）
        ip = CongestionAwareIPOptimizer(self.n_regions, dm, 
                                          self.true_speeds if use_true_speed_for_pred else self.pred_speeds,
                                          self.demand_model, self.beta)
        plan, cost, moved = ip.solve_rebalancing(supply, time_idx)
        s_ca = np.array(supply).ravel().astype(float)
        for i, j, amt in plan:
            s_ca[i] -= amt; s_ca[j] += amt
        results['congestion_aware'] = DispatchStrategies.calculate_service_rate(s_ca, true_demand)
        results['ca_cost'] = cost
        results['ca_moved'] = moved

        return results

    def run_multi_seed_multi_time(self, seeds, time_indices, label="STGCN"):
        """多种子 × 多时间片实验"""
        all_results = []
        for seed in seeds:
            seed_results = []
            for t in time_indices:
                r = self.run_single_experiment(seed, t, use_true_speed_for_pred=(label=="Oracle"))
                seed_results.append(r)
            all_results.append(seed_results)

        # 统计：对每个策略，计算跨时间片的平均服务率
        strategies = ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']
        summary = {}
        for strat in strategies:
            rates = []
            for seed_res in all_results:
                for t_res in seed_res:
                    rates.append(t_res[strat])
            summary[strat] = {'mean': np.mean(rates), 'std': np.std(rates), 'n': len(rates)}

        return all_results, summary


# ==================== 6. 主实验流程 ====================
def main():
    print("="*80)
    print("🚀 拥堵感知前置调度 —— 硕士论文完整实验 v6.0")
    print("="*80)

    # 加载数据
    road = RoadNetwork(REL_PATH, mode=DISTANCE_MODE)
    true_speeds = load_true_speeds(DYNAPATH)

    # 加载预测
    stgcn_raw = np.load(STGCN_PRED)["prediction"][:, 0, :]
    rnn_raw = np.load(RNN_PRED)["prediction"][:, 0, :]
    if len(stgcn_raw.shape) == 3: stgcn_raw = stgcn_raw.squeeze()
    if len(rnn_raw.shape) == 3: rnn_raw = rnn_raw.squeeze()

    print(f"\n📊 STGCN预测: {stgcn_raw.shape}, 范围[{stgcn_raw.min():.2f}, {stgcn_raw.max():.2f}]")
    print(f"📊 RNN预测: {rnn_raw.shape}, 范围[{rnn_raw.min():.2f}, {rnn_raw.max():.2f}]")
    print(f"📊 真实速度: {true_speeds.shape}, 范围[{true_speeds.min():.2f}, {true_speeds.max():.2f}]")

    # 对齐：取预测和真实速度的最后N_TIME_SLICES个时间片（假设是测试集末尾）
    n_pred = stgcn_raw.shape[0]
    n_true = true_speeds.shape[0]
    start_idx = max(0, n_true - n_pred)  # 对齐真实速度到预测
    true_aligned = true_speeds[start_idx:start_idx + n_pred]

    # 确保长度一致
    min_len = min(n_pred, true_aligned.shape[0])
    stgcn_pred = np.clip(stgcn_raw[:min_len], 0, SPEED_CAP)
    rnn_pred = np.clip(rnn_raw[:min_len], 0, SPEED_CAP)
    true_aligned = true_aligned[:min_len]
    time_indices = list(range(min(N_TIME_SLICES, min_len)))

    print(f"\n📊 对齐后: {min_len}个时间片, 使用前{len(time_indices)}个做滚动实验")

    # ========== 实验1: 模型对比 (STGCN vs RNN vs Oracle) ==========
    if RUN_MODEL_COMPARISON:
        print(f"\n{'='*80}")
        print("📊 实验1: 预测模型对比 (STGCN vs RNN vs Oracle)")
        print(f"{'='*80}")

        results_comparison = {}

        for label, pred_speeds in [("STGCN", stgcn_pred), ("RNN", rnn_pred), ("Oracle", true_aligned)]:
            print(f"\n🔬 运行 {label} 驱动调度...")
            sim = CongestionAwareSimulator(road, pred_speeds, true_aligned, 
                                           total_cars=260, base_demand=BASE_DEMAND, beta=BETA)
            _, summary = sim.run_multi_seed_multi_time(RANDOM_SEEDS, time_indices, label=label)
            results_comparison[label] = summary

            print(f"\n{label} 结果:")
            for strat in ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']:
                s = summary[strat]
                print(f"  {strat}: {s['mean']:.1%} ± {s['std']:.1%} (n={s['n']})")

        # 打印对比表
        print(f"\n{'='*80}")
        print("📊 模型对比汇总表")
        print(f"{'='*80}")
        print(f"{'策略':<20} {'STGCN':<15} {'RNN':<15} {'Oracle':<15}")
        print("-"*65)
        for strat in ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']:
            vals = [results_comparison[label][strat]['mean'] for label in ['STGCN', 'RNN', 'Oracle']]
            print(f"{strat:<20} {vals[0]:<14.1%} {vals[1]:<14.1%} {vals[2]:<14.1%}")

        # 关键结论
        ca_stgcn = results_comparison['STGCN']['congestion_aware']['mean']
        ca_rnn = results_comparison['RNN']['congestion_aware']['mean']
        ca_oracle = results_comparison['Oracle']['congestion_aware']['mean']
        print(f"\n🎉 关键结论:")
        print(f"  拥堵感知MIP (STGCN驱动): {ca_stgcn:.1%}")
        print(f"  拥堵感知MIP (RNN驱动):   {ca_rnn:.1%}")
        print(f"  拥堵感知MIP (Oracle):    {ca_oracle:.1%}")
        print(f"  STGCN vs RNN差距: {ca_stgcn - ca_rnn:+.1%}")
        print(f"  预测误差损失: {ca_oracle - ca_stgcn:+.1%} (Oracle - STGCN)")

    # ========== 实验2: 车辆数敏感性 ==========
    if RUN_SENSITIVITY:
        print(f"\n{'='*80}")
        print("📊 实验2: 车辆数敏感性分析")
        print(f"{'='*80}")

        for cars in TOTAL_CARS_LIST:
            print(f"\n🔬 总车辆数 = {cars}")
            sim = CongestionAwareSimulator(road, stgcn_pred, true_aligned, 
                                           total_cars=cars, base_demand=BASE_DEMAND, beta=BETA)
            _, summary = sim.run_multi_seed_multi_time(RANDOM_SEEDS[:5], time_indices[:5], label="STGCN")
            ca = summary['congestion_aware']['mean']
            nn = summary['nearest']['mean']
            print(f"  拥堵感知MIP: {ca:.1%} | 就近调度: {nn:.1%} | 提升: {ca-nn:+.1%}")

    # ========== 实验3: β敏感性 ==========
    if RUN_BETA_SWEEP:
        print(f"\n{'='*80}")
        print("📊 实验3: 未满足需求惩罚系数β敏感性")
        print(f"{'='*80}")

        for beta in BETA_LIST:
            print(f"\n🔬 β = {beta}")
            sim = CongestionAwareSimulator(road, stgcn_pred, true_aligned, 
                                           total_cars=260, base_demand=BASE_DEMAND, beta=beta)
            _, summary = sim.run_multi_seed_multi_time(RANDOM_SEEDS[:5], time_indices[:5], label="STGCN")
            ca = summary['congestion_aware']['mean']
            cost = np.mean([r['ca_cost'] for seed_res in _ for r in seed_res if 'ca_cost' in r]) if _ else 0
            print(f"  服务率: {ca:.1%} | 平均成本: {cost:.1f}")

    print(f"\n{'='*80}")
    print("✅ 全部实验完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
