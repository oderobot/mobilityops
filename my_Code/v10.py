"""
============================================================
📦 拥堵感知前置调度实验系统 —— v10.0
============================================================
修改说明：
1. 记录 seed/time_idx，支持按 seed 聚合统计
2. 增加 IP 求解时间计时
3. 增加 λ 敏感性分析实验
4. 增加规范统计检验（Shapiro-Wilk, paired t-test, Cohen's d）
5. 参数命令行化，避免硬编码
"""

import os
import numpy as np
import pandas as pd
import warnings
import scipy.sparse.csgraph as csg
import scipy.sparse as sp
from ortools.linear_solver import pywraplp
from scipy import stats
import time
import argparse

warnings.filterwarnings('ignore')


# ==================== 配置区（命令行参数化） ====================
def get_config():
    parser = argparse.ArgumentParser(description='Congestion-Aware Rebalancing v10')
    parser.add_argument('--base_demand', type=float, default=0.5)
    parser.add_argument('--source_penalty', type=float, default=3.0)
    parser.add_argument('--total_cars', type=int, default=260)
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--n_time_slices', type=int, default=10)
    parser.add_argument('--max_dist', type=int, default=500)
    parser.add_argument('--lambda_scan', action='store_true', help='Run λ sensitivity')
    parser.add_argument('--no_stat_test', action='store_true', help='Skip statistical tests')
    args = parser.parse_args()
    return args


CONFIG = None

PROJECT_ROOT = "/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity"
REL_PATH = os.path.join(PROJECT_ROOT, "raw_data/SZ_TAXI/SZ_TAXI.rel")
DYNAPATH = os.path.join(PROJECT_ROOT, "raw_data/SZ_TAXI/SZ_TAXI.dyna")

STGCN_PRED = os.path.join(PROJECT_ROOT,
                          "libcity/cache/STGCN-ROUND1/evaluate_cache/2026_06_02_19_52_43_STGCN_SZ_TAXI_predictions.npz")
RNN_PRED = os.path.join(PROJECT_ROOT,
                        "libcity/cache/RNN-ROUND1/evaluate_cache/2026_06_02_20_15_08_RNN_SZ_TAXI_predictions.npz")

SPEED_CAP = 86.43
RANDOM_SEEDS = [42, 100, 2025, 7, 19, 88, 123, 456, 789, 1024]


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
                            [self.min_ratio, lambda v: self.min_ratio + 0.1 * (v - 5) / 5,
                             lambda v: 0.4 + 0.2 * (v - 10) / 10, lambda v: 0.5 + 0.2 * (v - 20) / 20, 1.0])

    def get_effective_demand(self, speed, base_demand_per_region=1.0):
        ratio = self.get_capacity_ratio(speed)
        return base_demand_per_region / np.maximum(ratio, 0.01)


# ==================== 2. MIP优化器（v10：增加求解计时） ====================
class CongestionAwareIPOptimizer:
    def __init__(self, num_regions, distance_matrix, speed_predictions, demand_model, source_penalty=3.0):
        self.num_regions = num_regions
        self.distance_matrix = distance_matrix.astype(float)
        self.speed_predictions = speed_predictions
        self.demand_model = demand_model
        self.source_penalty = source_penalty

    def solve_rebalancing(self, current_supply, future_time_idx=0):
        future_speed = np.clip(self.speed_predictions[future_time_idx].flatten(), 0, SPEED_CAP)
        effective_demand = self.demand_model.get_effective_demand(future_speed, CONFIG.base_demand)
        current_supply_flat = np.array(current_supply).ravel().astype(float)

        surplus = np.maximum(0, current_supply_flat - effective_demand)
        deficit = np.maximum(0, effective_demand - current_supply_flat)

        if surplus.sum() < 0.01 or deficit.sum() < 0.01:
            return [], 0.0, 0, 0.0

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise RuntimeError("SCIP求解器初始化失败")

        surplus_idx = np.where(surplus > 0.01)[0]
        deficit_idx = np.where(deficit > 0.01)[0]

        X = {}
        for i in surplus_idx:
            for j in deficit_idx:
                if self.distance_matrix[i, j] < CONFIG.max_dist:
                    X[(i, j)] = solver.IntVar(0, int(np.ceil(surplus[i])), f'X_{i}_{j}')

        if len(X) == 0:
            return [], 0.0, 0, 0.0

        obj = solver.Objective()
        for (i, j), var in X.items():
            reward = 1.0
            penalty = max(0.0, 1.0 - surplus[i]) * self.source_penalty
            net = reward - penalty
            obj.SetCoefficient(var, net)
        obj.SetMaximization()

        for i in surplus_idx:
            relevant = [X[(i, j)] for j in deficit_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.ceil(surplus[i])))
        for j in deficit_idx:
            relevant = [X[(i, j)] for i in surplus_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.ceil(deficit[j])))

        start_t = time.time()
        status = solver.Solve()
        solve_time = time.time() - start_t

        plan, cost, total_moved = [], 0.0, 0

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            for (i, j), var in X.items():
                val = int(var.solution_value())
                if val > 0:
                    plan.append((i, j, val))
                    cost += val * self.distance_matrix[i, j]
                    total_moved += val
            plan.sort(key=lambda x: -x[2])

        return plan, cost, total_moved, solve_time


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
                move += m;
                dist += m * dm[i, j]
                surplus[i] -= m;
                deficit[j] -= m;
                s[i] -= m;
                s[j] += m
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
                    move += m;
                    dist += m * dst
                    surplus[i] -= m;
                    deficit[j] -= m;
                    s[i] -= m;
                    s[j] += m
        return s, move, dist, DispatchStrategies.calculate_service_rate(s, d)

    @staticmethod
    def demand_only_dispatch(supply, base_demand, dm):
        s = np.array(supply).ravel().astype(float)
        d = np.full_like(s, base_demand)
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
                    move += m;
                    dist += m * dst
                    surplus[i] -= m;
                    deficit[j] -= m;
                    s[i] -= m;
                    s[j] += m
        return s, move, dist


# ==================== 5. 仿真器（v10：记录seed/time，seed-level聚合） ====================
class CongestionAwareSimulator:
    def __init__(self, road_network, pred_speeds, true_speeds, total_cars=None, base_demand=None, source_penalty=None):
        self.road = road_network
        self.pred_speeds = pred_speeds
        self.true_speeds = true_speeds
        self.total_cars = total_cars if total_cars is not None else CONFIG.total_cars
        self.base_demand = base_demand if base_demand is not None else CONFIG.base_demand
        self.n_regions = road_network.n
        self.demand_model = CongestionAwareDemand()
        self.source_penalty = source_penalty if source_penalty is not None else CONFIG.source_penalty
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
        results['seed'] = seed
        results['time_idx'] = time_idx
        results['no_dispatch'] = DispatchStrategies.calculate_service_rate(supply, true_demand)

        _, _, _, sr_rand = DispatchStrategies.random_dispatch(supply, true_demand, dm, seed)
        results['random'] = sr_rand

        _, _, _, sr_nn = DispatchStrategies.nearest_dispatch(supply, true_demand, dm)
        results['nearest'] = sr_nn

        s_do, _, _ = DispatchStrategies.demand_only_dispatch(supply, self.base_demand, dm)
        results['demand_only'] = DispatchStrategies.calculate_service_rate(s_do, true_demand)

        ip = CongestionAwareIPOptimizer(self.n_regions, dm, self.pred_speeds, self.demand_model, self.source_penalty)
        plan, cost, moved, solve_time = ip.solve_rebalancing(supply, time_idx)
        results['solve_time_ms'] = solve_time * 1000
        s_ca = np.array(supply).ravel().astype(float)
        for i, j, amt in plan:
            s_ca[i] -= amt;
            s_ca[j] += amt
        results['congestion_aware'] = DispatchStrategies.calculate_service_rate(s_ca, true_demand)
        results['moved'] = moved
        results['cost'] = cost

        return results

    def run_multi_seed_multi_time(self, seeds, time_indices):
        all_results = []
        solve_times = []
        for seed in seeds:
            for t in time_indices:
                r = self.run_single_experiment(seed, t)
                all_results.append(r)
                solve_times.append(r['solve_time_ms'])

        strategies = ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']
        summary = {}
        for strat in strategies:
            rates = [r[strat] for r in all_results]
            summary[strat] = {'mean': np.mean(rates), 'std': np.std(rates), 'n': len(rates)}

        # v10: 按seed聚合，计算seed-level mean和SE
        seed_summary = {}
        for strat in strategies:
            seed_means = []
            for seed in seeds:
                seed_rates = [r[strat] for r in all_results if r['seed'] == seed]
                seed_means.append(np.mean(seed_rates))
            seed_summary[strat] = {
                'mean': np.mean(seed_means),
                'se': np.std(seed_means, ddof=1) / np.sqrt(len(seeds)),
                'std': np.std(seed_means, ddof=1),
                'seed_means': seed_means
            }

        print("\n📊 Seed-level statistics (N={} seeds):".format(len(seeds)))
        for strat in strategies:
            s = seed_summary[strat]
            print(f"  {strat:<20}: {s['mean']:.1%} ± {s['se']:.2%} (SE)")

        print(f"\n📊 IP求解时间: {np.mean(solve_times):.2f} ± {np.std(solve_times):.2f} ms (n={len(solve_times)})")

        return all_results, summary, seed_summary


# ==================== 6. 统计检验（v10新增） ====================
def statistical_test(seed_summary, strat_a, strat_b, name_a, name_b):
    a = np.array(seed_summary[strat_a]['seed_means'])
    b = np.array(seed_summary[strat_b]['seed_means'])
    diff = a - b

    _, p_norm = stats.shapiro(diff)
    t_stat, p_val = stats.ttest_rel(a, b)
    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0

    print(f"\n📊 {name_a} vs {name_b}")
    print(f"  Mean diff: {diff.mean():+.1%}")
    print(f"  Shapiro-Wilk p: {p_norm:.3f} ({'Normal' if p_norm > 0.05 else 'Non-normal'})")
    print(f"  Paired t-test: t={t_stat:+.2f}, df={len(diff) - 1}, p={p_val:.4f}")
    print(f"  Cohen's d = {cohens_d:.2f}")

    return {'t': t_stat, 'p': p_val, 'cohens_d': cohens_d, 'normal': p_norm > 0.05}


# ==================== 7. λ敏感性分析（v10新增） ====================
def lambda_sensitivity_experiment(road, true_speeds, pred_speeds, seeds, time_indices,
                                  lambdas=[1.0, 2.0, 3.0, 4.0, 5.0]):
    print("\n" + "=" * 80)
    print("📊 λ敏感性分析 (Source Penalty Weight)")
    print("=" * 80)
    results = {}
    for lam in lambdas:
        sim = CongestionAwareSimulator(road, pred_speeds, true_speeds, source_penalty=lam)
        _, summary, seed_summary = sim.run_multi_seed_multi_time(seeds, time_indices)
        results[lam] = {
            'seed_mean': seed_summary['congestion_aware']['mean'],
            'seed_se': seed_summary['congestion_aware']['se']
        }
        print(f"  λ={lam:<4}: OSR={results[lam]['seed_mean']:.1%} ± {results[lam]['seed_se']:.2%} (SE)")

    best_lam = max(results.keys(), key=lambda k: results[k]['seed_mean'])
    print(f"\n  最优λ = {best_lam}, OSR = {results[best_lam]['seed_mean']:.1%}")
    return results


# ==================== 8. 主实验 ====================
def main():
    global CONFIG
    CONFIG = get_config()

    print("=" * 80)
    print("🚀 拥堵感知前置调度 —— v10.0")
    print(f"   λ={CONFIG.source_penalty} | 车队={CONFIG.total_cars} | 需求={CONFIG.base_demand}")
    print(f"   种子数={CONFIG.n_seeds} | 时间片={CONFIG.n_time_slices}")
    print("=" * 80)

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

    n_time = min(CONFIG.n_time_slices, min_len)
    time_indices = list(range(n_time))
    seeds = RANDOM_SEEDS[:CONFIG.n_seeds]

    print(f"\n📊 STGCN: {stgcn_pred.shape}, MAE≈2.62 (step-1)")
    print(f"📊 RNN: {rnn_pred.shape}, MAE≈3.05 (step-1)")
    print(f"📊 实验规模: {n_time}时间片 × {len(seeds)}种子 = {n_time * len(seeds)}次")

    # 实验1: STGCN驱动
    print(f"\n{'=' * 80}")
    print("📊 实验1: STGCN预测驱动")
    print(f"{'=' * 80}")
    sim_stgcn = CongestionAwareSimulator(road, stgcn_pred, true_aligned)
    _, summary_stgcn, seed_stgcn = sim_stgcn.run_multi_seed_multi_time(seeds, time_indices)

    for strat in ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']:
        s = summary_stgcn[strat]
        print(f"  {strat}: {s['mean']:.1%} ± {s['std']:.1%} (n={s['n']})")

    # 实验2: RNN驱动
    print(f"\n{'=' * 80}")
    print("📊 实验2: RNN预测驱动")
    print(f"{'=' * 80}")
    sim_rnn = CongestionAwareSimulator(road, rnn_pred, true_aligned)
    _, summary_rnn, seed_rnn = sim_rnn.run_multi_seed_multi_time(seeds, time_indices)

    for strat in ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']:
        s = summary_rnn[strat]
        print(f"  {strat}: {s['mean']:.1%} ± {s['std']:.1%} (n={s['n']})")

    # 对比表
    print(f"\n{'=' * 80}")
    print("📊 STGCN vs RNN 端到端对比")
    print(f"{'=' * 80}")
    print(f"{'策略':<<20} {'STGCN':<<15} {'RNN':<<15} {'差距':<<15}")
    print("-" * 65)
    for strat in ['no_dispatch', 'random', 'nearest', 'demand_only', 'congestion_aware']:
        stgcn_val = summary_stgcn[strat]['mean']
        rnn_val = summary_rnn[strat]['mean']
        diff = stgcn_val - rnn_val
        print(f"{strat:<20} {stgcn_val:<14.1%} {rnn_val:<14.1%} {diff:+.1%}")

    # v10: 规范统计检验
    if not CONFIG.no_stat_test:
        print(f"\n{'=' * 80}")
        print("📊 规范统计检验 (seed-level, N=10, df=9)")
        print(f"{'=' * 80}")
        statistical_test(seed_stgcn, 'congestion_aware', 'demand_only', 'CA', 'DO')
        statistical_test(seed_stgcn, 'congestion_aware', 'random', 'CA', 'Random')
        statistical_test(seed_stgcn, 'congestion_aware', 'nearest', 'CA', 'Nearest')
        statistical_test(seed_stgcn, 'congestion_aware', 'no_dispatch', 'CA', 'No-Dispatch')

    # v10: λ敏感性分析
    if CONFIG.lambda_scan:
        lambda_sensitivity_experiment(road, true_aligned, stgcn_pred, seeds, time_indices)

    # 核心结论
    ca_stgcn = summary_stgcn['congestion_aware']['mean']
    ca_rnn = summary_rnn['congestion_aware']['mean']
    do_stgcn = summary_stgcn['demand_only']['mean']
    nn_stgcn = summary_stgcn['nearest']['mean']

    print(f"\n{'=' * 80}")
    print("🎉 核心结论")
    print(f"{'=' * 80}")
    print(f"  STGCN-MIP: {ca_stgcn:.1%}")
    print(f"  RNN-MIP:   {ca_rnn:.1%}")
    print(f"  Demand-Only: {do_stgcn:.1%}")
    print(f"  就近调度:    {nn_stgcn:.1%}")
    print(f"  拥堵感知 vs Demand-Only: {ca_stgcn - do_stgcn:+.1%}")
    print(f"  拥堵感知 vs 就近: {ca_stgcn - nn_stgcn:+.1%}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()