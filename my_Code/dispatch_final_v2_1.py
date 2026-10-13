"""
============================================================
📦 拥堵感知前置调度实验系统 —— 最终学术版 v2.1 (Bug修复)
============================================================
修复：
1. OR-Tools Objective.SetCoefficient 需要 float() 转换
2. 拓扑分析：检查.rel文件的真实结构
3. 如果最大跳数=1（全连接），距离矩阵退化为均匀距离
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

TOTAL_CARS = 260
BASE_DEMAND = 0.5
SPEED_CAP = 86.43

N_SEEDS = 15
RANDOM_SEEDS = [42, 100, 2025, 7, 19, 88, 123, 456, 789, 1024,
                2048, 333, 777, 999, 1111]


# ==================== 0. 拓扑分析工具 ====================
def analyze_rel_topology(rel_path):
    """分析.rel文件的真实结构"""
    rel = pd.read_csv(rel_path)
    print("\n" + "="*60)
    print("📊 .rel 文件拓扑分析")
    print("="*60)
    print(f"总行数: {len(rel)}")
    print(f"\nlink_weight 分布:")
    print(rel['link_weight'].value_counts().sort_index())

    # 自环分析
    self_loops = rel[rel['origin_id'] == rel['destination_id']]
    print(f"\n自环数量: {len(self_loops)}")
    if len(self_loops) > 0:
        print(f"自环link_weight分布: {self_loops['link_weight'].value_counts().to_dict()}")

    # 非自环边
    non_self = rel[rel['origin_id'] != rel['destination_id']]
    print(f"\n非自环边: {len(non_self)}")
    print(f"  link_weight=0: {len(non_self[non_self.link_weight==0])}")
    print(f"  link_weight=1: {len(non_self[non_self.link_weight==1])}")

    # 检查每个节点的出度
    out_degrees = non_self.groupby('origin_id').size()
    print(f"\n节点出度统计:")
    print(f"  平均出度: {out_degrees.mean():.1f}")
    print(f"  最大出度: {out_degrees.max()}")
    print(f"  最小出度: {out_degrees.min()}")
    print(f"  唯一节点数: {len(out_degrees)}")

    # 检查是否全连接（每个节点连接所有其他节点）
    n_nodes = len(set(rel['origin_id']) | set(rel['destination_id']))
    expected_full = n_nodes * (n_nodes - 1)  # 有向全连接
    print(f"\n全连接判断:")
    print(f"  节点数: {n_nodes}")
    print(f"  有向全连接期望边数: {expected_full}")
    print(f"  实际非自环边数: {len(non_self)}")
    print(f"  是否接近全连接: {'是' if len(non_self) >= expected_full * 0.9 else '否'}")

    # 看看link_weight=1的边有什么特征
    lw1 = non_self[non_self.link_weight == 1]
    if len(lw1) > 0:
        print(f"\nlink_weight=1 的边分析:")
        print(f"  数量: {len(lw1)}")
        # 检查是否对称
        pairs = set()
        for _, row in lw1.iterrows():
            pairs.add((min(row['origin_id'], row['destination_id']),
                      max(row['origin_id'], row['destination_id'])))
        print(f"  去重后: {len(pairs)} 对")
        print(f"  示例: {list(pairs)[:5]}")

    print("="*60)
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


# ==================== 2. MIP优化器（修复float转换） ====================
class CongestionAwareIPOptimizer:
    def __init__(self, num_regions, distance_matrix, speed_predictions, demand_model):
        self.num_regions = num_regions
        self.distance_matrix = distance_matrix.astype(float)  # 确保是float64
        self.speed_predictions = speed_predictions
        self.demand_model = demand_model

    def solve_rebalancing(self, current_supply, future_time_idx=0):
        future_speed = self.speed_predictions[future_time_idx].flatten()
        future_speed = np.clip(future_speed, 0, SPEED_CAP)

        effective_demand = self.demand_model.get_effective_demand(future_speed, BASE_DEMAND)
        current_supply_flat = np.array(current_supply).ravel().astype(float)

        surplus = np.maximum(0, current_supply_flat - effective_demand)
        deficit = np.maximum(0, effective_demand - current_supply_flat)

        print(f"   [MIP] 速度均值{future_speed.mean():.1f} | 总需求{effective_demand.sum():.1f} | "
              f"盈余{np.sum(surplus>0.5)}区 缺口{np.sum(deficit>0.5)}区")

        if surplus.sum() < 0.5 or deficit.sum() < 0.5:
            return [], 0.0

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise RuntimeError("SCIP求解器初始化失败")

        surplus_idx = np.where(surplus >= 0.5)[0]
        deficit_idx = np.where(deficit >= 0.5)[0]

        # 决策变量
        X = {}
        for i in surplus_idx:
            for j in deficit_idx:
                if self.distance_matrix[i, j] < 200:
                    X[(i, j)] = solver.IntVar(0, int(np.ceil(surplus[i])), f'X_{i}_{j}')

        # 目标：最小化总调度成本（修复：显式转float）
        obj = solver.Objective()
        for (i, j), var in X.items():
            obj.SetCoefficient(var, float(self.distance_matrix[i, j]))  # 🔴 修复：加float()
        obj.SetMinimization()

        # 约束1
        for i in surplus_idx:
            relevant = [X[(i, j)] for j in deficit_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.floor(surplus[i])))

        # 约束2
        for j in deficit_idx:
            relevant = [X[(i, j)] for i in surplus_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.floor(deficit[j])))

        status = solver.Solve()
        plan = []
        cost = 0.0

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            for (i, j), var in X.items():
                val = int(var.solution_value())
                if val > 0:
                    plan.append((i, j, val))
                    cost += val * self.distance_matrix[i, j]
            plan.sort(key=lambda x: -x[2])
            print(f"   [MIP] ✅ 求解成功, {len(plan)}条路线, 总成本{cost:.1f}")
        else:
            print(f"   [MIP] ❌ 求解失败({status}), 回退贪心")
            return self._greedy_fallback(surplus, deficit)

        return plan, cost

    def _greedy_fallback(self, surplus, deficit):
        surplus_regions = np.where(surplus >= 0.5)[0]
        deficit_regions = np.where(deficit >= 0.5)[0]
        plan = []
        cost = 0.0
        s_list = surplus.copy()
        d_list = deficit.copy()

        for i in sorted(surplus_regions, key=lambda x: -s_list[x]):
            if s_list[i] < 0.5:
                continue
            candidates = [(j, self.distance_matrix[i, j])
                          for j in deficit_regions if d_list[j] >= 0.5]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])

            for j, dist in candidates:
                if s_list[i] < 0.5 or d_list[j] < 0.5:
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
    def __init__(self, rel_path: str, dist_matrix_path: str = None):
        rel = pd.read_csv(rel_path)

        all_ids = sorted(set(rel["origin_id"]) | set(rel["destination_id"]))
        self.nodes = all_ids
        self.n = len(all_ids)
        self.id2idx = {nid: i for i, nid in enumerate(all_ids)}

        print(f"✅ 路网加载: {self.n} 个区域")

        # 使用所有非自环边
        edges = rel[rel["origin_id"] != rel["destination_id"]]
        print(f"   非自环边: {len(edges)} (lw=0: {len(edges[edges.link_weight==0])}, "
              f"lw=1: {len(edges[edges.link_weight==1])})")

        edge_pairs = set()
        for _, row in edges.iterrows():
            i, j = row["origin_id"], row["destination_id"]
            edge_pairs.add((min(i, j), max(i, j)))

        print(f"   去重无向边: {len(edge_pairs)}")

        if dist_matrix_path is None:
            dist_matrix_path = os.path.join(os.path.dirname(rel_path), "sz_taxi_dist_matrix_final.npy")

        if os.path.exists(dist_matrix_path):
            self._dist_matrix = np.load(dist_matrix_path).astype(np.float64)  # float64
            print(f"✅ 距离矩阵已加载: {dist_matrix_path}")
        else:
            print(f"⚠️  计算距离矩阵...")
            self._dist_matrix = self._compute_distance_matrix(edge_pairs)
            np.save(dist_matrix_path, self._dist_matrix)
            print(f"✅ 已保存: {dist_matrix_path}")

    def _compute_distance_matrix(self, edge_pairs):
        n = self.n
        adj = np.zeros((n, n), dtype=np.float32)

        for i_id, j_id in edge_pairs:
            i = self.id2idx[i_id]
            j = self.id2idx[j_id]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        sparse_adj = sp.csr_matrix(adj)

        n_comp, labels = csg.connected_components(sparse_adj, directed=False)
        print(f"   连通分量: {n_comp}")
        if n_comp > 1:
            sizes = np.bincount(labels)
            print(f"   分量大小: {sorted(sizes, reverse=True)}")

        dist = csg.shortest_path(sparse_adj, method="D", directed=False, unweighted=True)

        # 检查跳数分布
        finite_dist = dist[np.isfinite(dist) & (dist > 0)]
        if len(finite_dist) > 0:
            print(f"   跳数分布: min={finite_dist.min():.0f}, max={finite_dist.max():.0f}, "
                  f"mean={finite_dist.mean():.2f}")

        finite_max = dist[np.isfinite(dist)].max() if np.any(np.isfinite(dist)) else 10
        penalty = int(finite_max * 2) + 10
        dist[np.isinf(dist)] = penalty
        np.fill_diagonal(dist, 0)

        n_inf = np.sum(dist >= penalty)
        print(f"   不可达填充: {penalty}, 不可达对: {n_inf}")

        return dist.astype(np.float64)  # 🔴 用float64

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
        s_regions = list(np.where(surplus > 0.1)[0])
        d_regions = list(np.where(deficit > 0.1)[0])
        np.random.shuffle(s_regions)

        for i in s_regions:
            if surplus[i] < 0.1 or not d_regions:
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
                if deficit[j] < 0.1:
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
            if surplus[i] < 0.1:
                continue
            candidates = [(j, distance_matrix[i, j]) for j in range(len(s)) if deficit[j] > 0.1]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])

            for j, dist in candidates:
                if surplus[i] < 0.1 or deficit[j] < 0.1:
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
            if surplus[i] < 0.1:
                continue
            candidates = [(j, distance_matrix[i, j]) for j in range(len(s)) if deficit[j] > 0.1]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])

            for j, dist in candidates:
                if surplus[i] < 0.1 or deficit[j] < 0.1:
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
    print("🚀 拥堵感知前置调度 —— 最终学术版 v2.1")
    print("="*80)

    # 先分析拓扑
    analyze_rel_topology(REL_PATH)

    # 加载路网
    road = RoadNetwork(REL_PATH)

    # 加载预测
    pred_data = np.load(PRED_PATH)
    raw_pred = pred_data["prediction"][:, 0, :]
    if len(raw_pred.shape) == 3:
        raw_pred = raw_pred.squeeze()

    print(f"\n📊 STGCN原始: 范围[{raw_pred.min():.2f}, {raw_pred.max():.2f}], 均值{raw_pred.mean():.2f}")
    speed_predictions = np.clip(raw_pred, 0, SPEED_CAP)
    print(f"📊 后处理后: 范围[{speed_predictions.min():.2f}, {speed_predictions.max():.2f}], 均值{speed_predictions.mean():.2f}")

    # 运行
    simulator = CongestionAwareSimulator(road, speed_predictions, total_cars=TOTAL_CARS, base_demand=BASE_DEMAND)
    all_results, summary = simulator.run_multiple_seeds(RANDOM_SEEDS, time_idx=0)

    print("\n✅ 实验完成！")
    return all_results, summary


if __name__ == "__main__":
    main()
