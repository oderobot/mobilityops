"""
============================================================
📦 拥堵感知前置调度实验系统 —— 最终学术版 v2.0
============================================================
基于真实数据集统计重校准：
- 训练集速度均值12.2, 95分位数36.5 → 自由流速度设为40
- .rel使用所有非自环边(link_weight=0和1都用)
- STGCN预测后处理: ReLU + 上限86.43
- 真正的MIP求解 + 统计检验 + 消融实验
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

# 实验参数（基于数据集统计重校准）
TOTAL_CARS = 260          # 车辆数
BASE_DEMAND = 0.5         # 基础需求/区域
FREE_FLOW_SPEED = 40.0    # 自由流速度 = 训练集95分位数
JAM_SPEED = 0.0           # 完全拥堵速度
MIN_CAPACITY_RATIO = 0.3  # 最低运力保留
SPEED_CAP = 86.43         # STGCN预测上限 = 训练集最大值

N_SEEDS = 15
RANDOM_SEEDS = [42, 100, 2025, 7, 19, 88, 123, 456, 789, 1024,
                2048, 333, 777, 999, 1111]


# ==================== 1. 容量衰减模型（数据集自适应） ====================
class CongestionAwareDemand:
    """
    基于数据集统计的分段线性模型。

    训练集速度特征：
    - 均值12.2, 中位数9.5, 95%分位数36.5
    - 说明大部分区域处于低速状态

    模型设计：
    - v >= 40: 运力100% (自由流，基于95分位数)
    - 20 <= v < 40: 运力70%-100% (缓行)
    - 10 <= v < 20: 运力50%-70% (拥堵)
    - 5 <= v < 10: 运力40%-50% (严重拥堵)
    - v < 5: 运力30% (瘫痪)
    """
    def __init__(self, free_flow=40.0, jam=0.0, min_ratio=0.3):
        self.free_flow = free_flow
        self.jam = jam
        self.min_ratio = min_ratio
        self.thresholds = [0, 5, 10, 20, 40]
        self.ratios = [0.3, 0.4, 0.5, 0.7, 1.0]

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


# ==================== 2. 真正的MIP优化器 ====================
class CongestionAwareIPOptimizer:
    def __init__(self, num_regions, distance_matrix, speed_predictions, demand_model):
        self.num_regions = num_regions
        self.distance_matrix = distance_matrix
        self.speed_predictions = speed_predictions
        self.demand_model = demand_model

    def solve_rebalancing(self, current_supply, future_time_idx=0):
        # 后处理速度
        future_speed = self.speed_predictions[future_time_idx].flatten()
        future_speed = np.clip(future_speed, 0, SPEED_CAP)

        # 等效需求
        effective_demand = self.demand_model.get_effective_demand(future_speed, BASE_DEMAND)
        current_supply_flat = np.array(current_supply).ravel().astype(float)

        surplus = np.maximum(0, current_supply_flat - effective_demand)
        deficit = np.maximum(0, effective_demand - current_supply_flat)

        print(f"   [MIP] 速度: [{future_speed.min():.1f}, {future_speed.max():.1f}], "
              f"均值{future_speed.mean():.1f} | 总需求{effective_demand.sum():.1f}")
        print(f"   [MIP] 盈余区域{np.sum(surplus>0.5)}/{self.num_regions}, "
              f"缺口区域{np.sum(deficit>0.5)}/{self.num_regions}")

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
                if self.distance_matrix[i, j] < 200:  # 排除极端惩罚值
                    X[(i, j)] = solver.IntVar(0, int(np.ceil(surplus[i])), f'X_{i}_{j}')

        # 目标：最小化总调度成本
        obj = solver.Objective()
        for (i, j), var in X.items():
            obj.SetCoefficient(var, self.distance_matrix[i, j])
        obj.SetMinimization()

        # 约束1：流出<=盈余
        for i in surplus_idx:
            relevant = [X[(i, j)] for j in deficit_idx if (i, j) in X]
            if relevant:
                solver.Add(sum(relevant) <= int(np.floor(surplus[i])))

        # 约束2：流入<=缺口
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
            return self._greedy_fallback(surplus, deficit, future_speed)

        return plan, cost

    def _greedy_fallback(self, surplus, deficit, future_speed):
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


# ==================== 3. 路网（使用所有非自环边） ====================
class RoadNetwork:
    def __init__(self, rel_path: str, dist_matrix_path: str = None):
        rel = pd.read_csv(rel_path)

        # 所有节点
        all_ids = sorted(set(rel["origin_id"]) | set(rel["destination_id"]))
        self.nodes = all_ids
        self.n = len(all_ids)
        self.id2idx = {nid: i for i, nid in enumerate(all_ids)}

        print(f"✅ 路网加载: {self.n} 个区域")

        # 🔴 关键：使用所有非自环边（link_weight=0和1都用）
        edges = rel[rel["origin_id"] != rel["destination_id"]]
        print(f"   非自环边数: {len(edges)} (link_weight=0: {len(edges[edges.link_weight==0])}, "
              f"link_weight=1: {len(edges[edges.link_weight==1])})")

        # 去重为无向边
        edge_pairs = set()
        for _, row in edges.iterrows():
            i, j = row["origin_id"], row["destination_id"]
            edge_pairs.add((min(i, j), max(i, j)))

        print(f"   去重无向边: {len(edge_pairs)}")

        # 距离矩阵路径
        if dist_matrix_path is None:
            dist_matrix_path = os.path.join(os.path.dirname(rel_path), "sz_taxi_dist_matrix_final.npy")

        if os.path.exists(dist_matrix_path):
            self._dist_matrix = np.load(dist_matrix_path).astype(np.float32)
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

        # 连通性
        n_comp, labels = csg.connected_components(sparse_adj, directed=False)
        print(f"   连通分量: {n_comp}")
        if n_comp > 1:
            sizes = np.bincount(labels)
            print(f"   分量大小: {sorted(sizes, reverse=True)}")

        # 最短路径
        dist = csg.shortest_path(sparse_adj, method="D", directed=False, unweighted=True)

        finite_max = dist[np.isfinite(dist)].max() if np.any(np.isfinite(dist)) else 10
        penalty = int(finite_max * 2) + 10
        dist[np.isinf(dist)] = penalty
        np.fill_diagonal(dist, 0)

        n_inf = np.sum(dist >= penalty)
        print(f"   最大跳数: {finite_max:.0f}, 不可达填充: {penalty}, 不可达对: {n_inf}")

        return dist.astype(np.float32)

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
        """消融基线：只用原始需求调度"""
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

        print(f"\n🌱 种子{seed} | 速度均值{speeds.mean():.1f} | "
              f"总需求{congestion_demand.sum():.1f} | 天花板{self.total_cars/congestion_demand.sum():.1%}")

        results = {}

        # 1. 无调度
        _, _, _, sr_no = DispatchStrategies.no_dispatch(supply, congestion_demand)
        results['no_dispatch'] = {'service_rate': sr_no, 'move': 0, 'distance': 0}

        # 2. 随机
        _, move_r, dist_r, sr_r = DispatchStrategies.random_dispatch(supply, congestion_demand, distance_matrix, seed)
        results['random'] = {'service_rate': sr_r, 'move': move_r, 'distance': dist_r}

        # 3. 就近
        _, move_n, dist_n, sr_n = DispatchStrategies.nearest_dispatch(supply, congestion_demand, distance_matrix)
        results['nearest'] = {'service_rate': sr_n, 'move': move_n, 'distance': dist_n}

        # 4. Demand-Only消融
        s_do, move_do, dist_do = DispatchStrategies.demand_only_dispatch(supply, self.base_demand, distance_matrix)
        sr_do = DispatchStrategies.calculate_service_rate(s_do, congestion_demand)
        results['demand_only'] = {'service_rate': sr_do, 'move': move_do, 'distance': dist_do}

        # 5. 拥堵感知MIP
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
    print("🚀 拥堵感知前置调度 —— 最终学术版 v2.0")
    print("="*80)

    # 1. 加载路网（使用所有非自环边）
    road = RoadNetwork(REL_PATH)

    # 2. 加载STGCN预测
    pred_data = np.load(PRED_PATH)
    raw_pred = pred_data["prediction"][:, 0, :]
    if len(raw_pred.shape) == 3:
        raw_pred = raw_pred.squeeze()

    print(f"\n📊 STGCN原始预测: 范围[{raw_pred.min():.2f}, {raw_pred.max():.2f}], 均值{raw_pred.mean():.2f}")

    # 后处理：ReLU + 上限（训练集最大值86.43）
    speed_predictions = np.clip(raw_pred, 0, SPEED_CAP)
    print(f"📊 后处理后: 范围[{speed_predictions.min():.2f}, {speed_predictions.max():.2f}], 均值{speed_predictions.mean():.2f}")

    # 3. 运行
    simulator = CongestionAwareSimulator(road, speed_predictions, total_cars=TOTAL_CARS, base_demand=BASE_DEMAND)
    all_results, summary = simulator.run_multiple_seeds(RANDOM_SEEDS, time_idx=0)

    print("\n✅ 实验完成！")
    return all_results, summary


if __name__ == "__main__":
    main()
