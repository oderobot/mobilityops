"""
完整的车辆调度工程管道 - 完全适配SZ_TAXI数据格式

执行流程：
1. 加载.rel（道路邻接）和.dyna（速度时序）
2. 构建道路网络图
3. 加载STGCN预测结果
4. 速度 → 容量权重转换
5. IP优化调度
6. 仿真对比验证

作者：Yan
日期：2026年6月
"""

import numpy as np
import pandas as pd
import networkx as nx
import logging
from typing import Tuple, Dict, List, Optional
from scipy.optimize import linprog
import os
import sys

# 尝试导入OR-Tools，如果失败则用LP降级
try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except:
    HAS_ORTOOLS = False
    print("⚠️ OR-Tools未安装，将使用scipy.linprog降级求解")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 模块1：数据加载（适配SZ_TAXI格式）
# ============================================================================

class SZTaxiDataLoader:
    """专门针对SZ_TAXI数据格式的加载器"""
    
    def __init__(self, rel_path: str, dyna_path: str):
        """
        初始化加载器
        
        Args:
            rel_path: SZ_TAXI.rel 文件路径
            dyna_path: SZ_TAXI.dyna 文件路径
        
        数据格式说明：
        .rel: rel_id, type, origin_id, destination_id, link_weight
        .dyna: dyna_id, type, time, entity_id, traffic_speed
        """
        self.rel_path = rel_path
        self.dyna_path = dyna_path
        self.df_rel = None
        self.df_dyna = None
        
        self._load_and_validate()
    
    def _load_and_validate(self):
        """加载并验证数据"""
        logger.info("=" * 60)
        logger.info("加载SZ_TAXI数据")
        logger.info("=" * 60)
        
        # 加载.rel文件
        try:
            self.df_rel = pd.read_csv(self.rel_path)
            logger.info(f"\n✓ 加载.rel文件")
            logger.info(f"  路径：{self.rel_path}")
            logger.info(f"  形状：{self.df_rel.shape}")
            logger.info(f"  列名：{list(self.df_rel.columns)}")
            logger.info(f"  样本：\n{self.df_rel.head(3)}")
        except Exception as e:
            logger.error(f"加载.rel文件失败：{e}")
            raise
        
        # 加载.dyna文件
        try:
            self.df_dyna = pd.read_csv(self.dyna_path)
            logger.info(f"\n✓ 加载.dyna文件")
            logger.info(f"  路径：{self.dyna_path}")
            logger.info(f"  形状：{self.df_dyna.shape}")
            logger.info(f"  列名：{list(self.df_dyna.columns)}")
            logger.info(f"  样本：\n{self.df_dyna.head(3)}")
            
            # 解析时间列
            self.df_dyna['time'] = pd.to_datetime(self.df_dyna['time'])
            logger.info(f"  时间范围：{self.df_dyna['time'].min()} 到 {self.df_dyna['time'].max()}")
            logger.info(f"  速度范围：{self.df_dyna['traffic_speed'].min():.1f} - {self.df_dyna['traffic_speed'].max():.1f} km/h")
        except Exception as e:
            logger.error(f"加载.dyna文件失败：{e}")
            raise
    
    def get_rel(self) -> pd.DataFrame:
        return self.df_rel
    
    def get_dyna(self) -> pd.DataFrame:
        return self.df_dyna


# ============================================================================
# 模块2：图构建与拓扑分析
# ============================================================================

class RoadNetworkGraph:
    """道路网络图"""
    
    def __init__(self, df_rel: pd.DataFrame, df_dyna: pd.DataFrame):
        """
        初始化图
        
        Args:
            df_rel: 关系数据框（包含origin_id, destination_id）
            df_dyna: 动态数据框（包含entity_id, traffic_speed）
        """
        self.df_rel = df_rel.copy()
        self.df_dyna = df_dyna.copy()
        
        self.G = None
        self.degree_centrality = None
        self.id_to_idx = None
        self.idx_to_id = None
        self.num_regions = None
        
        self._build_graph()
        self._compute_metrics()
    
    def _build_graph(self):
        """构建有向图"""
        logger.info("\n" + "=" * 60)
        logger.info("构建道路网络图")
        logger.info("=" * 60)
        
        # 从.rel构建图（移除自环）
        df_filtered = self.df_rel[
            self.df_rel['origin_id'] != self.df_rel['destination_id']
        ].copy()
        
        self.G = nx.from_pandas_edgelist(
            df_filtered,
            source='origin_id',
            target='destination_id',
            create_using=nx.DiGraph()
        )
        
        logger.info(f"\n✓ 图构建完成")
        logger.info(f"  节点数（路段）：{self.G.number_of_nodes()}")
        logger.info(f"  边数（连接）：{self.G.number_of_edges()}")
        
        # 创建映射
        all_node_ids = sorted(self.G.nodes())
        self.id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
        self.idx_to_id = {idx: node_id for node_id, idx in self.id_to_idx.items()}
        self.num_regions = len(all_node_ids)
        
        logger.info(f"  区域总数：{self.num_regions}")
        logger.info(f"  节点ID范围：{min(all_node_ids)} - {max(all_node_ids)}")
    
    def _compute_metrics(self):
        """计算拓扑指标"""
        logger.info("\n✓ 计算拓扑指标")
        
        # 度中心性
        self.degree_centrality = nx.degree_centrality(self.G)
        degree_values = list(self.degree_centrality.values())
        logger.info(f"  度中心性范围：{min(degree_values):.4f} - {max(degree_values):.4f}")
        
        # 速度统计
        if 'entity_id' in self.df_dyna.columns and 'traffic_speed' in self.df_dyna.columns:
            speed_mean = self.df_dyna.groupby('entity_id')['traffic_speed'].mean()
            speed_std = self.df_dyna.groupby('entity_id')['traffic_speed'].std()
            
            logger.info(f"  覆盖的路段数（有速度数据）：{len(speed_mean)}")
            logger.info(f"  平均速度：{speed_mean.mean():.1f} km/h")
            logger.info(f"  速度波动：{speed_std.mean():.1f} km/h (标准差)")
    
    def get_graph(self) -> nx.DiGraph:
        return self.G
    
    def get_mappings(self) -> Tuple[Dict, Dict]:
        return self.id_to_idx, self.idx_to_id
    
    def get_num_regions(self) -> int:
        return self.num_regions


# ============================================================================
# 模块3：STGCN预测加载
# ============================================================================

class STGCNPredictionLoader:
    """加载STGCN预测结果"""
    
    def __init__(self, npz_path: str):
        """
        Args:
            npz_path: .npz文件路径
        
        预期格式：
        - prediction: (batch, horizon, num_regions, 1) 预测速度
        - truth: (batch, horizon, num_regions, 1) 真实速度
        """
        self.npz_path = npz_path
        self.y_pred = None
        self.y_truth = None
        
        self._load()
    
    def _load(self):
        """加载预测文件"""
        logger.info("\n" + "=" * 60)
        logger.info("加载STGCN预测结果")
        logger.info("=" * 60)
        
        try:
            data = np.load(self.npz_path, allow_pickle=True)
            self.y_pred = data['prediction']
            self.y_truth = data['truth']
            
            logger.info(f"\n✓ 加载预测文件成功")
            logger.info(f"  文件路径：{self.npz_path}")
            logger.info(f"  预测形状：{self.y_pred.shape}")
            logger.info(f"    - 样本数（batch）：{self.y_pred.shape[0]}")
            logger.info(f"    - 预测步长（horizon）：{self.y_pred.shape[1]}")
            logger.info(f"    - 区域数（regions）：{self.y_pred.shape[2]}")
            logger.info(f"  真实值形状：{self.y_truth.shape}")
            
        except Exception as e:
            logger.error(f"加载预测文件失败：{e}")
            raise
    
    def get_predictions(self) -> np.ndarray:
        """返回预测值 (batch, horizon, regions, 1)"""
        return self.y_pred
    
    def get_first_step_predictions(self, num_samples: int = None) -> np.ndarray:
        """
        提取所有样本的第一个预测时间步
        
        Args:
            num_samples: 要提取的样本数（默认全部）
        
        Returns:
            (num_samples, num_regions) 速度预测矩阵
        """
        if num_samples is None:
            num_samples = self.y_pred.shape[0]
        else:
            num_samples = min(num_samples, self.y_pred.shape[0])
        
        # 提取第一步预测：shape (batch, 1, regions, 1) -> (batch, regions)
        first_step = self.y_pred[:num_samples, 0, :, 0]
        
        # 处理负值：将负速度设为0
        first_step = np.maximum(first_step, 0)

        logger.info(f"\n✓ 提取预测数据")
        logger.info(f"  提取样本数：{num_samples}")
        logger.info(f"  提取的数据形状：{first_step.shape}")
        logger.info(f"  速度范围：{first_step.min():.1f} - {first_step.max():.1f} km/h")

        return first_step


# ============================================================================
# 模块4：速度→容量转换
# ============================================================================

class CapacityWeightConverter:
    """将速度转换为容量权重"""

    def __init__(self, reference_speed: float = 30.0, min_capacity: float = 0.3):
        """
        Args:
            reference_speed: 参考速度（自由流速度），默认30 km/h
            min_capacity: 最小容量限制，防止某区域完全无容量

        转换逻辑：
        - 速度 = reference_speed → 容量 = 1.0（满容量）
        - 速度 = reference_speed/2 → 容量 = 0.5（半容量）
        - 速度 < reference_speed*0.3 → 容量 = 0.3（最小容量）
        """
        self.reference_speed = reference_speed
        self.min_capacity = min_capacity

        logger.info("\n" + "=" * 60)
        logger.info("初始化容量权重转换器")
        logger.info("=" * 60)
        logger.info(f"  参考速度（自由流）：{reference_speed} km/h")
        logger.info(f"  最小容量限制：{min_capacity}")

    def convert(self, speed_values: np.ndarray) -> np.ndarray:
        """
        速度 → 容量权重

        Args:
            speed_values: (num_regions,) 或 (batch, num_regions) 速度数组

        Returns:
            capacity: 同shape 的容量权重 [min_capacity, 1.0]
        """
        capacity = np.clip(
            speed_values / self.reference_speed,
            self.min_capacity,
            1.0
        )
        return capacity

    def get_priority_score(self, speed_values: np.ndarray) -> np.ndarray:
        """
        速度 → 调度优先级分数

        优先级 = 1 - 容量
        低速度(拥堵) → 低容量 → 高优先级 → 优先调度

        Returns:
            priority: 优先级分数 [0, 1]，高分优先调度
        """
        capacity = self.convert(speed_values)
        priority = 1.0 - capacity
        return priority


# ============================================================================
# 模块5：IP优化调度
# ============================================================================

class IPVehicleDispatcher:
    """整数规划车辆调度优化器"""

    def __init__(self, alpha: float = 0.7, time_limit: int = 60):
        """
        Args:
            alpha: 目标函数权衡系数
                  目标 = α*运力缺口 + (1-α)*调度成本
                  α=0.7: 优先缓解拥堵
                  α=0.3: 优先降低成本
            time_limit: 求解器超时（秒）
        """
        self.alpha = alpha
        self.time_limit = time_limit

        logger.info("\n" + "=" * 60)
        logger.info("初始化IP优化调度器")
        logger.info("=" * 60)
        logger.info(f"  权衡系数 α：{alpha}")
        logger.info(f"    - 运力缺口权重：{alpha}")
        logger.info(f"    - 调度成本权重：{1-alpha}")
        logger.info(f"  求解超时：{time_limit}秒")
        logger.info(f"  求解器：{'OR-Tools SCIP' if HAS_ORTOOLS else 'scipy.linprog (LP降级)'}")

    def optimize(self,
                 capacity_weight: np.ndarray,
                 current_supply: np.ndarray,
                 distance_matrix: np.ndarray,
                 baseline_demand: Optional[np.ndarray] = None) -> Dict:
        """
        主优化函数

        Args:
            capacity_weight: (num_regions,) 各区域容量权重 [0.3, 1.0]
            current_supply: (num_regions,) 各区域当前车数
            distance_matrix: (num_regions, num_regions) 区域间距离矩阵
            baseline_demand: (num_regions,) 基础需求，默认100

        Returns:
            {
                'dispatch_plan': (num_regions, num_regions) X[i][j]矩阵,
                'total_vehicles_moved': 调度的总车数,
                'status': 'OPTIMAL' | 'LP_RELAXATION' | 'FAILED'
            }
        """
        n = len(capacity_weight)

        if baseline_demand is None:
            baseline_demand = np.ones(n) * 100.0

        # 尝试用OR-Tools
        if HAS_ORTOOLS:
            try:
                return self._optimize_with_ortools(
                    capacity_weight, current_supply, distance_matrix, baseline_demand, n
                )
            except Exception as e:
                logger.warning(f"OR-Tools优化失败，降级到LP：{e}")
                return self._optimize_with_linprog(
                    capacity_weight, current_supply, distance_matrix, baseline_demand, n
                )
        else:
            return self._optimize_with_linprog(
                capacity_weight, current_supply, distance_matrix, baseline_demand, n
            )

    def _optimize_with_ortools(self, capacity_weight, current_supply, distance_matrix, baseline_demand, n):
        """使用OR-Tools SCIP求解器"""
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise Exception("SCIP求解器不可用")

        # ===== 决策变量：X[i][j] = 从区域i调度到j的车数（整数） =====
        X = {}
        for i in range(n):
            for j in range(n):
                X[i, j] = solver.IntVar(
                    0, int(current_supply[i]) + 1,
                    f'X_{i}_{j}'
                )

        # ===== 约束1：流量守恒 =====
        for i in range(n):
            outflow = solver.Sum(X[i, j] for j in range(n) if j != i)
            solver.Add(outflow <= current_supply[i], f'flow_{i}')

        # ===== 目标函数 =====
        # 有效需求 = 基础需求 / 容量权重
        # （拥堵降低容量，所以需要更多车来满足相同需求）
        effective_demand = baseline_demand / np.clip(capacity_weight, 0.3, 1.0)

        # 各区域调度后的供给
        supply_after = []
        for j in range(n):
            inflow = solver.Sum(X[i, j] for i in range(n) if i != j)
            outflow = solver.Sum(X[j, i] for i in range(n) if i != j)
            supply_j_after = current_supply[j] + inflow - outflow
            supply_after.append(supply_j_after)

        shortfall = []
        for j in range(n):
            Z = solver.NumVar(0, solver.infinity(), f"short_{j}")
            # Z ≥ 有效需求-调度后供给
            solver.Add(Z >= effective_demand[j] - supply_after[j])
            shortfall.append(Z)

        # 调度成本 = Σ X[i,j] * distance[i,j]
        relocation_cost = solver.Sum(
            X[i, j] * int(distance_matrix[i, j] * 10)
            for i in range(n) for j in range(n) if i != j
        )

        # 目标函数：加权组合
        total_shortfall = solver.Sum(shortfall)
        objective = (
            int(self.alpha * 10000) * total_shortfall +
            int((1 - self.alpha) * 10000) * relocation_cost
        )
        solver.Minimize(objective)

        # ===== 求解 =====
        solver.SetTimeLimit(self.time_limit * 1000)
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            dispatch_plan = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(n):
                    dispatch_plan[i, j] = int(X[i, j].solution_value())

            vehicles_moved = dispatch_plan.sum() - np.diag(dispatch_plan).sum()

            logger.info(f"\n✓ IP优化求解成功")
            logger.info(f"  目标函数值：{solver.ObjectiveValue():.0f}")
            logger.info(f"  调度车数：{vehicles_moved}")

            return {
                'dispatch_plan': dispatch_plan,
                'total_vehicles_moved': vehicles_moved,
                'status': 'OPTIMAL'
            }
        else:
            logger.warning(f"IP求解器未收敛（状态码{status}），返回None")
            return {
                'dispatch_plan': None,
                'total_vehicles_moved': 0,
                'status': f'FAILED_{status}'
            }

    def _optimize_with_linprog(self, capacity_weight, current_supply, distance_matrix, baseline_demand, n):
        """LP降级方案"""
        # 简化为单变量分配（按优先级分配车辆）
        priority = 1.0 - np.clip(capacity_weight / 30.0, 0.3, 1.0)

        c = -priority  # 负号因为linprog最小化

        A_eq = np.ones((1, n))
        b_eq = np.array([current_supply.sum()])

        bounds = [(0, current_supply[i]) for i in range(n)]

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            allocation = np.round(result.x).astype(int)
            dispatch_plan = np.diag(allocation)

            vehicles_moved = allocation.sum() - allocation[allocation > 0].sum()

            logger.info(f"\n✓ LP优化求解成功（降级方案）")
            logger.info(f"  目标值：{result.fun:.0f}")
            logger.info(f"  分配车数总和：{allocation.sum()}")

            return {
                'dispatch_plan': dispatch_plan,
                'total_vehicles_moved': vehicles_moved,
                'status': 'LP_RELAXATION'
            }
        else:
            logger.error("LP求解失败")
            return {
                'dispatch_plan': None,
                'total_vehicles_moved': 0,
                'status': 'FAILED'
            }


# ============================================================================
# 模块6：仿真与对比
# ============================================================================

class DispatchSimulator:
    """调度方法对比仿真器"""

    def __init__(self,
                 capacity_converter: CapacityWeightConverter,
                 dispatcher: IPVehicleDispatcher,
                 distance_matrix: np.ndarray):
        """初始化"""
        self.capacity_converter = capacity_converter
        self.dispatcher = dispatcher
        self.distance_matrix = distance_matrix

    def run_method(self,
                   speed_pred: np.ndarray,
                   current_supply: np.ndarray,
                   method: str = 'congestion_aware') -> Dict:
        """
        运行单个时间步仿真

        Args:
            speed_pred: (num_regions,) 预测速度
            current_supply: (num_regions,) 当前供给
            method: 'congestion_aware' | 'nearest_vehicle' | 'random'

        Returns:
            结果字典
        """
        n = len(speed_pred)

        if method == 'congestion_aware':
            # 你的方法：基于容量的IP优化
            capacity = self.capacity_converter.convert(speed_pred)
            result = self.dispatcher.optimize(
                capacity,
                current_supply,
                self.distance_matrix,
                baseline_demand=None
            )

            if result['dispatch_plan'] is None:
                return {
                    'method': method,
                    'success': False,
                    'avg_capacity': capacity.mean(),
                }

            return {
                'method': method,
                'success': True,
                'dispatch_plan': result['dispatch_plan'],
                'avg_capacity': capacity.mean(),
                'vehicles_moved': result['total_vehicles_moved'],
                'status': result['status']
            }

        elif method == 'nearest_vehicle':
            # Baseline：贪心分配（高优先级先分）
            capacity = self.capacity_converter.convert(speed_pred)
            priority = self.capacity_converter.get_priority_score(speed_pred)

            allocation = np.zeros(n)
            remaining = int(current_supply.sum())

            for idx in np.argsort(priority)[::-1]:
                allocate = min(int(priority[idx] * remaining), remaining)
                allocation[idx] = allocate
                remaining -= allocate

            dispatch_plan = np.diag(allocation.astype(int))

            return {
                'method': method,
                'success': True,
                'dispatch_plan': dispatch_plan,
                'avg_capacity': capacity.mean(),
                'vehicles_moved': 0,
                'status': 'BASELINE'
            }

        elif method == 'random':
            # Baseline：随机分配
            capacity = self.capacity_converter.convert(speed_pred)

            allocation = np.random.multinomial(
                int(current_supply.sum()),
                np.ones(n) / n
            )
            dispatch_plan = np.diag(allocation)

            return {
                'method': method,
                'success': True,
                'dispatch_plan': dispatch_plan,
                'avg_capacity': capacity.mean(),
                'vehicles_moved': 0,
                'status': 'BASELINE'
            }

        else:
            raise ValueError(f"Unknown method: {method}")

    def run_comparison(self,
                      speed_sequence: np.ndarray,
                      current_supplies: np.ndarray,
                      methods: List[str] = None,
                      verbose: bool = True) -> Dict[str, List]:
        """
        运行多方法、多时间步对比仿真

        Args:
            speed_sequence: (time_steps, num_regions) 速度预测序列
            current_supplies: (time_steps, num_regions) 或 (num_regions,) 供给
            methods: 要对比的方法列表
            verbose: 是否打印进度

        Returns:
            {
                'congestion_aware': [step_result_1, step_result_2, ...],
                'nearest_vehicle': [...],
                'random': [...]
            }
        """
        if methods is None:
            methods = ['congestion_aware', 'nearest_vehicle', 'random']

        time_steps = speed_sequence.shape[0]
        results = {method: [] for method in methods}

        logger.info("\n" + "=" * 60)
        logger.info(f"运行对比仿真：{time_steps}个时间步 × {len(methods)}种方法")
        logger.info("=" * 60)

        for t in range(time_steps):
            speed_t = speed_sequence[t]

            if current_supplies.ndim == 1:
                supply_t = current_supplies.copy()
            else:
                supply_t = current_supplies[t].copy()

            for method in methods:
                step_result = self.run_method(speed_t, supply_t, method)
                results[method].append(step_result)

            if verbose and (t + 1) % max(1, time_steps // 10) == 0:
                logger.info(f"  进度：{t + 1}/{time_steps}")

        logger.info(f"✓ 仿真完成")
        return results


# ============================================================================
# 模块7：结果分析
# ============================================================================

def analyze_results(results: Dict[str, List]) -> pd.DataFrame:
    """
    分析仿真结果

    Args:
        results: 仿真结果字典

    Returns:
        对比分析表格
    """
    logger.info("\n" + "=" * 60)
    logger.info("分析仿真结果")
    logger.info("=" * 60)

    analysis = {}

    for method, steps in results.items():
        successful = [s for s in steps if s.get('success', True)]

        if not successful:
            analysis[method] = {
                'avg_capacity': 0,
                'avg_vehicles_moved': 0,
                'success_rate': 0,
            }
            continue

        analysis[method] = {
            'avg_capacity': np.mean([s['avg_capacity'] for s in successful]),
            'avg_vehicles_moved': np.mean([s['vehicles_moved'] for s in successful]),
            'success_rate': len(successful) / len(steps),
        }

    df = pd.DataFrame(analysis).T

    logger.info("\n" + "=" * 60)
    logger.info("仿真结果对比表")
    logger.info("=" * 60)
    logger.info("\n" + df.to_string())
    logger.info("\n")

    # 计算改善
    if 'congestion_aware' in df.index and 'nearest_vehicle' in df.index:
        capacity_improvement = (
            df.loc['congestion_aware', 'avg_capacity'] -
            df.loc['nearest_vehicle', 'avg_capacity']
        ) / df.loc['nearest_vehicle', 'avg_capacity'] * 100

        logger.info(f"📊 核心指标：")
        logger.info(f"  你的方法相比Baseline的容量改善：{capacity_improvement:+.2f}%")

    return df


# ============================================================================
# 主函数：完整管道
# ============================================================================

def main():
    """完整的调度工程管道"""

    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "  车辆调度工程管道 - 路线A（拥堵感知再平衡）".center(78) + "║")
    logger.info("║" + "  SZ_TAXI数据 + STGCN预测 + IP优化 + 仿真验证".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")

    # ===== 第1步：加载数据 =====
    # 设置项目根目录（绝对路径）
    PROJECT_ROOT = "/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity"

    rel_file = os.path.join(PROJECT_ROOT, "raw_data/SZ_TAXI/SZ_TAXI.rel")
    dyna_file = os.path.join(PROJECT_ROOT, "raw_data/SZ_TAXI/SZ_TAXI.dyna")

    # 检查文件存在
    if not os.path.exists(rel_file):
        logger.error(f"文件不存在：{rel_file}")
        logger.error(f"当前工作目录：{os.getcwd()}")
        logger.error(f"请确保项目路径正确")
        return None, None
    if not os.path.exists(dyna_file):
        logger.error(f"文件不存在：{dyna_file}")
        return None, None

    logger.info(f"\n当前工作目录：{os.getcwd()}")
    logger.info(f"项目根目录：{PROJECT_ROOT}")

    data_loader = SZTaxiDataLoader(rel_file, dyna_file)

    # ===== 第2步：构建图 =====
    road_graph = RoadNetworkGraph(
        data_loader.get_rel(),
        data_loader.get_dyna()
    )
    num_regions = road_graph.get_num_regions()
    id_to_idx, idx_to_id = road_graph.get_mappings()

    # ===== 第3步：加载预测 =====
    pred_file = os.path.join(PROJECT_ROOT, "libcity/cache/STGCN-ROUND1/evaluate_cache/2026_06_02_19_52_43_STGCN_SZ_TAXI_predictions.npz")

    if not os.path.exists(pred_file):
        logger.error(f"文件不存在：{pred_file}")
        return None, None

    pred_loader = STGCNPredictionLoader(pred_file)
    speed_preds = pred_loader.get_first_step_predictions(num_samples=100)

    # ===== 第4步：初始化模块 =====
    capacity_converter = CapacityWeightConverter(reference_speed=30.0, min_capacity=0.3)
    dispatcher = IPVehicleDispatcher(alpha=0.7, time_limit=60)

    # 构造距离矩阵（基于图的最短路径）
    logger.info("\n" + "=" * 60)
    logger.info("构造距离矩阵")
    logger.info("=" * 60)

    # 获取图并计算最短路径
    G = road_graph.get_graph()
    distance_matrix = np.zeros((num_regions, num_regions))

    # 计算所有节点对之间的最短路径
    for i in range(num_regions):
        if i % 50 == 0:
            logger.info(f"  计算进度：{i}/{num_regions}")
        for j in range(num_regions):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                try:
                    # 使用图的最短路径长度
                    path_length = nx.shortest_path_length(G,
                                                           source=idx_to_id[i],
                                                           target=idx_to_id[j],
                                                           weight=None)  # 无权图，跳数作为距离
                    distance_matrix[i, j] = path_length
                except nx.NetworkXNoPath:
                    # 如果没有路径，使用大数
                    distance_matrix[i, j] = 100

    logger.info(f"✓ 距离矩阵构建完成")
    logger.info(f"  距离范围：{distance_matrix.min():.1f} - {distance_matrix.max():.1f}")

    # ===== 第5步：运行仿真 =====
    simulator = DispatchSimulator(
        capacity_converter,
        dispatcher,
        distance_matrix
    )

    # 模拟供给：每个区域20辆车
    num_samples = speed_preds.shape[0]
    current_supplies = np.ones((num_samples, num_regions)) * 20

    results = simulator.run_comparison(
        speed_preds,
        current_supplies,
        methods=['congestion_aware', 'nearest_vehicle', 'random'],
        verbose=True
    )

    # ===== 第6步：分析结果 =====
    analysis_df = analyze_results(results)

    # ===== 第7步：保存结果 =====
    output_dir = "/mnt/user-data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, "dispatch_comparison_results.csv")
    analysis_df.to_csv(csv_file)
    logger.info(f"\n✓ 结果已保存：{csv_file}")

    # 保存详细结果
    detailed_results = []
    for method, steps in results.items():
        for t, step in enumerate(steps):
            detailed_results.append({
                'method': method,
                'time_step': t,
                'success': step.get('success', False),
                'avg_capacity': step.get('avg_capacity', 0),
                'vehicles_moved': step.get('vehicles_moved', 0),
                'status': step.get('status', 'UNKNOWN')
            })

    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = os.path.join(output_dir, "dispatch_detailed_results.csv")
    detailed_df.to_csv(detailed_file, index=False)
    logger.info(f"✓ 详细结果已保存：{detailed_file}")

    logger.info("\n" + "=" * 80)
    logger.info("🎉 工程管道执行完成！")
    logger.info("=" * 80)

    return results, analysis_df


if __name__ == "__main__":
    results, analysis = main()