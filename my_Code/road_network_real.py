"""
road_network_real.py
====================
替换 dispatch_pipeline_fixed.py 中的 RoadNetwork 类。

核心变化：
  generate_distance_matrix() 不再随机生成坐标，
  而是加载由 build_real_distance.py 预计算的真实拓扑跳数距离矩阵。

使用方法：
  1. 先运行 build_real_distance.py 生成 sz_taxi_dist_matrix.npy
  2. 将本文件中的 RoadNetwork 类替换掉原来的版本
  3. 其余代码不需要改动
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csg


class RoadNetwork:
    def __init__(self, rel_path: str, dist_matrix_path: str = None):
        """
        Parameters
        ----------
        rel_path : str
            SZ_TAXI.rel 文件路径
        dist_matrix_path : str, optional
            预计算的距离矩阵路径（.npy）。
            若为 None，则自动在 rel_path 同目录下寻找
            sz_taxi_dist_matrix.npy；若不存在则即时计算。
        """
        rel = pd.read_csv(rel_path)

        # 节点列表（排序保证与 STGCN 预测的区域顺序一致）
        all_ids = sorted(set(rel["origin_id"]) | set(rel["destination_id"]))
        self.nodes = all_ids
        self.n = len(all_ids)
        self.id2idx = {nid: i for i, nid in enumerate(all_ids)}

        print(f"✅ 路网加载完成：总计 {self.n} 个区域")

        # 距离矩阵路径推断
        if dist_matrix_path is None:
            dist_matrix_path = os.path.join(
                os.path.dirname(rel_path), "sz_taxi_dist_matrix.npy"
            )

        if os.path.exists(dist_matrix_path):
            self._dist_matrix = np.load(dist_matrix_path).astype(np.float32)
            print(f"✅ 真实拓扑距离矩阵已加载: {dist_matrix_path}")
            print(f"   平均跳数: {self._dist_matrix[self._dist_matrix > 0].mean():.2f}")
        else:
            print(f"⚠️  未找到预计算距离矩阵，即时计算中...")
            self._dist_matrix = self._compute_distance_matrix(rel)
            np.save(dist_matrix_path, self._dist_matrix)
            print(f"✅ 距离矩阵已保存至: {dist_matrix_path}")

    def _compute_distance_matrix(self, rel: pd.DataFrame) -> np.ndarray:
        """从 .rel 真实边即时计算 BFS 跳数距离矩阵"""
        n = self.n

        # 构建邻接矩阵
        edges = rel[rel["link_weight"] == 1][["origin_id", "destination_id"]]
        adj = np.zeros((n, n), dtype=np.float32)
        for _, row in edges.iterrows():
            i = self.id2idx[row["origin_id"]]
            j = self.id2idx[row["destination_id"]]
            adj[i, j] = 1.0
            adj[j, i] = 1.0  # 强制对称

        sparse_adj = sp.csr_matrix(adj)

        # 最短路径
        dist = csg.shortest_path(
            sparse_adj, method="D", directed=False, unweighted=True
        )

        # 不可达节点对填充惩罚值
        finite_max = dist[np.isfinite(dist)].max()
        penalty = int(finite_max * 3)
        dist[np.isinf(dist)] = penalty
        np.fill_diagonal(dist, 0)

        n_inf = np.sum(dist >= penalty)
        print(f"   最大跳数: {finite_max:.0f}, 不可达对(已填充{penalty}): {n_inf}")

        return dist.astype(np.float32)

    def generate_distance_matrix(self, seed=None) -> np.ndarray:
        """
        返回真实拓扑距离矩阵。
        保留 seed 参数签名以兼容原代码，但不再使用随机种子。
        """
        # seed 参数保留兼容性，不影响结果
        return self._dist_matrix.copy()
