"""
build_real_distance.py
======================
从 SZ_TAXI.rel 的真实拓扑边（link_weight=1）出发，
用图最短路径跳数构建 156×156 距离矩阵，
替换 dispatch_pipeline_fixed.py 中的随机距离矩阵。

输出：
  - sz_taxi_dist_matrix.npy   距离矩阵 (156, 156)，不可达点用999填充
  - sz_taxi_adj_matrix.npy    邻接矩阵 (156, 156)，0/1二值
  - sz_taxi_node_index.csv    节点ID ↔ 矩阵index 映射表

用法：
  python build_real_distance.py --rel_path /path/to/SZ_TAXI.rel
"""

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

REL_DEFAULT = (
    "/home/dase/PyCharmMiscProject/diss/"
    "Bigscity-LibCity/raw_data/SZ_TAXI/SZ_TAXI.rel"
)


def build_matrices(rel_path: str):
    rel = pd.read_csv(rel_path)

    # ── 1. 提取所有节点，建立 ID→index 映射 ──────────────────────────
    all_ids = sorted(set(rel["origin_id"]) | set(rel["destination_id"]))
    n = len(all_ids)
    id2idx = {nid: i for i, nid in enumerate(all_ids)}

    print(f"节点总数: {n}")
    print(f"节点ID范围: {all_ids[0]} ~ {all_ids[-1]}")

    # ── 2. 构建邻接矩阵（只用 link_weight=1 的真实边）─────────────────
    edges = rel[rel["link_weight"] == 1][["origin_id", "destination_id"]]
    print(f"真实边数量(link_weight=1): {len(edges)}")

    adj = np.zeros((n, n), dtype=np.float32)
    for _, row in edges.iterrows():
        i = id2idx[row["origin_id"]]
        j = id2idx[row["destination_id"]]
        adj[i, j] = 1.0
        # .rel 已近似对称(530/532双向)，但保险起见强制对称
        adj[j, i] = 1.0

    connected_nodes = np.sum(adj.sum(axis=1) > 0)
    print(f"有连接的节点数: {connected_nodes}/{n}")
    print(f"邻接矩阵密度: {adj.sum()/(n*n)*100:.2f}%")

    # ── 3. 检查连通性 ─────────────────────────────────────────────────
    sparse_adj = sp.csr_matrix(adj)
    n_components, labels = csg.connected_components(
        sparse_adj, directed=False, return_labels=True
    )
    print(f"\n连通分量数: {n_components}")
    if n_components > 1:
        sizes = np.bincount(labels)
        print(f"各分量大小: {sorted(sizes, reverse=True)}")
        largest = np.argmax(sizes)
        isolated = np.sum(sizes == 1)
        print(f"最大分量: {sizes[largest]} 节点")
        print(f"孤立节点数(度=0): {isolated}")

    # ── 4. 用 BFS 最短路径（跳数）构建距离矩阵 ───────────────────────
    # floyd_warshall 对 156 节点完全可行（156²=24336，ms级）
    # 不可达路径距离设为 inf，后面替换为 999（惩罚值）
    print("\n计算最短路径距离矩阵（BFS跳数）...")

    # 将邻接矩阵中的 1 视为边权=1
    dist_matrix = csg.shortest_path(
        sparse_adj, method="D", directed=False, unweighted=True
    )

    inf_count = np.sum(np.isinf(dist_matrix))
    print(f"不可达节点对数量: {inf_count}")

    # 不可达节点对用惩罚距离填充（设为最大跳数的3倍）
    finite_max = dist_matrix[np.isfinite(dist_matrix)].max()
    penalty = int(finite_max * 3)
    dist_matrix[np.isinf(dist_matrix)] = penalty
    np.fill_diagonal(dist_matrix, 0)

    print(f"\n距离矩阵统计:")
    print(f"  最小跳数(非零): {dist_matrix[dist_matrix>0].min():.0f}")
    print(f"  最大跳数(非惩罚): {finite_max:.0f}")
    print(f"  平均跳数(非惩罚): {dist_matrix[dist_matrix<penalty].mean():.2f}")
    print(f"  不可达填充值: {penalty}")

    # ── 5. 保存结果 ───────────────────────────────────────────────────
    np.save("sz_taxi_dist_matrix.npy", dist_matrix.astype(np.float32))
    np.save("sz_taxi_adj_matrix.npy", adj)

    node_df = pd.DataFrame({
        "index": list(range(n)),
        "geo_id": all_ids
    })
    node_df.to_csv("sz_taxi_node_index.csv", index=False)

    print("\n✅ 已保存:")
    print("  sz_taxi_dist_matrix.npy  —— 距离矩阵 (156×156)")
    print("  sz_taxi_adj_matrix.npy   —— 邻接矩阵 (156×156)")
    print("  sz_taxi_node_index.csv   —— 节点ID映射表")

    return dist_matrix, adj, id2idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rel_path", default=REL_DEFAULT)
    args = parser.parse_args()
    build_matrices(args.rel_path)


if __name__ == "__main__":
    main()
