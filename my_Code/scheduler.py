import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linprog

# ==========================
# 1. 计算交通枢纽权重（来自 .rel + .dyna）
# ==========================
df_rel = pd.read_csv('/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity/raw_data/SZ_TAXI/SZ_TAXI.rel')
df_rel = df_rel[df_rel['origin_id'] != df_rel['destination_id']]
G = nx.from_pandas_edgelist(df_rel, 'origin_id', 'destination_id')
deg_central = nx.degree_centrality(G)

df_dyna = pd.read_csv('/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity/raw_data/SZ_TAXI/SZ_TAXI.dyna')
speed_std = df_dyna.groupby('entity_id')['traffic_speed'].std()
speed_norm = speed_std / speed_std.max()

hub_final = {}
for nid in deg_central.keys():
    topo_w = deg_central[nid]
    traffic_w = speed_norm.get(nid, 0)
    hub_final[nid] = 0.5 * topo_w + 0.5 * traffic_w

# ==========================
# 2. 路段ID映射（真实ID ↔ 模型索引）
# ==========================
all_node_ids = sorted(deg_central.keys())
id_to_idx = {node: i for i, node in enumerate(all_node_ids)}
idx_to_id = {i: node for i, node in enumerate(all_node_ids)}

# ==========================
# 3. 读取STGCN预测结果
# ==========================
y_pred_file = np.load('/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity/libcity/cache/STGCN-ROUND1/evaluate_cache/2026_06_02_19_52_43_STGCN_SZ_TAXI_predictions.npz')
y_pred = y_pred_file['prediction']
y_true = y_pred_file['truth']

print("y_pred 形状：", y_pred.shape)
print("真实路段数量：", len(all_node_ids))

# ==========================
# 4. 计算调度优先级分数
# ==========================
def get_dispatch_recommendation(y_pred, hub_final, idx_to_id, threshold=20.0, top_k=3):
    first_step_pred = y_pred[:, 0, :, 0]
    congestion_score = np.maximum(0, threshold - first_step_pred)
    hub_vector = np.array([hub_final[idx_to_id[i]] for i in range(len(idx_to_id))])
    final_score = 0.6 * congestion_score + 0.4 * hub_vector
    top_indices = np.argsort(final_score, axis=1)[:, ::-1][:, :top_k]
    top_ids = [[idx_to_id[i] for i in row] for row in top_indices]
    return top_ids, final_score

top_node_ids, final_scores = get_dispatch_recommendation(y_pred, hub_final, idx_to_id)

print("\n✅ 优先调度区域：")
for i, node_id in enumerate(top_node_ids[0]):
    print(f"TOP {i+1}: 区域 {node_id}")

# ==========================
# 5. ✅ 运筹优化分配车辆（你开题要的核心！）
# ==========================
print("\n" + "="*60)
print("🚕 运筹优化 — 有限车辆最优分配")
print("="*60)

demand_score = final_scores[0]
N = len(demand_score)
TOTAL_CARS = 50

# 线性规划优化
c = -demand_score
A_eq = np.ones((1, N))
b_eq = [TOTAL_CARS]
bounds = [(0, None) for _ in range(N)]

result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
allocation = result.x

# 输出最优方案
top_indices = np.argsort(allocation)[::-1][:5]
for i, idx in enumerate(top_indices):
    region = idx_to_id[idx]
    cars = int(round(allocation[idx]))
    print(f"TOP {i+1} 区域 {region} → 分配 {cars} 辆车")
