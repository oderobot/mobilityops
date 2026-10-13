import pandas as pd
import networkx as nx
import numpy as np

# 1. 读取 rel 路网（正确）
df_rel = pd.read_csv('/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity/raw_data/SZ_TAXI/SZ_TAXI.rel')
df_rel = df_rel[df_rel['origin_id'] != df_rel['destination_id']]
G = nx.from_pandas_edgelist(df_rel, 'origin_id', 'destination_id')
deg_central = nx.degree_centrality(G)

# 2. 读取速度数据 ✅ 修复文件名：dyna 不是 dynana
df_dyna = pd.read_csv('/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity/raw_data/SZ_TAXI/SZ_TAXI.dyna')
speed_std = df_dyna.groupby('entity_id')['traffic_speed'].std()
speed_norm = speed_std / speed_std.max()

# 3. 融合枢纽权重
hub_final = {}
for nid in deg_central.keys():
    topo_w = deg_central[nid]
    traffic_w = speed_norm.get(nid, 0)
    hub_final[nid] = 0.5 * topo_w + 0.5 * traffic_w

# 输出结果
print("融合型枢纽权重（越大=越重要）：")
for k, v in list(hub_final.items())[:15]:
    print(f"区域ID: {k}, 枢纽权重: {v:.4f}")
# 把所有枢纽权重转成 DataFrame
hub_df = pd.DataFrame(list(hub_final.items()), columns=["区域ID", "枢纽权重"])

# 保存成 CSV 文件（你可以用 Excel 打开）
hub_df.to_csv("hub_weights.csv", index=False)
print("已保存全部枢纽权重到 hub_weights.csv")