import numpy as np
import os

# 你的预测文件路径（和主程序一致）
pred_path = "/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity/libcity/cache/STGCN-ROUND1/evaluate_cache/2026_06_02_19_52_43_STGCN_SZ_TAXI_predictions.npz"

# 读取并打印所有键名
data = np.load(pred_path)
print("✅ 你的STGCN预测文件包含的键：")
print(list(data.keys()))

# 打印第一个数组的形状（确认数据格式）
if len(data.files) > 0:
    first_key = data.files[0]
    print(f"\n✅ 第一个键 {first_key} 的形状：", data[first_key].shape)


