import os
import pandas as pd
import numpy as np

# 定义项目根目录
PROJECT_ROOT = "/home/dase/PyCharmMiscProject/diss/Bigscity-LibCity"

print("\n" + "=" * 80)
print("数据诊断脚本 - SZ_TAXI工程管道")
print("=" * 80 + "\n")

# ===== 检查1：.rel文件 =====
print("[1/4] 检查 .rel 文件...")
rel_file = os.path.join(PROJECT_ROOT, "raw_data/SZ_TAXI/SZ_TAXI.rel")

if os.path.exists(rel_file):
    print(f"✓ 文件存在：{rel_file}")
    df_rel = pd.read_csv(rel_file)
    print(f"  形状：{df_rel.shape}")
    print(f"  列名：{list(df_rel.columns)}")
    print(f"  前3行：\n{df_rel.head(3).to_string()}\n")
else:
    print(f"✗ 文件不存在：{rel_file}\n")

# ===== 检查2：.dyna文件 =====
print("[2/4] 检查 .dyna 文件...")
dyna_file = os.path.join(PROJECT_ROOT, "raw_data/SZ_TAXI/SZ_TAXI.dyna")

if os.path.exists(dyna_file):
    print(f"✓ 文件存在：{dyna_file}")
    df_dyna = pd.read_csv(dyna_file)
    print(f"  形状：{df_dyna.shape}")
    print(f"  列名：{list(df_dyna.columns)}")
    print(f"  前3行：\n{df_dyna.head(3).to_string()}\n")
    
    # 检查速度统计
    if 'traffic_speed' in df_dyna.columns:
        print(f"  速度统计：")
        print(f"    - 最小：{df_dyna['traffic_speed'].min():.1f} km/h")
        print(f"    - 最大：{df_dyna['traffic_speed'].max():.1f} km/h")
        print(f"    - 平均：{df_dyna['traffic_speed'].mean():.1f} km/h")
        print(f"    - 标准差：{df_dyna['traffic_speed'].std():.1f} km/h\n")
else:
    print(f"✗ 文件不存在：{dyna_file}\n")

# ===== 检查3：STGCN预测文件 =====
print("[3/4] 检查 STGCN预测文件...")
pred_dir = os.path.join(PROJECT_ROOT, "libcity/cache/STGCN-ROUND1/evaluate_cache")

# 列出目录下所有文件
if os.path.exists(pred_dir):
    print(f"✓ 目录存在：{pred_dir}")
    files = os.listdir(pred_dir)
    print(f"  目录下的文件（共{len(files)}个）：")
    for f in sorted(files)[:5]:  # 只显示前5个
        print(f"    - {f}")
    if len(files) > 5:
        print(f"    ... 还有 {len(files) - 5} 个文件")
    
    # 查找.npz文件
    npz_files = [f for f in files if f.endswith('.npz')]
    if npz_files:
        npz_file = os.path.join(pred_dir, npz_files[0])
        print(f"\n  找到.npz文件：{npz_files[0]}")
        
        try:
            data = np.load(npz_file, allow_pickle=True)
            print(f"  ✓ 文件可读")
            print(f"  键值：{list(data.keys())}")
            
            if 'prediction' in data:
                y_pred = data['prediction']
                print(f"  预测形状：{y_pred.shape}")
                print(f"    - 样本数（batch）：{y_pred.shape[0]}")
                print(f"    - 预测步长（horizon）：{y_pred.shape[1]}")
                print(f"    - 区域数（regions）：{y_pred.shape[2]}")
                print(f"    - 输出维度：{y_pred.shape[3]}")
                print(f"  预测值范围：{y_pred.min():.1f} - {y_pred.max():.1f}\n")
        except Exception as e:
            print(f"  ✗ 无法读取文件：{e}\n")
    else:
        print(f"  ✗ 目录中没有.npz文件\n")
else:
    print(f"✗ 目录不存在：{pred_dir}\n")

# ===== 检查4：输出目录 =====
print("[4/4] 检查输出目录...")
output_dir = "/mnt/user-data/outputs"

if os.path.exists(output_dir):
    print(f"✓ 输出目录存在：{output_dir}")
    print(f"  目录可写：{os.access(output_dir, os.W_OK)}")
else:
    print(f"⚠ 输出目录不存在，创建中...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 已创建：{output_dir}\n")
    except PermissionError:
        print(f"✗ 无法创建目录\n")

# ===== 总结 =====
print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

print("\n接下来的步骤：")
print("1. 如果所有文件都✓了，运行：python dispatch_pipeline_for_sz_taxi.py")
print("2. 如果有❌或⚠，请检查：")
print("   - 数据文件是否在正确的目录下")
print("   - STGCN模型是否已经训练过并生成了预测结果")
print(f"   - 项目根目录应为：{PROJECT_ROOT}")
print("\n")