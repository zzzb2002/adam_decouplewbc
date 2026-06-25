import os
import pickle
import pprint
import joblib
import numpy as np

file_path = "src/output_pkl/CLOT2GMR/0-dance2_Skeleton_007_z_up_x_forward_gym.pkl"

# 读取 pkl 文件
with open(file_path, "rb") as f:
    data = joblib.load(f)
    # 如果是嵌套字典，取第一个 key 的值
    if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
        data = data[list(data.keys())[0]]

# 统一字段名
key_mapping = {"dof": "dof_pos"}
data = {(key_mapping.get(k, k)): v for k, v in data.items()}

# 添加 link_body_list (如果不存在)
if "link_body_list" not in data:
    if "local_body_pos" in data:
        n_bodies = data["local_body_pos"].shape[1]
        data["link_body_list"] = [f"body_{i}" for i in range(n_bodies)]
    elif "dof_pos" in data:
        n_dof = data["dof_pos"].shape[1]
        n_bodies = n_dof // 3
        data["link_body_list"] = [f"joint_{i}" for i in range(n_bodies)]

# 打印
print("Keys:", list(data.keys()))
print()
for k, v in data.items():
    print(f"{k}:")
    print(f"  type = {type(v)}")
    if hasattr(v, "shape"):
        print(f"  shape = {v.shape}")
    pprint.pprint(v if not hasattr(v, "shape") else f"ndarray shape={v.shape}")
    print()

