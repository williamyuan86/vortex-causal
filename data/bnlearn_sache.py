import os
from bnlearn import import_example, import_DAG

# 指定保存目录
save_dir = "./"  # 替换为你的目录路径
os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）

# 下载 Sachs 数据集
df = import_example('sachs')

# 保存数据集到指定目录
df.to_csv(os.path.join(save_dir, "sachs_data.csv"), index=False)
print(f"Sachs 数据集已保存到: {os.path.join(save_dir, 'sachs_data.csv')}")

# 获取 Sachs 的 ground truth 因果图
sachs_dag = import_DAG("sachs")

# 获取模型
model = sachs_dag["model"]  # BayesianNetwork 对象

# 保存为 BIF 格式
model.save(os.path.join(save_dir, "sachs_true_graph.bif"))

print(f"Sachs 的 ground truth 因果图（BIF 格式）已保存到: {os.path.join(save_dir, 'sachs_true_graph.bif')}")
