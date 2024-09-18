# 步骤 1：下载 Stereoseq 数据集
from Impeller import download_example_data

# 下载 Stereoseq 数据集
download_example_data('Stereoseq')

# 步骤 2：加载和处理 Stereoseq 数据
from Impeller import load_and_process_example_data

# 加载和处理 Stereoseq 数据
data, val_mask, test_mask, x, original_x = load_and_process_example_data('Stereoseq')

# 步骤 3：初始化模型参数并训练模型
from Impeller import create_args, train
import torch

# 创建模型参数
args = create_args()

# 强制将模型和数据转移到CPU上
args.device = torch.device('cpu')  # 将设备设置为CPU
data = data.to(args.device)
val_mask = val_mask.to(args.device)
test_mask = test_mask.to(args.device)
x = x.to(args.device)
original_x = original_x.to(args.device)

# 使用 Stereoseq 数据训练模型
test_l1_distance, test_cosine_sim, test_rmse = train(args, data, val_mask, test_mask, x, original_x)

# 打印最终的评估指标
print(f"最终 L1 距离: {test_l1_distance}, 余弦相似度: {test_cosine_sim}, 均方根误差 (RMSE): {test_rmse}.")
