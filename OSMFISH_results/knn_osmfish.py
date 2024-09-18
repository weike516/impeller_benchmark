import os
import numpy as np
import torch
import scanpy as sc
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# 设置随机种子以确保结果可复现
np.random.seed(1)
torch.manual_seed(1)

# 实验参数
benchmark_samples = ['one sample']
data_dir = r"./1"
data_modes = ["all_gene"]
Ks = [10, 30, 60, 90, 120]  # 定义不同的 K 值
# 初始化空列表用于保存所有结果
results = []
# 设置设备为 GPU，如果不可用则回退到 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_process_h5ad(h5ad_path):
    """
    读取和处理 .h5ad 文件，将数据转换为适合 KNN 插补使用的格式。
    """
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)  # 使用 scanpy 读取 .h5ad 文件
    adata.var_names_make_unique()  # 确保基因名称唯一
    print("Data loaded and processed.")
    return adata

def knn_impute(data, k=5):
    """
    使用 KNN 方法对数据进行插补
    """
    print(f"Starting KNN imputation with k={k}...")
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # 对每个样本进行插补，使用邻居的均值进行填充
    imputed_data = data.copy()
    for i in range(data.shape[0]):
        neighbors = indices[i, :]
        imputed_data[i, :] = np.mean(data[neighbors, :], axis=0)
    
    print(f"KNN imputation with k={k} completed.")
    return imputed_data

for data_mode in data_modes:
    for sample_number in benchmark_samples:
        for k in Ks:  # 循环不同的 K 值
            test_mses, val_mses = [], []
            test_l1_distances, val_l1_distances = [], []
            test_cosine_sims, val_cosine_sims = [], []
            test_rmses, val_rmses = [], []

            print(f"----STARTING NEW RUN WITH K={k}----")
            print(f"sample_number: {sample_number}, data_mode: {data_mode}")

            for split_version in range(10):
                print(f"Processing split version {split_version}...")
                h5ad_path = os.path.join(data_dir, "filtered_adata.h5ad")
                adata = load_and_process_h5ad(h5ad_path)

                # 设置路径
                split_dir = r"./1/DataSplit/all_gene"
                test_mask_path = os.path.join(split_dir, f"split_{split_version}_test_mask.npz")
                val_mask_path = os.path.join(split_dir, f"split_{split_version}_val_mask.npz")

                # 加载掩码
                test_mask = np.load(test_mask_path)['arr_0']
                val_mask = np.load(val_mask_path)['arr_0']
                
                print("Masks loaded.")

                # 构建 PyTorch Geometric 的 Data 对象
                x = torch.from_numpy(adata.X.toarray() if hasattr(adata.X, 'todense') else adata.X).float().to(device)
                original_x = x.clone()

                # 使用 KNN 方法对测试和验证数据进行插补
                print("Performing KNN imputation on test and validation data...")
                imputed_data = knn_impute(original_x.cpu().numpy(), k=k)  # 使用当前 K 值插补
                imputed_data = torch.tensor(imputed_data).to(device)
                imputed_data[torch.isnan(imputed_data)] = 0

                # 计算评价指标
                test_indices = np.where(test_mask != 0)
                val_indices = np.where(val_mask != 0)

                test_mse = torch.mean((imputed_data[test_indices] - x[test_indices]) ** 2).item()
                val_mse = torch.mean((imputed_data[val_indices] - x[val_indices]) ** 2).item()

                test_l1_distance = torch.mean(torch.abs(imputed_data[test_indices] - x[test_indices])).item()
                val_l1_distance = torch.mean(torch.abs(imputed_data[val_indices] - x[val_indices])).item()

                # 计算余弦相似度
                norm_imputed_test = torch.sqrt(torch.sum(imputed_data[test_indices] ** 2))
                norm_original_test = torch.sqrt(torch.sum(x[test_indices] ** 2))
                dot_product_test = torch.sum(imputed_data[test_indices] * x[test_indices])
                test_cosine_sim = 1 - (dot_product_test / (norm_imputed_test * norm_original_test)).item()

                norm_imputed_val = torch.sqrt(torch.sum(imputed_data[val_indices] ** 2))
                norm_original_val = torch.sqrt(torch.sum(x[val_indices] ** 2))
                dot_product_val = torch.sum(imputed_data[val_indices] * x[val_indices])
                val_cosine_sim = 1 - (dot_product_val / (norm_imputed_val * norm_original_val)).item()

                test_rmse = torch.sqrt(torch.mean((imputed_data[test_indices] - x[test_indices]) ** 2)).item()
                val_rmse = torch.sqrt(torch.mean((imputed_data[val_indices] - x[val_indices]) ** 2)).item()

                # 存储结果
                test_mses.append(test_mse)
                val_mses.append(val_mse)
                test_l1_distances.append(test_l1_distance)
                val_l1_distances.append(val_l1_distance)
                test_cosine_sims.append(test_cosine_sim)
                val_cosine_sims.append(val_cosine_sim)
                test_rmses.append(test_rmse)
                val_rmses.append(val_rmse)
                
                # 添加每次运行的结果到结果列表
                results.append({
                    "sample_number": sample_number,
                    "data_mode": data_mode,
                    "split_version": split_version,
                    "k": k,  # 记录 K 值
                    "test_mse": test_mse,
                    "val_mse": val_mse,
                    "test_l1_distance": test_l1_distance,
                    "val_l1_distance": val_l1_distance,
                    "test_cosine_sim": test_cosine_sim,
                    "val_cosine_sim": val_cosine_sim,
                    "test_rmse": test_rmse,
                    "val_rmse": val_rmse
                })

                print(f"Split version {split_version} with K={k} completed.")
                print(f"Test MSE: {test_mse}, Validation MSE: {val_mse}")
                print(f"Test L1 Distance: {test_l1_distance}, Validation L1 Distance: {val_l1_distance}")
                print(f"Test Cosine Similarity: {test_cosine_sim}, Validation Cosine Similarity: {val_cosine_sim}")
                print(f"Test RMSE: {test_rmse}, Validation RMSE: {val_rmse}")

            print(f"----RUN WITH K={k} COMPLETED----")
            print(f"Results for sample_number: {sample_number}, data_mode: {data_mode}, K={k}")
            print(f"Test MSEs: {test_mses}")
            print(f"Validation MSEs: {val_mses}")
            print(f"Test L1 Distances: {test_l1_distances}")
            print(f"Validation L1 Distances: {val_l1_distances}")
            print(f"Test Cosine Similarities: {test_cosine_sims}")
            print(f"Validation Cosine Similarities: {val_cosine_sims}")
            print(f"Test RMSEs: {test_rmses}")
            print(f"Validation RMSEs: {val_rmses}")

# 将结果转换为 DataFrame 并保存为 CSV 文件
results_df = pd.DataFrame(results)
results_df.to_csv("knn_1_OSMFISH_results.csv", index=False)
print("Results saved to results_knn.csv")
