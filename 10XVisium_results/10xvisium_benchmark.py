import os
import numpy as np
import torch
import scanpy as sc
from torch_geometric.data import Data
from Impeller import create_args#, train
from train import train,inference
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pandas as pd



# 设置随机种子以确保结果可复现
np.random.seed(1)
torch.manual_seed(1)

# 实验参数
benchmark_samples = ['one sample']
#Ks = [10,30]#, 30, 60, 90, 120]
data_dir = r"./10XVisium/DLPFC/Preprocessed/151507"
data_modes = ["all_gene"]
# 初始化空列表用于保存所有结果
results = []
# 设置设备为 GPU，如果不可用则回退到 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_process_h5ad(h5ad_path):
    """
    读取和处理 .h5ad 文件，将数据转换为适合 Impeller 使用的格式。
    """
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)  # 使用 scanpy 读取 .h5ad 文件
    adata.var_names_make_unique()  # 确保基因名称唯一
    print("Data loaded and processed.")
    return adata

def construct_pyg_data(adata, val_mask, test_mask):
    """
    构造 PyTorch Geometric 的 Data 对象并设置相关属性。
    """
    print("Constructing PyTorch Geometric Data object...")
    spatial_graph_radius_cutoff = 150
    gene_graph_knn_cutoff = 5
    gene_graph_pca_num = 50

    spatial_graph_coor = pd.DataFrame(adata.obsm['spatial'])
    spatial_graph_coor.index = adata.obs.index
    spatial_graph_coor.columns = ['imagerow', 'imagecol']
    id_cell_trans = dict(zip(range(spatial_graph_coor.shape[0]), np.array(spatial_graph_coor.index)))
    cell_to_index = {id_cell_trans[idx]: idx for idx in id_cell_trans}

    # 构建空间图
    nbrs = NearestNeighbors(radius=spatial_graph_radius_cutoff).fit(spatial_graph_coor)
    distances, indices = nbrs.radius_neighbors(spatial_graph_coor, return_distance=True)
    spatial_graph_KNN_list = [pd.DataFrame(zip([it] * len(indices[it]), indices[it], distances[it])) for it in range(len(indices))]
    spatial_graph_KNN_df = pd.concat(spatial_graph_KNN_list)
    spatial_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = spatial_graph_KNN_df[spatial_graph_KNN_df['Distance'] > 0]
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    spatial_graph_edge_index = []
    for _, row in Spatial_Net.iterrows():
        cell1 = cell_to_index[row['Cell1']]
        cell2 = cell_to_index[row['Cell2']]
        spatial_graph_edge_index.append([cell1, cell2])
        spatial_graph_edge_index.append([cell2, cell1])

    spatial_graph_edge_index = torch.tensor(spatial_graph_edge_index, dtype=torch.long).t().contiguous()

    # 构建基因图
    pca = PCA(n_components=gene_graph_pca_num)
    pca_data = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'todense') else adata.X)
    nbrs_gene = NearestNeighbors(n_neighbors=gene_graph_knn_cutoff).fit(pca_data)
    distances, indices = nbrs_gene.kneighbors(pca_data, return_distance=True)
    gene_graph_KNN_list = [pd.DataFrame(zip([it] * len(indices[it]), indices[it], distances[it])) for it in range(len(indices))]
    gene_graph_KNN_df = pd.concat(gene_graph_KNN_list)
    gene_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Gene_Net = gene_graph_KNN_df.copy()
    Gene_Net['Cell1'] = Gene_Net['Cell1'].map(id_cell_trans)
    Gene_Net['Cell2'] = Gene_Net['Cell2'].map(id_cell_trans)

    gene_graph_edge_index = []
    for _, row in Gene_Net.iterrows():
        cell1 = cell_to_index[row['Cell1']]
        cell2 = cell_to_index[row['Cell2']]
        gene_graph_edge_index.append([cell1, cell2])
        gene_graph_edge_index.append([cell2, cell1])

    gene_graph_edge_index = torch.tensor(gene_graph_edge_index, dtype=torch.long).t().contiguous()

    # 构建节点特征
    x = torch.from_numpy(adata.X.toarray() if hasattr(adata.X, 'todense') else adata.X).float().to(device)
    original_x = x.clone()

    # 构建 PyTorch Geometric 的 Data 对象
    edge_index = torch.cat([spatial_graph_edge_index, gene_graph_edge_index], dim=1).to(device)
    edge_type = torch.cat([torch.zeros(spatial_graph_edge_index.size(1), dtype=torch.long),
                           torch.ones(gene_graph_edge_index.size(1), dtype=torch.long)], dim=0).to(device)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type).to(device)
    data.num_node_features = x.shape[1]

    print("PyTorch Geometric Data object constructed.")
    return data, val_mask, test_mask, x, original_x

for data_mode in data_modes:
    for sample_number in benchmark_samples:
        
            test_mses, val_mses = [], []
            test_l1_distances, val_l1_distances = [], []
            test_cosine_sims, val_cosine_sims = [], []
            test_rmses, val_rmses = [], []

            print("----STARTING NEW RUN----")
            print(f"sample_number: {sample_number}, data_mode: {data_mode}")

            for split_version in range(10):
                print(f"Processing split version {split_version}...")
                h5ad_path = os.path.join(data_dir, "filtered_adata.h5ad")
                adata = load_and_process_h5ad(h5ad_path)

                # 设置路径
                split_dir = r"./10XVisium/DLPFC/Preprocessed/151507/DataSplit/all_gene"
                test_mask_path = os.path.join(split_dir, f"split_{split_version}_test_mask.npz")
                val_mask_path = os.path.join(split_dir, f"split_{split_version}_val_mask.npz")

                # 加载掩码
                test_mask = np.load(test_mask_path)['arr_0']
                val_mask = np.load(val_mask_path)['arr_0']
                
                print("Masks loaded.")

                # 构建 PyTorch Geometric 的 Data 对象
                data, val_mask_tensor, test_mask_tensor, x, original_x = construct_pyg_data(adata, val_mask, test_mask)
                # 将 val_mask_tensor 和 test_mask_tensor 转换为布尔类型并移动到 GPU 上
                #val_mask_tensor = val_mask_tensor.to(device).type(torch.bool)
                #test_mask_tensor = test_mask_tensor.to(device).type(torch.bool)

                # 创建 Impeller 模型参数
                args = create_args()  # 创建模型参数
                print("Model parameters created.")

                # 将掩码转换为 PyTorch 张量并确保它们在设备上
                val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool).to(device)
                test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool).to(device)

                # 使用 Impeller 进行训练并计算评价指标
                '''print("Starting training...")
                test_l1_distance, test_cosine_sim, test_rmse = train(
                    args, data, val_mask_tensor, test_mask_tensor, x, original_x
                )
                print("Training completed.")'''
                
                # Step 1: 训练模型并保存参数
                print("Starting training...")
                model, test_l1_distance, test_cosine_sim, test_rmse = train(args, data, val_mask_tensor, test_mask_tensor, x, original_x)
                torch.save(model.state_dict(), f"trained_model_split_{split_version}.pth")
                print("Training completed and model saved.")

                # Step 2: 加载已保存的模型参数
                # 创建相同的模型实例以进行推断
                '''inference_model = create_model(args)  # Ensure `create_model` correctly initializes the model
                inference_model.load_state_dict(torch.load(f"trained_model_split_{split_version}.pth"))
                inference_model.eval()  # Set the model to evaluation mode
                inference_model.to(device)'''

                # Step 3: 使用保存的参数进行插补
                with torch.no_grad():
                    Impeller_imputed_data = inference(args, data, test_mask_tensor, split_version)  # Update inference call as needed
                    imputed_data = Impeller_imputed_data#.x
                    imputed_data[torch.isnan(imputed_data)] = 0

                # Step 4: 计算评价指标
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
                    "test_mse": test_mse,
                    "val_mse": val_mse,
                    "test_l1_distance": test_l1_distance,
                    "val_l1_distance": val_l1_distance,
                    "test_cosine_sim": test_cosine_sim,
                    "val_cosine_sim": val_cosine_sim,
                    "test_rmse": test_rmse,
                    "val_rmse": val_rmse
                })

                print(f"Split version {split_version} completed.")
                print(f"Test MSE: {test_mse}, Validation MSE: {val_mse}")
                print(f"Test L1 Distance: {test_l1_distance}, Validation L1 Distance: {val_l1_distance}")
                print(f"Test Cosine Similarity: {test_cosine_sim}, Validation Cosine Similarity: {val_cosine_sim}")
                print(f"Test RMSE: {test_rmse}, Validation RMSE: {val_rmse}")

            print("----RUN COMPLETED----")
            print(f"Results for sample_number: {sample_number}, data_mode: {data_mode}")
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
results_df.to_csv("DGL_151507_10XVisium_results.csv", index=False)
print("Results saved to results.csv")
