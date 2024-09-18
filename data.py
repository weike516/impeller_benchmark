import os
import gdown
import pickle
import zipfile

import numpy as np
import scanpy as sc
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import torch

from torch_geometric.data import Data

def download_example_data(example_dataset = '10XVisium'):
    valid_datasets = ['10XVisium', 'Stereoseq', 'SlideseqV2']
    if example_dataset not in valid_datasets:
        raise NotImplementedError(f"Dataset {example_dataset} is not supported.")
    
    if example_dataset == '10XVisium':
        # Google Drive file download link
        url = 'https://drive.google.com/uc?export=download&id=1cR-iuGbbGpKpoeIkjbS0Q1xO4hLgpiKX'
        # Local directory to save the file
        output_directory = 'example_data/10XVisium/DLPFC/151673'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Local file path
        file_path = os.path.join(output_directory, 'dlpfc_filtered_151673.zip')
        extracted_marker = os.path.join(output_directory, 'dlpfc_filtered_151673_extracted')
    elif example_dataset == 'Stereoseq':
        url = 'https://drive.google.com/uc?export=download&id=1diTt-g5sOXRbyQ7gU-KvwmanwTvOwQCy'
        output_directory = 'example_data/Stereoseq/MouseOlfactoryBulb'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_path = os.path.join(output_directory, 'Stereoseq_MouseOlfactoryBulb.zip')
        extracted_marker = os.path.join(output_directory, 'Stereoseq_MouseOlfactoryBulb_extracted')
    elif example_dataset == 'SlideseqV2':
        url = 'https://drive.google.com/uc?export=download&id=17DUe9uc2UylTo-9xM4pGtMx31h9qkhCD'
        output_directory = 'example_data/SlideseqV2/MouseOlfactoryBulb'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_path = os.path.join(output_directory, 'SlideseqV2_MouseOlfactoryBulb.zip')
        extracted_marker = os.path.join(output_directory, 'SlideseqV2_MouseOlfactoryBulb_extracted')
        
    # Check if the file has already been downloaded and extracted
    if os.path.exists(extracted_marker):
        print(f'The file has already been downloaded and extracted to {output_directory}.')
        return

    # Download the file if it does not exist
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)
    else:
        print(f'File already exists at {file_path}. Skipping download.')

    # Extract the ZIP file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    # Remove the ZIP file after extraction
    os.remove(file_path)

    # Create a marker file to indicate extraction is done
    with open(extracted_marker, 'w') as f:
        f.write('')

    print(f'File downloaded to {file_path} and extracted successfully.')

def load_example_data(example_dataset = '10XVisium'):
    valid_datasets = ['10XVisium', 'Stereoseq', 'SlideseqV2']
    if example_dataset not in valid_datasets:
        raise NotImplementedError(f"Dataset {example_dataset} is not supported.")

    if example_dataset == "10XVisium":
        file_path = "./example_data/10XVisium/DLPFC/151673/"
    elif example_dataset == "Stereoseq":
        file_path = "./example_data/Stereoseq/MouseOlfactoryBulb/"
    elif example_dataset == "SlideseqV2":
        file_path = "./example_data/SlideseqV2/MouseOlfactoryBulb/"
    
    adata = sc.read_h5ad(file_path+"filtered_adata.h5ad")
    val_mask = np.load(file_path+"split_0_val_mask.npz")['arr_0']
    test_mask = np.load(file_path+"split_0_test_mask.npz")['arr_0']
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)
    
    return adata, val_mask, test_mask

def load_and_process_example_data(example_dataset = '10XVisium'):
    
    adata, val_mask, test_mask = load_example_data(example_dataset)
    
    spatial_graph_radius_cutoff = 150
    gene_graph_knn_cutoff = 5
    gene_graph_pca_num = 50
    
    # adata.obs['Ground Truth'] = list(ground_truth_layer)
    spatial_graph_coor = pd.DataFrame(adata.obsm['spatial'])
    spatial_graph_coor.index = adata.obs.index
    spatial_graph_coor.columns = ['imagerow', 'imagecol']
    id_cell_trans = dict(zip(range(spatial_graph_coor.shape[0]), np.array(spatial_graph_coor.index)))
    cell_to_index = {id_cell_trans[idx]: idx for idx in id_cell_trans}
    nbrs = NearestNeighbors(radius=spatial_graph_radius_cutoff).fit(spatial_graph_coor)
    distances, indices = nbrs.radius_neighbors(spatial_graph_coor, return_distance=True)

    spatial_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
    spatial_graph_KNN_df = pd.concat(spatial_graph_KNN_list)
    spatial_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = spatial_graph_KNN_df.copy()
    Spatial_Net = Spatial_Net[Spatial_Net['Distance']>0]
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    print(f'The spatial graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
    print(f'{Spatial_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
    adata.uns['Spatial_Net'] = Spatial_Net

    spatial_graph_edge_index = []
    for idx, row in Spatial_Net.iterrows():
        cell1 = cell_to_index[row['Cell1']]
        cell2 = cell_to_index[row['Cell2']]
        spatial_graph_edge_index.append([cell1, cell2])
        spatial_graph_edge_index.append([cell2, cell1])
        
    spatial_graph_edge_index = torch.tensor(spatial_graph_edge_index, dtype=torch.long).t().contiguous()
    
    pca = PCA(n_components=gene_graph_pca_num)
    if hasattr(adata.X, 'todense'):
        pca_data = pca.fit_transform(np.array(adata.X.todense().astype(np.float32)))
    else:
        pca_data = pca.fit_transform(adata.X.astype(np.float32))
    nbrs_gene = NearestNeighbors(n_neighbors=gene_graph_knn_cutoff).fit(pca_data)
    distances, indices = nbrs_gene.kneighbors(pca_data, return_distance=True)
    
    gene_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
    gene_graph_KNN_df = pd.concat(gene_graph_KNN_list)
    gene_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Gene_Net = gene_graph_KNN_df.copy()
    Gene_Net['Cell1'] = Gene_Net['Cell1'].map(id_cell_trans)
    Gene_Net['Cell2'] = Gene_Net['Cell2'].map(id_cell_trans)
    print(f'The gene graph contains {Gene_Net.shape[0]} edges, {adata.n_obs} cells.')
    print(f'{Gene_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
    gene_graph_edge_index = []
    for idx, row in Gene_Net.iterrows():
        cell1 = cell_to_index[row['Cell1']]
        cell2 = cell_to_index[row['Cell2']]
        gene_graph_edge_index.append([cell1, cell2])
        gene_graph_edge_index.append([cell2, cell1])  # Add reversed edge since the graph is undirected
    gene_graph_edge_index = torch.tensor(gene_graph_edge_index, dtype=torch.long).t().contiguous()
    
    # Create a PyTorch tensor for node features
    if hasattr(adata.X, 'todense'):
        x = torch.from_numpy(adata.X.todense().astype(np.float32))
    else:
        x = torch.from_numpy(adata.X.astype(np.float32))
    original_x = x.clone()  # Save the original features
    
    edge_index = torch.cat([spatial_graph_edge_index, gene_graph_edge_index], dim=1)
    edge_type = torch.cat([torch.zeros(spatial_graph_edge_index.size(1), dtype=torch.long),
                            torch.ones(gene_graph_edge_index.size(1), dtype=torch.long)], dim=0)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    return data, val_mask, test_mask, x, original_x
    
def process_inference_data(adata, spatial_graph_radius_cutoff = 150, gene_graph_knn_cutoff = 5, gene_graph_pca_num = 50):

    spatial_graph_coor = pd.DataFrame(adata.obsm['spatial'])
    spatial_graph_coor.index = adata.obs.index
    spatial_graph_coor.columns = ['imagerow', 'imagecol']
    id_cell_trans = dict(zip(range(spatial_graph_coor.shape[0]), np.array(spatial_graph_coor.index)))
    cell_to_index = {id_cell_trans[idx]: idx for idx in id_cell_trans}
    nbrs = NearestNeighbors(radius=spatial_graph_radius_cutoff).fit(spatial_graph_coor)
    distances, indices = nbrs.radius_neighbors(spatial_graph_coor, return_distance=True)

    spatial_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
    spatial_graph_KNN_df = pd.concat(spatial_graph_KNN_list)
    spatial_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = spatial_graph_KNN_df.copy()
    Spatial_Net = Spatial_Net[Spatial_Net['Distance']>0]
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    print(f'The spatial graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
    print(f'{Spatial_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
    adata.uns['Spatial_Net'] = Spatial_Net

    spatial_graph_edge_index = []
    for idx, row in Spatial_Net.iterrows():
        cell1 = cell_to_index[row['Cell1']]
        cell2 = cell_to_index[row['Cell2']]
        spatial_graph_edge_index.append([cell1, cell2])
        spatial_graph_edge_index.append([cell2, cell1])
        
    spatial_graph_edge_index = torch.tensor(spatial_graph_edge_index, dtype=torch.long).t().contiguous()
    
    pca = PCA(n_components=gene_graph_pca_num)
    if hasattr(adata.X, 'todense'):
        pca_data = pca.fit_transform(np.array(adata.X.todense().astype(np.float32)))
    else:
        pca_data = pca.fit_transform(adata.X.astype(np.float32))
    nbrs_gene = NearestNeighbors(n_neighbors=gene_graph_knn_cutoff).fit(pca_data)
    distances, indices = nbrs_gene.kneighbors(pca_data, return_distance=True)
    
    gene_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
    gene_graph_KNN_df = pd.concat(gene_graph_KNN_list)
    gene_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Gene_Net = gene_graph_KNN_df.copy()
    Gene_Net['Cell1'] = Gene_Net['Cell1'].map(id_cell_trans)
    Gene_Net['Cell2'] = Gene_Net['Cell2'].map(id_cell_trans)
    print(f'The gene graph contains {Gene_Net.shape[0]} edges, {adata.n_obs} cells.')
    print(f'{Gene_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
    gene_graph_edge_index = []
    for idx, row in Gene_Net.iterrows():
        cell1 = cell_to_index[row['Cell1']]
        cell2 = cell_to_index[row['Cell2']]
        gene_graph_edge_index.append([cell1, cell2])
        gene_graph_edge_index.append([cell2, cell1])  # Add reversed edge since the graph is undirected
    gene_graph_edge_index = torch.tensor(gene_graph_edge_index, dtype=torch.long).t().contiguous()
    
    # Create a PyTorch tensor for node features
    if hasattr(adata.X, 'todense'):
        x = torch.from_numpy(adata.X.todense().astype(np.float32))
    else:
        x = torch.from_numpy(adata.X.astype(np.float32))
    
    edge_index = torch.cat([spatial_graph_edge_index, gene_graph_edge_index], dim=1)
    edge_type = torch.cat([torch.zeros(spatial_graph_edge_index.size(1), dtype=torch.long),
                            torch.ones(gene_graph_edge_index.size(1), dtype=torch.long)], dim=0)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    return data
    