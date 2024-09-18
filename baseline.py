import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import issparse

class SpatialKNNImputer:
    def __init__(self, adata, n_neighbors=5):
        self.adata = adata
        self.n_neighbors = n_neighbors
        self.spatial_net = self._build_spatial_net()
        
    def _build_spatial_net(self):
        """Build spatial network from adata.uns['Spatial_Net']."""
        spatial_net = pd.DataFrame(self.adata.uns['Spatial_Net'], columns=['Cell1', 'Cell2', 'Distance'])
        cell_to_index = {cell: idx for idx, cell in enumerate(self.adata.obs_names)}
        spatial_net['Cell1_idx'] = spatial_net['Cell1'].map(cell_to_index)
        spatial_net['Cell2_idx'] = spatial_net['Cell2'].map(cell_to_index)
        return spatial_net[['Cell1_idx', 'Cell2_idx', 'Distance']]

    def _find_neighbors(self, cell_idx):
        """Find the nearest neighbors for a given cell index based on spatial network."""
        neighbors = self.spatial_net[self.spatial_net['Cell1_idx'] == cell_idx]
        sorted_neighbors = neighbors.sort_values(by='Distance').head(self.n_neighbors)
        return sorted_neighbors['Cell2_idx'].values

    def impute(self, inference_mask):
        """Impute the gene expression matrix using KNN based on the spatial network."""
        X = self.adata.X
        if issparse(X):
            X = X.toarray() 

        X_masked = np.where(inference_mask, 0, X) 
        
        for i in tqdm(range(X.shape[0]), desc="Imputing rows"):
            if inference_mask[i].any(): 
                neighbors_idx = self._find_neighbors(i)
                for j in range(X.shape[1]):
                    if inference_mask[i, j]:
                        neighbor_values = X[neighbors_idx, j]
                        non_zero_values = neighbor_values[neighbor_values > 0]
                        if non_zero_values.size > 0: 
                            X_masked[i, j] = np.mean(non_zero_values)
        
        return X_masked