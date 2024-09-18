import dgl
import numpy as np
import torch
import torch.nn.functional as F

def get_paths(args, data):
    device = args.device
    # Separate the graph by edge type
    g0, g1 = separate_graph_by_edge_type(data.cpu())
    
    # Perform random walk on each graph
    path0 = get_random_walk_path(g0, args.num_paths, args.path_length-1, p=args.spatial_walk_p, q=args.spatial_walk_q)
    path1 = get_random_walk_path(g1, args.num_paths, args.path_length-1, p=args.gene_walk_p, q=args.gene_walk_q)
    paths = torch.vstack((path0,path1)).to(device).long()

    path0_types = torch.zeros(path0.shape[0], dtype=torch.long)
    path1_types = torch.ones(path1.shape[0], dtype=torch.long)
    path_types = torch.hstack((path0_types, path1_types)).to(device)

    return paths, path_types

def get_random_walk_path(g, num_walks, walk_length, p=1, q=1):
    """
    Get random walk paths.
    """
    device = g.device
    g = g.to("cpu")
    walks = []
    nodes = g.nodes()

    for _ in range(num_walks):
        walks.append(
            # dgl.sampling.random_walk(g, nodes, length=walk_length)[0]
            dgl.sampling.node2vec_random_walk(g, nodes, p=p, q=q, walk_length=walk_length)
        )
    walks = torch.stack(walks).to(device) # (num_walks, num_nodes, walk_length)
    return walks

def pyg_to_dgl(data):
    g = dgl.DGLGraph()
    g.add_nodes(data.num_nodes)
    g.add_edges(data.edge_index[0], data.edge_index[1])

    return g

def separate_graph_by_edge_type(data):
    mask_type_0 = data.edge_type == 0
    mask_type_1 = data.edge_type == 1
    
    edge_index_0 = data.edge_index[:, mask_type_0]
    edge_index_1 = data.edge_index[:, mask_type_1]
    
    g0 = dgl.DGLGraph()
    g0.add_nodes(data.num_nodes)
    g0.add_edges(edge_index_0[0], edge_index_0[1])
    
    g1 = dgl.DGLGraph()
    g1.add_nodes(data.num_nodes)
    g1.add_edges(edge_index_1[0], edge_index_1[1])

    return g0, g1

def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def inference_Impeller(model, x, paths, path_types):
    model.eval()
    with torch.no_grad():
        logits = model(x, paths, path_types)
    return logits

def evaluate_Impeller(model, x, paths, path_types, device, mask, original_x):
    model.eval()
    with torch.no_grad():
        # Compute the logits
        logits = model(x, paths, path_types)
        mask = mask.to(device)
        masked_logits = logits[mask]
        masked_original_x = original_x[mask]
    all_masked_logits = masked_logits
    all_masked_original_x = masked_original_x
    
    l1_distance = F.l1_loss(all_masked_logits, all_masked_original_x).item()
    cosine_sim = cosine_similarity(all_masked_logits, all_masked_original_x).mean().item()
    rmse = torch.sqrt(F.mse_loss(all_masked_logits, all_masked_original_x)).item()

    return l1_distance, cosine_sim, rmse