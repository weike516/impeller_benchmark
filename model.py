import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

class Impeller(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers, num_paths, path_length, num_edge_types, alpha, beta, operator_type="independent"):
        super(Impeller, self).__init__()
        self._dropout = dropout
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.in_act = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.out_act = nn.ReLU()
        self.layers = nn.ModuleList([
                ImpellerLayer(hidden_dim, num_paths, path_length, num_edge_types)
                for _ in range(num_layers)
            ])
        
        if operator_type == 'global':
            self.path_weights = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.Tensor(1, path_length, 1)) for _ in range(num_edge_types)])])
        elif operator_type == 'shared_layer':
            self.path_weights = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.Tensor(1, path_length, hidden_dim)) for _ in range(num_edge_types)])])
        elif operator_type == 'shared_channel':
            self.path_weights = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.Tensor(1, path_length, 1)) for _ in range(num_edge_types)]) for _ in range(num_layers)])
        elif operator_type == 'independent':
            self.path_weights = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.Tensor(1, path_length, hidden_dim)) for _ in range(num_edge_types)]) for _ in range(num_layers)])
                        
        for path_weight_layer in self.path_weights:
            for path_weight in path_weight_layer:
                nn.init.xavier_normal_(path_weight, gain=1.414)
        
        self.num_layers = num_layers
        self.num_paths = num_paths
        self.path_length = path_length
        self.num_edge_types = num_edge_types
        self.alpha = alpha
        self.beta = beta
        self.operator_type = operator_type
    
    def forward(self, input_x, paths, path_types):
        in_feats = F.dropout(input_x, p=self._dropout, training=self.training)
        in_feats = self.fc_in(in_feats)
        in_feats = self.in_act(in_feats)

        feats = in_feats
        for i in range(self.num_layers):
            feats_pre = feats
            if self.operator_type == "global" or self.operator_type == "shared_layer":
                feats = self.layers[i](feats, paths, path_types, self.path_weights[0])
            elif self.operator_type == "shared_channel" or self.operator_type == "independent":
                feats = self.layers[i](feats, paths, path_types, self.path_weights[i])
            else:
                raise NotImplementedError
            feats = (1-self.alpha-self.beta)*feats + self.beta*feats_pre + self.alpha*in_feats

        feats = F.dropout(feats, p=self._dropout, training=self.training)
        out = self.fc_out(feats)
        out = self.out_act(out)
        return out

    def setup_optimizer(self, lr, wd, lr_oc, wd_oc):
        param_list = [
            {"params": self.layers.parameters(), "lr": lr, "weight_decay": wd},
            {"params": itertools.chain(*[self.fc_in.parameters(), self.fc_out.parameters()]), "lr": lr_oc, "weight_decay": wd_oc} 
        ]
        return torch.optim.Adam(param_list)
    
class ImpellerLayer(nn.Module):
    def __init__(self, hidden_dim, num_path, path_length, num_edge_types):
        super(ImpellerLayer, self).__init__()
        
        self.fc = nn.Linear(num_edge_types*hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.num_path = num_path
        self.path_length = path_length
        self.num_edge_types = num_edge_types

    def forward(self, feats, paths, path_types, path_weights):
        """
            feats: (num_nodes, d),
            paths: (num_path, num_nodes, path_length)
            path_types: (num_path,) contains the edge type of each path
        """
        results = []
        for edge_type, path_weight in enumerate(path_weights):
            mask = (path_types == edge_type) # select the paths of this type
            paths_of_type = paths[mask] # (num_paths_of_type, num_nodes, path_length)
            path_feats = feats[paths_of_type] # (num_paths_of_type, num_nodes, path_length, d)
            path_feats = (path_feats * path_weight).sum(dim=2) # (num_paths_of_type, num_nodes, d)
            path_feats = path_feats.mean(dim=0) # (num_nodes, d)
            results.append(path_feats)
        if self.num_edge_types == 2:
            fout = torch.hstack((results[0],results[1]))
        else:
            fout = results[0]

        fout = self.fc(fout)
        fout = F.relu(fout)
        return fout