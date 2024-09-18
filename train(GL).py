import warnings
warnings.filterwarnings("ignore")

import copy
import torch

from Impeller.model import Impeller
from Impeller.utils import evaluate_Impeller, get_paths

def train(args, data, val_mask, test_mask, x, original_x):

    model = Impeller(in_dim=data.num_node_features, hidden_dim=args.hidden_size, out_dim=data.num_node_features, dropout=args.dropout,
                    num_layers=args.num_layers, num_paths=args.num_paths, path_length=args.path_length, num_edge_types=args.num_edge_types,\
                    alpha=args.alpha, beta=args.beta, operator_type=args.operator_type)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    x = x.to(device)
    original_x = original_x.to(device)

    print(f'Model structure: {model}, \n\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    optimizer = model.setup_optimizer(args.lr, args.weight_decay, args.lr_oc, args.wd_oc)
    paths, path_types = get_paths(args, data)
    data = data.to(device)
        
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    max_no_improvement_epochs = args.patience
    best_model_state = None  # To store the state of the best model

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, paths, path_types)
        target_x = x
        
        # 直接计算全局损失
        loss = criterion(logits, target_x)
        
        loss.backward()
        optimizer.step()
        
        if epoch % args.val_epoch == 0:
            val_l1_distance, val_cosine_sim, val_rmse = evaluate_Impeller(model, x, paths, path_types, device, val_mask, original_x)
        
        if epoch % args.print_epoch == 0: 
            print(f"Epoch: {epoch}, Loss: {loss.item()}, val_l1_distance: {val_l1_distance}, val_cosine_sim: {val_cosine_sim}, val_rmse: {val_rmse}")
        
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            no_improvement_epochs = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improvement_epochs += 1
            
        if no_improvement_epochs >= max_no_improvement_epochs:
            print(f"No improvement in validation loss for {max_no_improvement_epochs} epochs, stopping.")
            break

    # Load the best model state before testing
    model.load_state_dict(best_model_state)
    test_l1_distance, test_cosine_sim, test_rmse = evaluate_Impeller(model, x, paths, path_types, device, test_mask, original_x)
    
    return model, test_l1_distance, test_cosine_sim, test_rmse

def inference(args, data, inference_mask, split_version):
    # 创建模型实例并加载训练好的参数
    model = Impeller(
        in_dim=data.num_node_features, hidden_dim=args.hidden_size, out_dim=data.num_node_features, dropout=args.dropout,
        num_layers=args.num_layers, num_paths=args.num_paths, path_length=args.path_length, num_edge_types=args.num_edge_types,
        alpha=args.alpha, beta=args.beta, operator_type=args.operator_type
    )
    model.load_state_dict(torch.load(f"trained_model_split_{split_version}.pth"))  # 加载训练好的模型参数
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')  # 设置设备（CPU或GPU）
    model = model.to(device)  # 将模型移动到设备上
    data = data.to(device)  # 将数据移动到设备上
    inference_mask = inference_mask.to(device)  # 将推理掩码移动到设备上
    x = data.x.to(device)  # 获取输入特征并移动到设备

    print(f'Model structure: {model}, \n\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # 获取路径和路径类型
    paths, path_types = get_paths(args, data)
    model.eval()  # 设置模型为评估模式

    # 使用 no_grad 禁用梯度计算
    with torch.no_grad():
        logits = model(x, paths, path_types)  # 前向传播，计算输出

    return logits  # 返回推理结果
