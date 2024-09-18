import argparse
import sys

def create_args():
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_paths', type=int, default=10)
    parser.add_argument('--path_length', type=int, default=8)
    parser.add_argument('--num_edge_types', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--operator_type', type=str, choices=['global', 'shared_layer', 'shared_channel', 'independent'], default='independent')
    parser.add_argument('--spatial_walk_p', type=int, default=1)
    parser.add_argument('--spatial_walk_q', type=int, default=1)
    parser.add_argument('--gene_walk_p', type=int, default=1)
    parser.add_argument('--gene_walk_q', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=64)

    # Training process
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--lr_oc", type=float, default=1e-2)
    parser.add_argument("--wd_oc", type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--print_epoch', type=int, default=10)
    parser.add_argument('--val_epoch', type=int, default=10)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)

    # Parse arguments, ignoring any that are not recognized
    args = parser.parse_args(args=[] if sys.argv[0].endswith('ipykernel_launcher.py') else None)

    return args
