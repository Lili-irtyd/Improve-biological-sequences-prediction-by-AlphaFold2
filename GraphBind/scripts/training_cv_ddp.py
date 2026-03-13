# training_cv_ddp.py

import sys
import os
import time
import datetime
import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import prettytable as pt
from torch.utils.data import ConcatDataset
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# 确保你的项目路径是正确的
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
from data_io_af2 import NeighResidue3DPoint
from ModelCode.GN_model_gru import MetaBind_MultiEdges
from valid_metrices import CFM_eval_metrics  # 假设此文件和函数存在


# ==============================================================================
# DDP (DistributedDataParallel) 设置
# ==============================================================================

def setup_ddp():
    """初始化DDP进程组"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print(f"DDP setup: Rank {local_rank} initialized on GPU {local_rank}.")
    return local_rank


def cleanup_ddp():
    """销毁DDP进程组"""
    dist.destroy_process_group()


# ==============================================================================
# 参数解析与配置
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--ligand", dest="ligand", required=True, help="Ligand type (e.g., RNA, DNA, ATP).")
    parser.add_argument("--features", dest="features", nargs='+', required=True,
                        help="Feature groups (e.g., PSSM HMM AF2).")
    parser.add_argument("--context_radius", dest="context_radius", type=int, required=True,
                        help="Radius of structure context.")
    parser.add_argument("--psepos", dest="psepos", default='SC', help="Pseudo position of residues (SC, CA, C).")
    parser.add_argument("--edge_radius", dest='edge_radius', type=int, default=10, help='Radius for graph edges.')
    parser.add_argument("--hidden_size", dest='hidden_size', type=int, default=128, help='Hidden size for GNN.')
    parser.add_argument("--lr", dest='lr', type=float, default=0.00003, help='Learning rate.')
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=32, help='Batch size PER GPU.')
    parser.add_argument("--epoch", dest='epoch', type=int, default=30, help='Max training epochs.')
    parser.add_argument("--folds", dest='folds', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument("--gru_steps", dest='gru_steps', type=int, default=4, help='The number of GNN-blocks')
    parser.add_argument("--plot_results", action='store_true',
                        help="Only generate plots from results_summary.json without training.")
    # 其他参数可以按需添加
    parser.add_argument("--trans_anno", dest="trans_anno", type=bool, default=True)
    parser.add_argument("--use_GRU", dest='use_GRU', type=bool, default=True)
    parser.add_argument("--apply_edgeattr", dest='apply_edgeattr', type=bool, default=True)
    parser.add_argument("--apply_posemb", dest='apply_posemb', type=bool, default=True)
    parser.add_argument("--aggr", dest='aggr', default='sum')
    return parser.parse_args()


class Config():
    def __init__(self, args):
        self.ligand = 'P' + args.ligand if args.ligand != 'HEME' else 'PHEM'
        self.Dataset_dir = os.path.abspath('..') + '/Datasets/' + self.ligand
        self.psepos = args.psepos
        feature_map = {'PSSM': 'P', 'HMM': 'H', 'SS': 'S', 'AF2': 'A2', 'AF': 'A'}

        # 使用这个正确的映射来生成 feature_combine 字符串
        self.feature_combine = ''.join(
            feature_map[f] for f in args.features if f in feature_map
        )
        self.dist = args.context_radius
        self.data_root_dir = f'{self.Dataset_dir}/{self.ligand}_{self.psepos}_dist{self.dist}_{self.feature_combine}'
        if not os.path.exists(self.data_root_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_root_dir}")
        self.str_dataio = NeighResidue3DPoint
        self.str_model = MetaBind_MultiEdges
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.folds = args.folds
        self.early_stop_epochs = 10
        self.max_metric = 'MCC'  # Metric for early stopping
        # 将其他参数也添加到config中
        self.radius_list = [args.edge_radius]
        self.hidden_size = args.hidden_size
        self.gru_steps = args.gru_steps
        self.dropratio = 0.5
        self.bias = True
        self.L2_weight = 0
        self.edge_aggr = ['add' if args.aggr == 'sum' else args.aggr]
        self.node_aggr = ['add' if args.aggr == 'sum' else args.aggr]
        self.stack_method = 'GRU' if args.use_GRU else 'Composed'
        self.apply_edgeattr = args.apply_edgeattr
        self.apply_nodeposemb = args.apply_posemb
        self.max_nn = 40
        self.num_workers = 4  # 可以根据你的CPU核心数调整

        localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.output_path = f'./output/{localtime}_{self.ligand}_{self.feature_combine}'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path,exist_ok=True)

    def print_config(self, logger):
        for name, value in vars(self).items():
            logger.info(f'{name} = {value}')


# ==============================================================================
# 评估函数
# ==============================================================================

def evaluate(model, data_loader, device, desc="Validation"):
    """评估模型，返回所有重要指标"""
    model.eval()
    all_scores, all_targets = [], []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            score = model(data).float()
            all_scores.append(score.cpu())
            all_targets.append(data.y.cpu())

    all_scores = torch.cat(all_scores).numpy()
    all_targets = torch.cat(all_targets).numpy()

    auc = roc_auc_score(all_targets, all_scores)
    aupr = average_precision_score(all_targets, all_scores)

    # 基于MCC寻找最佳阈值
    precision, recall, thresholds = precision_recall_curve(all_targets, all_scores)
    best_mcc, best_th = -1, 0.5
    # 在阈值中采样以加速
    for th in np.concatenate([np.linspace(0, 1, 101), thresholds[::max(1, len(thresholds) // 100)]]):
        pred_bi = (all_scores > th).astype(int)
        tn, fp, fn, tp = np.bincount((2 * all_targets + pred_bi).astype(int), minlength=4)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
                    tn + fp) * (tn + fn) > 0 else 0
        if mcc > best_mcc:
            best_mcc = mcc
            best_th = th

    # 使用最佳阈值计算其他指标
    tn, fp, fn, tp = np.bincount((2 * all_targets + (all_scores > best_th).astype(int)).astype(int), minlength=4)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {'th': best_th, 'rec': rec, 'pre': pre, 'f1': f1, 'spe': spe, 'mcc': best_mcc, 'auc': auc, 'aupr': aupr}

    # 打印格式化结果
    log_str = (
        f"{desc} Result: th={metrics['th']:.3f} | Sen={metrics['rec']:.3f} | Pre={metrics['pre']:.3f} | "
        f"F1={metrics['f1']:.3f} | Spe={metrics['spe']:.3f} | MCC={metrics['mcc']:.3f} | "
        f"AUC={metrics['auc']:.3f} | AUPR={metrics['aupr']:.3f}"
    )
    print(log_str)

    return metrics


# ==============================================================================
# 训练流程
# ==============================================================================

def train_fold(opt, rank, world_size, train_dataset, val_dataset, fold_num):
    """在一个fold上进行训练和验证"""

    # DDP: 为每个fold的训练集创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                              num_workers=opt.num_workers, pin_memory=True)

    # 验证集通常不需要分布式，在主进程上评估即可
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size * 2, shuffle=False, num_workers=opt.num_workers)

    # DDP: 为每个fold重新初始化模型和优化器
    model = opt.str_model(
        edge_aggr=opt.edge_aggr, node_aggr=opt.node_aggr, gru_steps=opt.gru_steps,
        x_ind=train_dataset[0].x.shape[1] + 1, edge_ind=2,
        x_hs=opt.hidden_size, e_hs=opt.hidden_size, u_hs=opt.hidden_size,
        dropratio=opt.dropratio, bias=opt.bias, edge_method='radius', r_list=opt.radius_list,
        dist=opt.dist, max_nn=opt.max_nn, stack_method=opt.stack_method,
        apply_edgeattr=opt.apply_edgeattr, apply_nodeposemb=opt.apply_nodeposemb
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=opt.lr, weight_decay=opt.L2_weight)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-7)

    best_metric_val = -1
    best_model_state = None
    early_stop_counter = 0

    for epoch in range(opt.epoch):
        # DDP: 设置epoch以确保不同epoch有不同shuffle
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(rank)
            optimizer.zero_grad()
            score = ddp_model(data)
            loss = criterion(score, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # DDP: 只在主进程上进行验证和打印
        if rank == 0:
            print(
                f"--- Fold {fold_num + 1}, Epoch {epoch + 1}/{opt.epoch}, Train Loss: {total_loss / len(train_loader):.5f} ---")
            val_metrics = evaluate(ddp_model.module, val_loader, rank, "Validation")
            metric_val = val_metrics[opt.max_metric.lower()]
            scheduler.step(metric_val)

            if metric_val > best_metric_val:
                best_metric_val = metric_val
                best_model_state = ddp_model.module.state_dict()
                early_stop_counter = 0
                print(f"** New best model found! {opt.max_metric}: {best_metric_val:.4f} **")
            else:
                early_stop_counter += 1

            if early_stop_counter >= opt.early_stop_epochs:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    # DDP: 确保所有进程都完成了训练
    dist.barrier()
    return best_model_state


# ==============================================================================
# 结果保存与绘图
# ==============================================================================
def save_results_for_plotting(opt, avg_auc, std_auc, avg_aupr, std_aupr):
    results_file = 'results_summary.json'
    all_results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)

    all_results[opt.feature_combine] = {
        'avg_auc': avg_auc, 'std_auc': std_auc,
        'avg_aupr': avg_aupr, 'std_aupr': std_aupr
    }

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results for '{opt.feature_combine}' saved to {results_file}")


def plot_results():
    results_file = 'results_summary.json'
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found.")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    features = sorted(results.keys())
    if not features:
        print("No results to plot.")
        return

    avg_aucs = [results[f]['avg_auc'] for f in features]
    std_aucs = [results[f]['std_auc'] for f in features]
    avg_auprs = [results[f]['avg_aupr'] for f in features]
    std_auprs = [results[f]['std_aupr'] for f in features]

    x = np.arange(len(features))

    # 绘制 AUC 图
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(x, avg_aucs, yerr=std_aucs, capsize=5, color='skyblue', ecolor='gray')
    ax.set_ylabel('AUC Score', fontsize=14)
    ax.set_title('Average AUC by Feature Combination', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=12)
    ax.bar_label(bars, padding=3, fmt='%.3f')
    ax.set_ylim(bottom=max(0, min(avg_aucs) - 0.1), top=1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plt.savefig('average_auc_comparison.png')
    print("Saved AUC comparison plot to 'average_auc_comparison.png'")
    plt.close()

    # 绘制 AUPR 图
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(x, avg_auprs, yerr=std_auprs, capsize=5, color='salmon', ecolor='gray')
    ax.set_ylabel('AUPR Score', fontsize=14)
    ax.set_title('Average AUPR by Feature Combination', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=12)
    ax.bar_label(bars, padding=3, fmt='%.3f')
    ax.set_ylim(bottom=max(0, min(avg_auprs) - 0.1), top=1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plt.savefig('average_aupr_comparison.png')
    print("Saved AUPR comparison plot to 'average_aupr_comparison.png'")
    plt.close()


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    args = parse_args()

    if args.plot_results:
        plot_results()
        return

    # DDP 初始化
    rank = setup_ddp()
    world_size = dist.get_world_size()

    opt = Config(args)

    # 日志和配置只在主进程上处理
    if rank == 0:
        log_file = os.path.join(opt.output_path, 'training.log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        logging.info("=========== Configuration ===========")
        opt.print_config(logging)
        logging.info("=====================================")

    # 加载数据
    train_dataset = NeighResidue3DPoint(root=opt.data_root_dir, dataset='train')
    valid_dataset = NeighResidue3DPoint(root=opt.data_root_dir, dataset='valid')
    test_data = NeighResidue3DPoint(root=opt.data_root_dir, dataset='test')
    test_loader = DataLoader(test_data, batch_size=opt.batch_size * 2, shuffle=False, num_workers=opt.num_workers)

    # 使用 ConcatDataset 将 train 和 valid 数据集合并
    train_val_dataset = ConcatDataset([train_dataset, valid_dataset])
    # 交叉验证
    kf = KFold(n_splits=opt.folds, shuffle=True, random_state=42)
    fold_test_aucs, fold_test_auprs = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        if rank == 0:
            logging.info(f"\n=============== Starting Fold {fold + 1}/{opt.folds} ================")

        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)

        best_model_state = train_fold(opt, rank, world_size, train_subset, val_subset, fold)

        # 测试只在主进程上进行
        if rank == 0 and best_model_state is not None:
            logging.info(f"--- Evaluating best model of Fold {fold + 1} on Test Set ---")
            # 重新创建一个模型实例来加载状态
            test_model = opt.str_model(
                edge_aggr=opt.edge_aggr, node_aggr=opt.node_aggr, gru_steps=opt.gru_steps,
                x_ind=train_val_dataset[0].x.shape[1] + 1, edge_ind=2,
                x_hs=opt.hidden_size, e_hs=opt.hidden_size, u_hs=opt.hidden_size,
                dropratio=opt.dropratio, bias=opt.bias, edge_method='radius', r_list=opt.radius_list,
                dist=opt.dist, max_nn=opt.max_nn, stack_method=opt.stack_method,
                apply_edgeattr=opt.apply_edgeattr, apply_nodeposemb=opt.apply_nodeposemb
            ).to(rank)
            test_model.load_state_dict(best_model_state)

            test_metrics = evaluate(test_model, test_loader, rank, "Test")
            fold_test_aucs.append(test_metrics['auc'])
            fold_test_auprs.append(test_metrics['aupr'])

            # 保存当前fold的最佳模型
            torch.save(best_model_state, os.path.join(opt.output_path, f'model_fold_{fold + 1}.pth'))

    # 总结和保存结果 (只在主进程)
    if rank == 0:
        avg_auc = np.mean(fold_test_aucs)
        std_auc = np.std(fold_test_aucs)
        avg_aupr = np.mean(fold_test_auprs)
        std_aupr = np.std(fold_test_auprs)

        logging.info("\n================= Cross-Validation Final Results ==================")
        logging.info(f"Feature Combination: {opt.feature_combine}")
        logging.info(f"Test AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        logging.info(f"Test AUPR: {avg_aupr:.4f} ± {std_aupr:.4f}")
        logging.info("==================================================================")

        save_results_for_plotting(opt, avg_auc, std_auc, avg_aupr, std_aupr)

    cleanup_ddp()


if __name__ == '__main__':
    # DDP需要这个主防护
    # 在 __main__ 保护块中添加 logging 的基本配置
    # 这样即使在非主进程中，也可以打印一些信息（如果需要的话）
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (Rank %(process)d) %(message)s')

    main()