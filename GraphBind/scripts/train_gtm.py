import sys
import os
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
import prettytable as pt
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse
import math  # <--- [新增] 必须导入这个

import torchmetrics
from torchmetrics.classification import BinaryAUROC
from torch.utils.data import DataLoader
from data_io_af2 import NeighResidue3DPoint
from valid_metrices import *
from gtm_model import GTM
from gtm_utils import collate_fn_gtm


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--ligand", dest="ligand", help="A ligand type.")
    parser.add_argument("--psepos", dest="psepos", default='SC', help="Pseudo position.")
    parser.add_argument("--features", dest="features", default='PSSM,HMM,SS,AF,AF2', help="Feature groups.")
    parser.add_argument("--context_radius", dest="context_radius",type=int,help="Radius.")
    parser.add_argument("--trans_anno", dest="trans_anno",type=bool, default=True, help="Transfer anno.")
    parser.add_argument("--hidden_size", dest='hidden_size',type=int, default=64, help='Hidden unit size.')
    parser.add_argument("--lr", dest='lr',type=float, default=0.0001, help='Learning rate.')
    parser.add_argument("--batch_size", dest='batch_size',type=int, default=16, help='Batch size.')
    parser.add_argument("--epoch", dest='epoch',type=int, default=30, help='Training epochs.')
    parser.add_argument("--heads", dest='heads',type=int, default=4, help='Attention heads.')
    parser.add_argument("--layers", dest='layers',type=int, default=2, help='Attention layers.')
    
    parser.add_argument("--edge_radius", dest='edge_radius',type=int, default=10)
    parser.add_argument("--use_GRU", dest='use_GRU',type=bool, default=True)
    parser.add_argument("--apply_edgeattr", dest='apply_edgeattr',type=bool, default=True)
    parser.add_argument("--apply_posemb", dest='apply_posemb',type=bool, default=True)
    parser.add_argument("--aggr", dest='aggr', default='sum')
    parser.add_argument("--gru_steps", dest='gru_steps',type=int, default=4)
    parser.add_argument("--cutoff_dist", dest='cutoff_dist', type=float, default=20.0, help='Hard cutoff distance for attention.')
    
    return parser.parse_args()

def checkargs(args):
    if args.ligand is None: raise ValueError('ERROR: please input ligand type!')
    if args.context_radius is None: raise ValueError('ERROR: please input context_radius!')
    return

class Config():
    def __init__(self,args):
        self.ligand = 'P'+args.ligand if args.ligand != 'HEME' else 'PHEM'
        self.Dataset_dir = os.path.abspath('..') + '/Datasets/' + self.ligand
        self.psepos = args.psepos
        features_list = args.features.strip().split(',') 
        
        self.feature_combine = ''
        if 'PSSM' in features_list: self.feature_combine+='P'
        if 'HMM' in features_list: self.feature_combine+='H'
        if 'SS' in features_list: self.feature_combine+='S'
        if 'AF2' in features_list: self.feature_combine+='A2'
        if 'AF' in features_list: self.feature_combine+='A'

        self.dist = args.context_radius
        self.data_root_dir = '{}/{}_{}_dist{}_{}'.format(self.Dataset_dir, self.ligand, self.psepos,
                                                         self.dist, self.feature_combine)
        self.str_dataio = NeighResidue3DPoint
        
        self.batch_size = args.batch_size
        self.test_batchsize = args.batch_size
        self.epoch = args.epoch
        self.lr = args.lr
        self.hidden_size = args.hidden_size
        self.heads = args.heads
        self.layers = args.layers
        
        self.num_workers = 0
        self.early_stop_epochs = 10
        self.saved_model_num = 1
        self.model_time = None
        self.train = True
        
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.model_path = self.Dataset_dir+'/checkpoints_GTM/' + localtime
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)
        self.submodel_path = self.model_path + '/model'
        self.sublog_path = self.model_path + '/log'
        if not os.path.exists(self.submodel_path): os.makedirs(self.submodel_path)
        if not os.path.exists(self.sublog_path): os.makedirs(self.sublog_path)

    def print_config(self):
        for name, value in vars(self).items():
            print('{} = {}'.format(name, value))

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab',buffering=0)
    def write(self, message):
        self.terminal.write(message)
        try: self.log.write(message.encode('utf-8'))
        except ValueError: pass
    def close(self): self.log.close(); sys.stdout = self.terminal
    def flush(self): pass

# --- 训练函数 ---
def train(opt, device, model, learning_rate, train_data, valid_data, test_data):
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, 
                                  num_workers=opt.num_workers, collate_fn=collate_fn_gtm)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)#Adam优化
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6)
    
    criterion = nn.BCELoss(reduction='none') 
    
    model.to(device)
    
    # 手动 Loss 累加器
    
    max_metric_val = -1
    early_stop_iter = 0
    
    print('Start Training...')
    for epoch in range(opt.epoch):
        epoch_loss = 0.0
        num_batches = 0
        
        for ii, batch_data in enumerate(train_dataloader):
            if batch_data[0] is None: continue 
            
            node_features, dist_matrix, masks, labels = batch_data
            
            node_features = node_features.to(device)
            dist_matrix = dist_matrix.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            model.train()
            optimizer.zero_grad()
            
            scores = model(node_features, dist_matrix, masks) 
            
            # --- 聚合 ---
            masked_scores = scores * masks
            sum_scores = masked_scores.sum(dim=1)
            lengths = masks.sum(dim=1)
            final_scores = (sum_scores / (lengths + 1e-8)).unsqueeze(-1)
            
            loss_mat = criterion(final_scores, labels)
            loss = loss_mat.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if ii % 100 == 0:
                print(f"Epoch {epoch} Step {ii} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / (num_batches + 1e-8)
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")

        # 验证
        val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc = val(opt, device, model, valid_data, 'valid')
        
        metric_val = val_mcc 
        scheduler.step(metric_val)
        
        if metric_val > max_metric_val:
            max_metric_val = metric_val
            save_path = '{}/model_best.pth'.format(opt.submodel_path)
            # 保存模型和最佳阈值
            torch.save({'state_dict': model.state_dict(), 'best_th': val_th}, save_path)
            print(f"New Best Model Saved! MCC: {metric_val:.4f}")
            early_stop_iter = 0
        else:
            early_stop_iter += 1
            if early_stop_iter >= opt.early_stop_epochs:
                print("Early stopping...")
                break

# --- 验证/测试函数 ---
def val(opt, device, model, valid_data, dataset_type, val_th=None):
    valid_dataloader = DataLoader(valid_data, batch_size=opt.test_batchsize, shuffle=False, 
                                  num_workers=opt.num_workers, collate_fn=collate_fn_gtm)
    model.eval()
    
    # 使用 torchmetrics 计算 AUC (比 torchnet 稳健)
    auc_metric = BinaryAUROC().to(device)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in valid_dataloader:
            if batch_data[0] is None: continue
            
            node_features, dist_matrix, masks, labels = batch_data
            
            node_features = node_features.to(device)
            dist_matrix = dist_matrix.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            scores = model(node_features, dist_matrix, masks) 
            
            masked_scores = scores * masks
            sum_scores = masked_scores.sum(dim=1)
            lengths = masks.sum(dim=1)
            final_scores = (sum_scores / (lengths + 1e-8)) # [B]
            
            auc_metric.update(final_scores, labels.squeeze(-1))
            
            all_scores.append(final_scores)
            all_labels.append(labels.squeeze(-1))
            
    val_auc = auc_metric.compute().item()
    auc_metric.reset()
    
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    
    # 寻找最佳阈值
    if val_th is None:
        best_mcc = -1
        best_th = 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            pred_bi = (all_scores > th).float()
            
            # [修复] 强制转 float，并使用 math.sqrt
            TP = float(((pred_bi == 1) & (all_labels == 1)).sum().item())
            TN = float(((pred_bi == 0) & (all_labels == 0)).sum().item())
            FP = float(((pred_bi == 1) & (all_labels == 0)).sum().item())
            FN = float(((pred_bi == 0) & (all_labels == 1)).sum().item())
            
            denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
            mcc = (TP*TN - FP*FN) / (denom + 1e-8)
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th
        val_th = best_th
    
    # 计算最终指标
    pred_bi = (all_scores > val_th).float()
    
    TP = float(((pred_bi == 1) & (all_labels == 1)).sum().item())
    TN = float(((pred_bi == 0) & (all_labels == 0)).sum().item())
    FP = float(((pred_bi == 1) & (all_labels == 0)).sum().item())
    FN = float(((pred_bi == 0) & (all_labels == 1)).sum().item())
    
    sen = TP / (TP + FN + 1e-8)
    pre = TP / (TP + FP + 1e-8)
    spe = TN / (TN + FP + 1e-8)
    f1 = 2*pre*sen / (pre + sen + 1e-8)
    
    # [修复] 使用 math.sqrt
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc = (TP*TN - FP*FN) / (denom + 1e-8)
    
    print('{} result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f} spe={:.3f} MCC={:.3f} AUC={:.3f}'
          .format(dataset_type, val_th, sen, pre, f1, spe, mcc, val_auc))
    
    return val_th, sen, pre, f1, spe, mcc, val_auc

# --- 主函数 ---
def main(opt, device):
    print('Parameter Config:')
    opt.print_config()
    
    print('Loading Data...')
    train_data = opt.str_dataio(root=opt.data_root_dir, dataset='train')
    valid_data = opt.str_dataio(root=opt.data_root_dir, dataset='valid')
    test_data = opt.str_dataio(root=opt.data_root_dir, dataset='test')
    
    input_dim = train_data[0].x.shape[1]
    print(f"Input Feature Dimension: {input_dim}")
    
    print('Initializing GTM...')
    model = GTM(
        protein_in_dim=input_dim,
        protein_out_dim=opt.hidden_size,
        target_dim=1,
        fc_layer_num=2,
        atten_layer_num=opt.layers,
        atten_head=opt.heads,
        num_neighbor=30,
        drop_rate1=0.2,
        drop_rate2=0.1
    )
    
    if opt.train:
        train(opt, device, model, opt.lr, train_data, valid_data, test_data)
        
    print('===== Testing Best Model =====')
    best_model_path = '{}/model_best.pth'.format(opt.submodel_path)
    
    # 加载时要小心，因为我们存的是字典
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    best_th = checkpoint['best_th']
    
    model.to(device)
    
    val(opt, device, model, test_data, 'test', val_th=best_th)

if __name__ == '__main__':
    args = parse_args()
    checkargs(args)
    opt = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sys.stdout = Logger(opt.model_path + '/training.log')
    main(opt, device)
    sys.stdout.log.close()