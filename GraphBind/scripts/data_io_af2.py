
##labels是从原始数据集文本文件 (例如 DNA-new_Train.txt) 中 读取 的。
import pickle
import pandas as pd
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm, trange
import random
import torch
from torch_geometric.data import InMemoryDataset, Data
import prettytable as pt
import math
import argparse
import re
import gc
import tempfile
import shutil
import os
import pickle
import numpy as np
import psutil


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--ligand", dest="ligand", help="A ligand type. It can be chosen from DNA,RNA,CA,MG,MN,ATP,HEME.")
    parser.add_argument("--psepos", dest="psepos", default='SC',
                        help="Pseudo position of residues. SC, CA, C stand for centroid of side chain, alpha-C atom and centroid of residue, respectively.")
    parser.add_argument("--features", dest="features", nargs='+',
                        help="Feature groups. Multiple features should be separated by spaces. You can combine features from PSSM, HMM, SS, AF, and AF2.")###修改逻辑，使得其匹配完整的特征名，而不是子串

    parser.add_argument("--context_radius", dest="context_radius",type=int, help="Radius of structure context.")
    parser.add_argument("--trans_anno", dest="trans_anno",type=bool, default=True,
                        help="Transfer binding annotations for DNA-(RNA-)binding protein training data sets or not.")
    parser.add_argument("--tvseed", dest='tvseed',type=int, default=1995, help='The random seed used to separate the validation set from training set.')
    return parser.parse_args()

# 2. 修改参数检查函数
def checkargs(args):
    if args.ligand not in ['DNA','RNA','CA','MN','MG','ATP','HEME']:
        print('ERROR: ligand "{}" is not supported by GraphBind!'.format(args.ligand))
        raise ValueError
    if args.psepos not in ['SC','CA','C']:
        print('ERROR: pseudo position of a residue "{}" is not supported by GraphBind!'.format(args.psepos))
        raise ValueError
    features = args.features
    for feature in features:
        if feature not in ['PSSM','HMM','SS','AF','AF2']:  # 添加AF2支持
            print('ERROR: feature "{}" is not supported by GraphBind!'.format(feature))
            raise ValueError
    if args.context_radius<=0:
        print('ERROR: radius of structure context should be positive!')
        raise ValueError
    return

import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import gc
import tempfile
import shutil


class NeighResidue3DPoint(InMemoryDataset):
    def __init__(self, root, dataset='train', transform=None, pre_transform=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform)

        # 直接加载最终处理好的单个文件
        if dataset == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif dataset == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif dataset == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"{dataset} loaded with Numres = {self.data.y.shape[0]}")

    @property
    def raw_file_names(self):
        return ['train_data.pkl', 'valid_data.pkl', 'test_data.pkl']

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        """
        超级内存优化的处理函数 - 流式处理，避免大内存占用
        """
        print("开始超级内存优化的数据处理...")

        for s, dataset in enumerate(['train', 'valid', 'test']):
            print(f"\n处理 {dataset} 数据集...")

            raw_file = os.path.join(self.raw_dir, f"{dataset}_data.pkl")
            final_file = self.processed_paths[s]

            if not os.path.exists(raw_file):
                print(f"跳过 {dataset}: 原始文件不存在")
                continue

            # 使用流式处理，避免一次性加载所有数据
            self._process_dataset_streaming(raw_file, final_file, dataset)

        print("所有数据集处理完成！")

    def _process_dataset_streaming(self, raw_file, output_file, dataset_name):
        """
        流式处理单个数据集，最小化内存使用
        """
        # 创建临时目录用于存储中间文件
        temp_dir = tempfile.mkdtemp(prefix=f"{dataset_name}_temp_")
        print(f"使用临时目录: {temp_dir}")

        try:
            # 第一步：将原始数据转换为小文件
            print("步骤1: 转换原始数据为小文件...")
            num_files = self._convert_to_small_files(raw_file, temp_dir)

            # 第二步：逐个处理小文件并合并
            print("步骤2: 处理小文件并合并...")
            self._merge_small_files_streaming(temp_dir, output_file, num_files)

        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
                print(f"清理临时目录: {temp_dir}")
            except:
                print(f"警告: 无法清理临时目录 {temp_dir}")

    def _convert_to_small_files(self, raw_file, temp_dir):
        """
        将大的pickle文件转换为多个小文件
        """
        print("正在加载原始数据...")
        with open(raw_file, 'rb') as f:
            data_dict, seqlist = pickle.load(f)

        # 按序列分组，每个文件包含少量序列
        seqs_per_file = 10  # 每个文件最多10个序列
        file_count = 0

        for i in range(0, len(seqlist), seqs_per_file):
            batch_seqs = seqlist[i:i + seqs_per_file]
            batch_data = []

            for seq in batch_seqs:
                if seq in data_dict:
                    seq_data = data_dict[seq]
                    for res_data in seq_data:
                        try:
                            # 立即转换为torch tensor
                            node_feas = torch.tensor(res_data['node_feas'], dtype=torch.float32)
                            pos = torch.tensor(res_data['pos'], dtype=torch.float32)
                            label = torch.tensor([res_data['label']], dtype=torch.float32)

                            data = Data(x=node_feas, pos=pos, y=label)
                            batch_data.append(data)

                        except Exception as e:
                            print(f"跳过有问题的残基数据: {e}")
                            continue

            if batch_data:
                # 立即collate并保存，然后释放内存
                try:
                    collated_data, collated_slices = self.collate(batch_data)
                    temp_file = os.path.join(temp_dir, f"batch_{file_count}.pt")
                    torch.save((collated_data, collated_slices), temp_file)
                    print(f"保存临时文件 {file_count}: {len(batch_data)} 个残基")
                    file_count += 1
                except Exception as e:
                    print(f"保存临时文件 {file_count} 失败: {e}")
                    continue

            # 清理内存
            batch_data = []
            gc.collect()

        # 清理原始数据
        data_dict = None
        seqlist = None
        gc.collect()

        return file_count

    def _merge_small_files_streaming(self, temp_dir, output_file, num_files):
        """
        流式合并小文件，避免同时加载所有数据
        """
        print(f"开始合并 {num_files} 个临时文件...")

        # 使用迭代器方式逐步合并，而不是一次性加载所有数据
        all_data = None
        all_slices = None
        total_residues = 0

        for i in tqdm(range(num_files), desc="合并文件"):
            temp_file = os.path.join(temp_dir, f"batch_{i}.pt")

            if not os.path.exists(temp_file):
                continue

            try:
                # 加载单个批次
                batch_data, batch_slices = torch.load(temp_file)
                batch_size = batch_data.y.shape[0]

                if all_data is None:
                    # 第一个文件，直接使用
                    all_data = batch_data
                    all_slices = batch_slices
                else:
                    # 合并到现有数据
                    all_data, all_slices = self._merge_two_batches(
                        all_data, all_slices, batch_data, batch_slices
                    )

                total_residues += batch_size
                print(f"合并批次 {i}: +{batch_size} 残基, 总计: {total_residues}")

                # 清理批次数据
                batch_data = None
                batch_slices = None

                # 每合并5个批次就强制垃圾回收
                if (i + 1) % 5 == 0:
                    gc.collect()

            except Exception as e:
                print(f"合并批次 {i} 失败: {e}")
                continue

        # 保存最终结果
        if all_data is not None:
            try:
                torch.save((all_data, all_slices), output_file)
                print(f"最终文件保存成功: {output_file}")
                print(f"总残基数: {total_residues}")
            except Exception as e:
                print(f"保存最终文件失败: {e}")
                # 尝试分块保存
                self._save_in_emergency_chunks(all_data, all_slices, output_file)

    def _merge_two_batches(self, data1, slices1, data2, slices2):
        """
        修复版本：正确合并两个批次的数据和slices
        """
        # 合并数据张量
        merged_data = data1.__class__()
        merged_slices = {}

        print(f"合并前检查:")
        print(f"  data1: x={data1.x.shape}, pos={data1.pos.shape}, y={data1.y.shape}")
        print(f"  data2: x={data2.x.shape}, pos={data2.pos.shape}, y={data2.y.shape}")
        print(f"  slices1: {[(k, v.shape, v[-1].item()) for k, v in slices1.items()]}")
        print(f"  slices2: {[(k, v.shape, v[-1].item()) for k, v in slices2.items()]}")

        # 对每个key单独处理
        for key in data1.keys:
            if key in data2:
                # 合并数据
                merged_data[key] = torch.cat([data1[key], data2[key]], dim=0)

                # 计算该key的正确offset
                offset = data1[key].shape[0]  # 使用该key对应数据的第一维大小

                # 调整第二个批次的slices
                adjusted_slices2 = slices2[key] + offset

                # 合并slices（去掉第二个批次的第一个元素，避免重复）
                merged_slices[key] = torch.cat([slices1[key], adjusted_slices2[1:]], dim=0)

                print(f"  合并 {key}: 数据 {merged_data[key].shape}, slices {merged_slices[key].shape}")

        # 验证合并结果的正确性
        try:
            for key in merged_slices:
                max_slice = merged_slices[key][-1].item()
                data_size = merged_data[key].shape[0]
                if max_slice != data_size:
                    print(f"警告: {key} 的最大slice {max_slice} != 数据大小 {data_size}")
        except Exception as e:
            print(f"验证合并结果时出错: {e}")

        return merged_data, merged_slices


    def _batch_to_data_list(self, batch_data, batch_slices):
        """
        将批次数据转换回Data对象列表
        """
        data_list = []
        num_graphs = batch_slices['y'].shape[0] - 1

        for i in range(num_graphs):
            data_item = Data()

            for key in batch_data.keys:
                start = batch_slices[key][i].item()
                end = batch_slices[key][i + 1].item()

                if start < end:  # 确保索引有效
                    data_item[key] = batch_data[key][start:end].clone()

            data_list.append(data_item)

        return data_list

    def _merge_two_batches_fallback(self, data1, slices1, data2, slices2):
        """
        回退方案：使用简单拼接（可能不完全正确，但至少不会崩溃）
        """
        merged_data = data1.__class__()
        merged_slices = {}

        for key in data1.keys:
            if key in data2:
                merged_data[key] = torch.cat([data1[key], data2[key]], dim=0)

                # 简单的slices处理
                offset = data1[key].shape[0]
                adjusted_slices2 = slices2[key] + offset
                merged_slices[key] = torch.cat([slices1[key][:-1], adjusted_slices2], dim=0)

        return merged_data, merged_slices

    # 修改主处理函数，使用更安全的合并方法
    def _merge_small_files_streaming_fixed(self, temp_dir, output_file, num_files):
        """
        修复版本的流式合并
        """
        print(f"开始合并 {num_files} 个临时文件...")

        all_data_list = []  # 改用列表存储所有数据
        total_residues = 0

        # 分批加载和收集数据
        batch_size = 5  # 每次合并5个文件

        for batch_start in range(0, num_files, batch_size):
            batch_end = min(batch_start + batch_size, num_files)
            batch_data_list = []

            print(f"处理文件批次 {batch_start}-{batch_end - 1}")

            for i in range(batch_start, batch_end):
                temp_file = os.path.join(temp_dir, f"batch_{i}.pt")

                if not os.path.exists(temp_file):
                    continue

                try:
                    batch_data, batch_slices = torch.load(temp_file)

                    # 转换为Data列表
                    data_list = self._batch_to_data_list(batch_data, batch_slices)
                    batch_data_list.extend(data_list)

                    total_residues += batch_data.y.shape[0]
                    print(f"  加载批次 {i}: {batch_data.y.shape[0]} 残基")

                except Exception as e:
                    print(f"加载批次 {i} 失败: {e}")
                    continue

            all_data_list.extend(batch_data_list)

            # 定期清理内存
            if len(all_data_list) > 10000:  # 如果数据太多，先保存部分
                print("数据过多，先保存部分数据...")
                self._save_partial_data(all_data_list, output_file, batch_start)
                all_data_list = []

            gc.collect()

        # 保存最终数据
        if all_data_list:
            try:
                print(f"最终合并 {len(all_data_list)} 个残基...")
                final_data, final_slices = self.collate(all_data_list)
                torch.save((final_data, final_slices), output_file)
                print(f"成功保存: {output_file}, 总残基数: {total_residues}")
            except Exception as e:
                print(f"最终保存失败: {e}")
                self._save_as_chunks(all_data_list, output_file)

    def _save_partial_data(self, data_list, base_file, chunk_id):
        """
        保存部分数据为临时chunk
        """
        chunk_file = base_file.replace('.pt', f'_partial_{chunk_id}.pt')
        try:
            chunk_data, chunk_slices = self.collate(data_list)
            torch.save((chunk_data, chunk_slices), chunk_file)
            print(f"保存部分数据: {chunk_file}")
        except Exception as e:
            print(f"保存部分数据失败: {e}")

    def _save_as_chunks(self, data_list, base_file):
        """
        将数据列表保存为多个chunk文件
        """
        chunk_size = 1000
        base_name = base_file.replace('.pt', '')

        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i:i + chunk_size]
            try:
                chunk_data, chunk_slices = self.collate(chunk)
                chunk_file = f"{base_name}_chunk_{i // chunk_size}.pt"
                torch.save((chunk_data, chunk_slices), chunk_file)
                print(f"保存chunk: {chunk_file} ({len(chunk)} 残基)")
            except Exception as e:
                print(f"保存chunk {i // chunk_size} 失败: {e}")

    def _save_in_emergency_chunks(self, data, slices, base_output_file):
        """
        紧急分块保存，当内存不足时使用
        """
        print("内存不足，启用紧急分块保存...")

        chunk_size = 500  # 每块500个残基
        total_residues = data.y.shape[0]
        base_name = base_output_file.replace('.pt', '')

        for start_idx in range(0, total_residues, chunk_size):
            end_idx = min(start_idx + chunk_size, total_residues)

            # 创建子数据
            chunk_data = data.__class__()
            chunk_slices = {}

            for key in data.keys:
                # 提取chunk数据
                start_pos = slices[key][start_idx].item()
                end_pos = slices[key][end_idx].item()
                chunk_data[key] = data[key][start_pos:end_pos]

                # 调整slice索引
                chunk_slices[key] = slices[key][start_idx:end_idx + 1] - start_pos

            # 保存chunk
            chunk_file = f"{base_name}_chunk_{start_idx // chunk_size}.pt"
            torch.save((chunk_data, chunk_slices), chunk_file)
            print(f"保存紧急块: {chunk_file} ({end_idx - start_idx} 残基)")


# 进一步的内存优化设置
def set_memory_optimization():
    """设置各种内存优化参数"""
    import os

    # PyTorch内存优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # 限制线程数减少内存使用
    torch.set_num_threads(1)

    # 如果使用CUDA，设置内存分配策略
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.8)
# 额外的内存监控函数
def get_memory_usage():
    """获取当前内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024  # GB


def monitor_memory(func):
    """内存监控装饰器"""

    def wrapper(*args, **kwargs):
        mem_before = get_memory_usage()
        print(f"执行前内存使用: {mem_before:.2f} GB")

        result = func(*args, **kwargs)

        mem_after = get_memory_usage()
        print(f"执行后内存使用: {mem_after:.2f} GB")
        print(f"内存变化: {mem_after - mem_before:.2f} GB")

        return result

    return wrapper

def Create_NeighResidue3DPoint(psepos, dist, feature_dir, raw_dir, seqanno, feature_combine,
                               train_list, valid_list, test_list, ligand, batch_size=20):
    """
    内存优化版 Create_NeighResidue3DPoint，按序列处理，防止大数据集被kill。
    输出结构与原始函数完全一致。
    """
    os.makedirs(raw_dir, exist_ok=True)

    # 1. 加载特征
    with open(feature_dir + '/' + ligand + '_psepos_{}.pkl'.format(psepos), 'rb') as f:
        residue_psepos = pickle.load(f)
    with open(feature_dir+'/'+ligand+'_residue_feas_{}.pkl'.format(feature_combine),'rb') as f:
        residue_feas = pickle.load(f)

    # 2. 按数据集处理
    for dataset_name, seqlist in zip(['train', 'valid', 'test'],
                                     [train_list, valid_list, test_list]):

        data_dict = {}

        # 逐序列处理，减少内存占用
        for start in tqdm(range(0, len(seqlist), batch_size), desc=f'Processing {dataset_name}'):
            batch = seqlist[start:start+batch_size]

            for seq in batch:
                try:
                    feas = residue_feas[seq].astype('float32')
                    pos = residue_psepos[seq].astype('float32')
                    label = np.array(list(map(int, list(seqanno[seq]['anno']))), dtype='float32')

                    seq_data = []
                    for i in range(len(label)):
                        res_psepos = pos[i]
                        res_dist = np.sqrt(np.sum((pos - res_psepos) ** 2, axis=1))
                        neigh_index = np.where(res_dist < dist)[0]

                        res_pos = pos[neigh_index] - res_psepos
                        res_feas = feas[neigh_index]
                        res_label = label[i]

                        res_data = {
                            'node_feas': res_feas,
                            'pos': res_pos,
                            'label': res_label,
                            'neigh_index': neigh_index.astype('int32')
                        }
                        seq_data.append(res_data)

                    data_dict[seq] = seq_data

                except KeyError:
                    print(f"[Warning] {seq} not found in residue features or psepos. Skipping.")
                    continue
                except Exception as e:
                    print(f"[Warning] Error processing {seq}: {e}. Skipping.")
                    continue

            # 分批处理完成后不立即写入，保留在 data_dict 中，防止内存峰值过大
            # 可以在 batch 内清理临时变量减少内存
            del batch

        # 3. 保存整个数据集到文件，与原函数结构一致
        out_file = os.path.join(raw_dir, f'{dataset_name}_data.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump([data_dict, seqlist], f)

        # 清理内存
        del data_dict

        print(f"Saved {dataset_name} dataset to {out_file}")

    print("All datasets processed successfully.")

def extract_seqid_from_filename_robust(filename):
    """
    Extracts the sequence ID (PDBID_ChainID) from the new filename format (e.g., 3pla_L.npy).
    """
    import re # Make sure re is imported

    if not isinstance(filename, str) or not filename.endswith('.npy'):
        return None

    # Get the base name without the .npy extension
    base_name = filename[:-4] # More robust than replace

    # Check if the base_name matches the PDBID_ChainID format
    # Example: 3pla_L (4 alphanumeric chars, underscore, 1 or 2 alphanumeric chars)
    match = re.match(r'^([a-zA-Z0-9]{4})_([a-zA-Z0-9]{1,2})$', base_name)
    if match:
        return base_name # Return the full PDBID_ChainID
    else:
        # Optional: Add a warning if the format doesn't match
        # print(f"Warning: Filename {filename} does not match expected PDBID_ChainID.npy format.")
        return None

# 定义每种氨基酸残基的原子特征（电荷、氢原子数、是否芳环等），做归一化处理。
def def_atom_features():
    A = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 3, 0]}
    V = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 3, 0],
         'CG2': [0, 3, 0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 1, 1]}
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 1], 'CG': [0, 2, 1],
         'CD': [0, 2, 1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 1, 0],
         'CD1': [0, 3, 0], 'CD2': [0, 3, 0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 2, 0],
         'CG2': [0, 3, 0], 'CD1': [0, 3, 0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 2, 0], 'CD': [0, 2, 0], 'NE': [0, 1, 0], 'CZ': [1, 0, 0], 'NH1': [0, 2, 0], 'NH2': [0, 2, 0]}
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [-1, 0, 0],
         'OD1': [-1, 0, 0], 'OD2': [-1, 0, 0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [-1, 0, 0], 'OE1': [-1, 0, 0], 'OE2': [-1, 0, 0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'OG': [0, 1, 0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'OG1': [0, 1, 0],
         'CG2': [0, 3, 0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'SG': [-1, 1, 0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 0, 0],
         'OD1': [0, 0, 0], 'ND2': [0, 2, 0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 0, 0], 'OE1': [0, 0, 0], 'NE2': [0, 2, 0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'ND1': [-1, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'NE2': [-1, 1, 1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 2, 0], 'CE': [0, 2, 0], 'NZ': [0, 3, 1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 0, 1],
         'OH': [-1, 1, 0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'SD': [0, 0, 0], 'CE': [0, 3, 0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 0, 1], 'NE1': [0, 1, 1], 'CE2': [0, 0, 1], 'CE3': [0, 1, 1],
         'CZ2': [0, 1, 1], 'CZ3': [0, 1, 1], 'CH2': [0, 1, 1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                     'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0] / 2 + 0.5, i_fea[1] / 3, i_fea[2]]

    return atom_features


def build_flexible_seqid_mapping(af2_dir):
    """
    构建灵活的序列ID映射，处理链ID不一致问题
    """
    filename_mapping = {}
    chain_id_variants = {}  # 存储链ID变体映射

    for folder in ['train', 'test']:
        folder_path = os.path.join(af2_dir, folder)
        if not os.path.exists(folder_path):
            continue

        try:
            all_files = os.listdir(folder_path)
            npy_files = [f for f in all_files if f.endswith('.npy')]

            for filename in npy_files:
                seqid = extract_seqid_from_filename_robust(filename)
                if seqid:
                    filepath = os.path.join(folder_path, filename)
                    filename_mapping[seqid] = filepath

                    # 同时处理链ID变体
                    parts = seqid.split('_')
                    if len(parts) == 2:
                        pdb_id, chain_id = parts

                        # 创建双字符链ID变体
                        if len(chain_id) == 1:
                            double_chain_id = chain_id + chain_id  # a -> aa
                            variant_seqid = f"{pdb_id}_{double_chain_id}"
                            chain_id_variants[variant_seqid] = seqid

        except Exception as e:
            print(f"扫描文件夹 {folder_path} 时出错: {e}")

    return filename_mapping, chain_id_variants

def safe_load_numpy(filepath):
    """
    安全加载numpy文件，处理pickle问题
    """
    try:
        # 首先尝试不允许pickle
        return np.load(filepath, allow_pickle=False)
    except ValueError as e:
        if "Object arrays cannot be loaded when allow_pickle=False" in str(e):
            # 如果出现pickle错误，允许pickle加载
            try:
                return np.load(filepath, allow_pickle=True)
            except Exception as e2:
                print(f"即使允许pickle也无法加载 {filepath}: {e2}")
                return None
        else:
            print(f"加载numpy文件出错 {filepath}: {e}")
            return None
    except Exception as e:
        print(f"未知错误加载 {filepath}: {e}")
        return None


def cal_AF2_custom_naming_complete_fix(ligand, seq_list, af2_dir, feature_dir):
    """
    完整修复版本的AF2特征处理函数
    """
    print(f"开始处理AF2特征，根目录: {af2_dir}")

    # 检查根目录是否存在
    if not os.path.exists(af2_dir):
        print(f"错误: AF2根目录不存在 - {af2_dir}")
        return {}

    # 构建灵活的文件映射
    print("构建灵活的文件名映射...")
    filename_mapping, chain_id_variants = build_flexible_seqid_mapping(af2_dir)

    print(f"找到 {len(filename_mapping)} 个直接匹配的AF2文件")
    print(f"找到 {len(chain_id_variants)} 个链ID变体映射")

    # 合并映射
    all_mappings = filename_mapping.copy()
    for variant_id, original_id in chain_id_variants.items():
        if variant_id not in all_mappings:
            all_mappings[variant_id] = filename_mapping[original_id]

    print(f"总映射数量: {len(all_mappings)}")

    # 显示一些映射示例
    if len(all_mappings) > 0:
        print("\n映射示例:")
        for i, (seqid, filepath) in enumerate(list(all_mappings.items())[:5]):
            print(f"  {seqid} -> {os.path.basename(filepath)}")
        if len(all_mappings) > 5:
            print(f"  ... 还有 {len(all_mappings) - 5} 个映射")

    # 分析未匹配的序列
    print(f"\n分析序列匹配情况 (总序列数: {len(seq_list)}):")
    matched_seqs = []
    unmatched_seqs = []

    for seqid in seq_list:
        if seqid in all_mappings:
            matched_seqs.append(seqid)
        else:
            unmatched_seqs.append(seqid)

    print(f"直接匹配: {len(matched_seqs)}")
    print(f"未匹配: {len(unmatched_seqs)}")

    # 显示未匹配序列的分析
    if len(unmatched_seqs) > 0:
        print(f"\n未匹配序列分析 (显示前10个):")
        for seqid in unmatched_seqs[:10]:
            print(f"  缺失: {seqid}")

            # 尝试找到相似的序列ID
            parts = seqid.split('_')
            if len(parts) == 2:
                pdb_id, chain_id = parts

                # 查找相同PDB的其他链
                similar = []
                for available_id in all_mappings.keys():
                    if available_id.startswith(pdb_id + '_'):
                        similar.append(available_id)

                if similar:
                    print(f"    相同PDB的可用链: {similar[:3]}")

    # 加载特征
    af2_dict = {}
    found_count = 0
    not_found_count = 0
    error_count = 0

    print(f"\n开始加载AF2特征...")

    for seqid in tqdm(seq_list):
        if seqid in all_mappings:
            filepath = all_mappings[seqid]

            # 使用安全加载函数
            af2_feature = safe_load_numpy(filepath)

            if af2_feature is not None:
                try:
                    # 处理特征维度
                    if len(af2_feature.shape) == 1:
                        af2_feature = af2_feature.reshape(-1, 1)
                    elif len(af2_feature.shape) > 2:
                        af2_feature = af2_feature.reshape(af2_feature.shape[0], -1)

                    af2_dict[seqid] = af2_feature
                    found_count += 1

                    # 显示前几个成功加载的信息
                    if found_count <= 3:
                        print(f"  成功加载 {seqid}: 形状 {af2_feature.shape}, 文件 {os.path.basename(filepath)}")

                except Exception as e:
                    print(f"处理特征维度时出错 {seqid}: {e}")
                    error_count += 1
            else:
                error_count += 1
        else:
            not_found_count += 1

    # 打印最终统计信息
    print(f"\nAF2特征加载最终统计:")
    print(f"成功加载: {found_count}/{len(seq_list)} ({found_count / len(seq_list) * 100:.1f}%)")
    print(f"未找到文件: {not_found_count}")
    print(f"加载出错: {error_count}")

    # 保存处理后的AF2特征
    if af2_dict:
        output_file = os.path.join(feature_dir, f'{ligand}_AF2.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(af2_dict, f)
        print(f"\nAF2特征已保存到: {output_file}")

        # 验证保存的文件
        try:
            with open(output_file, 'rb') as f:
                loaded_dict = pickle.load(f)
            print(f"文件验证成功: 包含 {len(loaded_dict)} 个序列的特征")

            # 显示特征形状统计
            shapes = [v.shape for v in loaded_dict.values()]
            unique_shapes = list(set(shapes))
            print(f"特征形状统计: {len(unique_shapes)} 种不同形状")
            for shape in unique_shapes[:5]:  # 显示前5种形状
                count = shapes.count(shape)
                print(f"  形状 {shape}: {count} 个序列")

        except Exception as e:
            print(f"文件验证失败: {e}")
    else:
        print(f"\n警告: 没有成功处理任何AF2特征！")

    return af2_dict



def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
                'TRP': 'W', 'CYS': 'C',
                'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
                'LYS': 'K', 'ARG': 'R'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path, 'r')
    pdb_res = pd.DataFrame(columns=['ID', 'atom', 'res', 'res_id', 'xyz', 'B_factor'])
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19, 'CO': 59,
                            'V': 51,
                            'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3, 'NA': 23, 'HG': 200.6,
                            'MN': 55,
                            'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9, 'SE': 79, 'NI': 58.7}

    while True:
        line = pdb_file.readline()
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count += 1
            res_pdb_id = int(line[22:26])
            if res_pdb_id != before_res_pdb_id:
                res_count += 1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N', 'CA', 'C', 'O', 'H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5, 0.5, 0.5]
            tmps = pd.Series(
                {'ID': atom_count, 'atom': line[12:16].strip(), 'atom_type': atom_type, 'res': res,
                 'res_id': int(line[22:26]),
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                 'occupancy': float(line[54:60]),
                 'B_factor': float(line[60:66]), 'mass': Relative_atomic_mass[atom_type], 'is_sidechain': is_sidechain,
                 'charge': atom_fea[0], 'num_H': atom_fea[1], 'ring': atom_fea[2]})
            if len(res_id_list) == 0:
                res_id_list.append(int(line[22:26]))
            elif res_id_list[-1] != int(line[22:26]):
                res_id_list.append(int(line[22:26]))
            pdb_res = pdb_res.append(tmps, ignore_index=True)
        if line.startswith('TER'):
            break

    return pdb_res, res_id_list

def normalize_chain_id(chain_id):
    """
    标准化链ID，处理单字符和双字符链ID的映射
    例如：jj -> j, oo -> o, ee -> e, ii -> i, aa -> a, gg -> g
    """
    if len(chain_id) == 2 and chain_id[0] == chain_id[1]:
        return chain_id[0]  # jj -> j
    return chain_id

def cal_PDBDF(seqlist, PDB_chain_dir, PDB_DF_dir):
    if not os.path.exists(PDB_DF_dir):
        os.mkdir(PDB_DF_dir)

    for seq_id in tqdm(seqlist):
        # print(seq_id)
        file_path = PDB_chain_dir + '/{}.pdb'.format(seq_id)
        with open(file_path, 'r') as f:
            text = f.readlines()
        if len(text) == 1:
            print('ERROR: PDB {} is empty.'.format(seq_id))
        if not os.path.exists(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id)):
            try:
                pdb_DF, res_id_list = get_pdb_DF(file_path)
                with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'wb') as f:
                    pickle.dump({'pdb_DF': pdb_DF, 'res_id_list': res_id_list}, f)
            except KeyError:
                print('ERROR: UNK in ', seq_id)
                raise KeyError

    return

def cal_Psepos(seqlist, PDB_DF_dir, Dataset_dir, psepos, ligand, seqanno):
    seq_CA_pos = {}
    seq_centroid = {}
    seq_sidechain_centroid = {}

    for seq_id in tqdm(seqlist):

        with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pickle.load(f)
        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']

        res_CA_pos = []
        res_centroid = []
        res_sidechain_centroid = []
        res_types = []
        for res_id in res_id_list:
            res_type = pdb_res_i[pdb_res_i['res_id'] == res_id]['res'].values[0]
            res_types.append(res_type)

            res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
            xyz = np.array(res_atom_df['xyz'].tolist())
            masses = np.array(res_atom_df['mass'].tolist()).reshape(-1, 1)
            centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_atom_df = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['is_sidechain'] == 1)]

            try:
                CA = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['atom'] == 'CA')]['xyz'].values[0]
            except IndexError:
                print('IndexError: no CA in seq:{} res_id:{}'.format(seq_id, res_id))
                CA = centroid

            res_CA_pos.append(CA)
            res_centroid.append(centroid)

            if len(res_sidechain_atom_df) == 0:
                res_sidechain_centroid.append(centroid)
            else:
                xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
                masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
                sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
                res_sidechain_centroid.append(sidechain_centroid)

        if ''.join(res_types) != seqanno[seq_id]['seq']:
            print(seq_id)
            print(''.join(res_types))
            print(seqanno[seq_id]['seq'])
            return
        res_CA_pos = np.array(res_CA_pos)
        res_centroid = np.array(res_centroid)
        res_sidechain_centroid = np.array(res_sidechain_centroid)
        seq_CA_pos[seq_id] = res_CA_pos
        seq_centroid[seq_id] = res_centroid
        seq_sidechain_centroid[seq_id] = res_sidechain_centroid

    if psepos == 'CA':
        with open(Dataset_dir + '/' + ligand + '_psepos_' + psepos + '.pkl', 'wb') as f:
            pickle.dump(seq_CA_pos, f)
    elif psepos == 'C':
        with open(Dataset_dir + '/' + ligand + '_psepos_' + psepos + '.pkl', 'wb') as f:
            pickle.dump(seq_centroid, f)
    elif psepos == 'SC':
        with open(Dataset_dir + '/' + ligand + '_psepos_' + psepos + '.pkl', 'wb') as f:
            pickle.dump(seq_sidechain_centroid, f)

    return


def cal_PSSM(ligand, seq_list, pssm_dir, feature_dir):
    nor_pssm_dict = {}
    for seqid in seq_list:
        file = seqid + '.pssm'
        with open(pssm_dir + '/' + file, 'r') as fin:
            fin_data = fin.readlines()
            pssm_begin_line = 3
            pssm_end_line = 0
            for i in range(1, len(fin_data)):
                if fin_data[i] == '\n':
                    pssm_end_line = i
                    break
            feature = np.zeros([(pssm_end_line - pssm_begin_line), 20])
            axis_x = 0
            for i in range(pssm_begin_line, pssm_end_line):
                raw_pssm = fin_data[i].split()[2:22]
                axis_y = 0
                for j in raw_pssm:
                    feature[axis_x][axis_y] = (1 / (1 + math.exp(-float(j))))
                    axis_y += 1
                axis_x += 1
            nor_pssm_dict[file.split('.')[0]] = feature
    with open(feature_dir + '/{}_PSSM.pkl'.format(ligand), 'wb') as f:
        pickle.dump(nor_pssm_dict, f)
    return


def cal_HMM(ligand, seq_list, hmm_dir, feature_dir):
    hmm_dict = {}
    for seqid in seq_list:
        file = seqid + '.hhm'
        with open(hmm_dir + '/' + file, 'r') as fin:
            fin_data = fin.readlines()
            hhm_begin_line = 0
            hhm_end_line = 0
            for i in range(len(fin_data)):
                if '#' in fin_data[i]:
                    hhm_begin_line = i + 5
                elif '//' in fin_data[i]:
                    hhm_end_line = i
            feature = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])
            axis_x = 0
            for i in range(hhm_begin_line, hhm_end_line, 3):
                line1 = fin_data[i].split()[2:-1]
                line2 = fin_data[i + 1].split()
                axis_y = 0
                for j in line1:
                    if j == '*':
                        feature[axis_x][axis_y] = 9999 / 10000.0
                    else:
                        feature[axis_x][axis_y] = float(j) / 10000.0
                    axis_y += 1
                for j in line2:
                    if j == '*':
                        feature[axis_x][axis_y] = 9999 / 10000.0
                    else:
                        feature[axis_x][axis_y] = float(j) / 10000.0
                    axis_y += 1
                axis_x += 1
            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
            hmm_dict[file.split('.')[0]] = feature
    with open(feature_dir + '/{}_HMM.pkl'.format(ligand), 'wb') as f:
        pickle.dump(hmm_dict, f)
    return


def cal_DSSP(ligand, seq_list, dssp_dir, feature_dir):
    maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
              'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319}
    map_ss_8 = {' ': [1, 0, 0, 0, 0, 0, 0, 0], 'S': [0, 1, 0, 0, 0, 0, 0, 0], 'T': [0, 0, 1, 0, 0, 0, 0, 0],
                'H': [0, 0, 0, 1, 0, 0, 0, 0],
                'G': [0, 0, 0, 0, 1, 0, 0, 0], 'I': [0, 0, 0, 0, 0, 1, 0, 0], 'E': [0, 0, 0, 0, 0, 0, 1, 0],
                'B': [0, 0, 0, 0, 0, 0, 0, 1]}
    dssp_dict = {}
    for seqid in seq_list:
        file = seqid + '.dssp'
        with open(dssp_dir + '/' + file, 'r') as fin:
            fin_data = fin.readlines()
        seq_feature = {}
        for i in range(25, len(fin_data)):
            line = fin_data[i]
            if line[13] not in maxASA.keys() or line[9] == ' ':
                continue
            res_id = float(line[5:10])
            feature = np.zeros([14])
            feature[:8] = map_ss_8[line[16]]
            feature[8] = min(float(line[35:38]) / maxASA[line[13]], 1)
            feature[9] = (float(line[85:91]) + 1) / 2
            feature[10] = min(1, float(line[91:97]) / 180)
            feature[11] = min(1, (float(line[97:103]) + 180) / 360)
            feature[12] = min(1, (float(line[103:109]) + 180) / 360)
            feature[13] = min(1, (float(line[109:115]) + 180) / 360)
            seq_feature[res_id] = feature.reshape((1, -1))
        dssp_dict[file.split('.')[0]] = seq_feature
    with open(feature_dir + '/{}_SS.pkl'.format(ligand), 'wb') as f:
        pickle.dump(dssp_dict, f)
    return


def PDBResidueFeature(seqlist, PDB_DF_dir, feature_dir, ligand, residue_feature_list, feature_combine, atomfea):
    # 加载所有特征字典
    for fea in residue_feature_list:
        if fea == 'AF2':
            fea_file = ligand + '_AF2.pkl'  # 实际保存的文件名
        else:
            fea_file = ligand + '_{}.pkl'.format(fea)

        with open(os.path.join(feature_dir, fea_file), 'rb') as f:
            locals()['residue_fea_dict_' + fea] = pickle.load(f)

        # with open(feature_dir + '/' + ligand + '_{}.pkl'.format(fea), 'rb') as f:
        #     locals()['residue_fea_dict_' + fea] = pickle.load(f)

    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85, 'H': 1.2, 'D': 1.2, 'SE': 1.9, 'P': 1.8, 'FE': 2.23,
                        'BR': 1.95,
                        'F': 1.47, 'CO': 2.23, 'V': 2.29, 'I': 1.98, 'CL': 1.75, 'CA': 2.81, 'B': 2.13, 'ZN': 2.29,
                        'MG': 1.73, 'NA': 2.27,
                        'HG': 1.7, 'MN': 2.24, 'K': 2.75, 'AC': 3.08, 'AL': 2.51, 'W': 2.39, 'NI': 2.22}
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    residue_feas_dict = {}
    for seq_id in tqdm(seqlist):
        with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pickle.load(f)

        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
        pdb_res_i = pdb_res_i[pdb_res_i['atom_type'] != 'H']

        # 原有的原子特征处理...
        mass = np.array(pdb_res_i['mass'].tolist()).reshape(-1, 1)
        mass = mass / 32
        B_factor = np.array(pdb_res_i['B_factor'].tolist()).reshape(-1, 1)
        if (max(B_factor) - min(B_factor)) == 0:
            B_factor = np.zeros(B_factor.shape) + 0.5
        else:
            B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))
        is_sidechain = np.array(pdb_res_i['is_sidechain'].tolist()).reshape(-1, 1)
        occupancy = np.array(pdb_res_i['occupancy'].tolist()).reshape(-1, 1)
        charge = np.array(pdb_res_i['charge'].tolist()).reshape(-1, 1)
        num_H = np.array(pdb_res_i['num_H'].tolist()).reshape(-1, 1)
        ring = np.array(pdb_res_i['ring'].tolist()).reshape(-1, 1)

        atom_type = pdb_res_i['atom_type'].tolist()
        atom_vander = np.zeros((len(atom_type), 1))
        for i, type in enumerate(atom_type):
            try:
                atom_vander[i] = atom_vander_dict[type]
            except:
                atom_vander[i] = atom_vander_dict['C']

        atom_feas = [mass, B_factor, is_sidechain, charge, num_H, ring, atom_vander]
        atom_feas = np.concatenate(atom_feas, axis=1)

        # 处理残基级别特征
        residue_feas = []
        for fea in residue_feature_list:
            fea_i = locals()['residue_fea_dict_' + fea][seq_id]
            if isinstance(fea_i, np.ndarray):
                residue_feas.append(fea_i)
            elif isinstance(fea_i, dict):
                fea_ii = []
                for res_id_i in res_id_list:
                    if res_id_i in fea_i.keys():
                        fea_ii.append(fea_i[res_id_i])
                    else:
                        fea_ii.append(np.zeros(list(fea_i.values())[0].shape))
                fea_ii = np.concatenate(fea_ii, axis=0)
                residue_feas.append(fea_ii)
        residue_feas = []
        for fea in residue_feature_list:
            fea_i = locals()['residue_fea_dict_' + fea][seq_id]

            if isinstance(fea_i, np.ndarray):
                print(f"[DEBUG] {seq_id} {fea} ndarray shape:", fea_i.shape)
                residue_feas.append(fea_i)

            elif isinstance(fea_i, dict):
                fea_ii = []
                for res_id_i in res_id_list:
                    if res_id_i in fea_i.keys():
                        fea_ii.append(fea_i[res_id_i])
                    else:
                        fea_ii.append(np.zeros(list(fea_i.values())[0].shape))
                try:
                    fea_ii = np.concatenate(fea_ii, axis=0)
                except Exception as e:
                    print(f"[ERROR] {seq_id} {fea} fea_ii contents shapes:",
                          [np.shape(x) for x in fea_ii])
                    raise e
                #print(f"[DEBUG] {seq_id} {fea} dict->array shape:", fea_ii.shape)
                residue_feas.append(fea_ii)

        print(f"[DEBUG] All features before concat for {seq_id}:",
              [r.shape for r in residue_feas])

        try:
            residue_feas = np.concatenate(residue_feas, axis=1)
        except ValueError:
            print('ERROR: Feature dimensions of {} are inconsistent!'.format(seq_id))
            raise ValueError
        except IndexError:
            print('ERROR: No residue features found for {}'.format(seq_id))
            continue

        if residue_feas.shape[0] != len(res_id_list):
            print(
                'ERROR: For {}, the number of residues with features is not consistent with the number of residues in the query!'.format(
                    seq_id))
            raise IndexError

        # 处理原子特征
        if atomfea:
            print("dealing atom")
            res_atom_feas = []
            atom_begin = 0
            for i, res_id in enumerate(res_id_list):
                res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
                atom_num = len(res_atom_df)
                res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
                res_atom_feas_i = np.average(res_atom_feas_i, axis=0).reshape(1, -1)
                res_atom_feas.append(res_atom_feas_i)
                atom_begin += atom_num
            res_atom_feas = np.concatenate(res_atom_feas, axis=0)
            residue_feas = np.concatenate((res_atom_feas, residue_feas), axis=1)

        residue_feas_dict[seq_id] = residue_feas

    with open(feature_dir + '/' + ligand + '_residue_feas_' + feature_combine + '.pkl', 'wb') as f:
        pickle.dump(residue_feas_dict, f)

    return


def tv_split(train_list, seed):
    random.seed(seed)
    random.shuffle(train_list)
    valid_list = train_list[:int(len(train_list) * 0.2)]
    train_list = train_list[int(len(train_list) * 0.2):]
    return train_list, valid_list


def StatisticsSampleNum(train_list, valid_list, test_list, seqanno):
    def sub(seqlist, seqanno):
        pos_num_all = 0
        res_num_all = 0
        for seqid in seqlist:
            anno = list(map(int, list(seqanno[seqid]['anno'])))
            pos_num = sum(anno)
            res_num = len(anno)
            pos_num_all += pos_num
            res_num_all += res_num
        neg_num_all = res_num_all - pos_num_all
        pnratio = pos_num_all / float(neg_num_all)
        return len(seqlist), res_num_all, pos_num_all, neg_num_all, pnratio

    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'NumSeq', 'NumRes', 'NumPos', 'NumNeg', 'PNratio']
    tb.float_format = '0.3'

    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list + valid_list, seqanno)
    tb.add_row(['train+valid', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list, seqanno)
    tb.add_row(['train', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(valid_list, seqanno)
    tb.add_row(['valid', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(test_list, seqanno)
    tb.add_row(['test', seq_num, res_num, pos_num, neg_num, pnratio])
    print(tb)
    return


if __name__ == '__main__':

    args = parse_args()
    checkargs(args)

    ligand = 'P' + args.ligand if args.ligand != 'HEME' else 'PHEM'
    psepos = args.psepos
    trans_anno = args.trans_anno
    dist = args.context_radius
    feature_list = []
    feature_combine = ''

    # 处理特征组合
    if 'PSSM' in args.features:
        feature_list.append('PSSM')
        feature_combine += 'P'
    if 'HMM' in args.features:
        feature_list.append('HMM')
        feature_combine += 'H'
    if 'SS' in args.features:
        feature_list.append('SS')
        feature_combine += 'S'
    if 'AF2' in args.features:  # 添加AF2特征处理
        feature_list.append('AF2')
        feature_combine += 'A2'
    if 'AF' in args.features:
        feature_list.append('AF')
        feature_combine += 'A'

    print( feature_combine)
    trainingset_dict = {'PDNA': 'DNA-new_Train.txt',
                        'PRNA': 'RNA-495_Train.txt',
                        'PMN': 'MN-440_Train.txt',
                        'PCA': 'CA-1022_Train.txt',
                        'PMG': 'MG-1194_Train.txt',
                        'PATP': 'ATP-388_Train.txt',
                        'PATP': 'ATP-update_Train.txt',
                        'PHEM': 'HEM-175_Train.txt'
                        }

    testset_dict = {'PDNA': 'DNA-129_Test.txt',
                    'PRNA': 'RNA-117_Test.txt',
                    'PMN': 'MN-144_Test.txt',
                    'PCA': 'CA-515_Test.txt',
                    'PMG': 'MG-651_Test.txt',
                    'PATP': 'ATP-41_Test.txt',
                    'PHEM': 'HEM-96_Test.txt'
                    }

    Dataset_dir = os.path.abspath('..') + '/Datasets' + '/' + ligand
    PDB_chain_dir = Dataset_dir + '/PDB'
    trainset_anno = Dataset_dir + '/{}'.format(trainingset_dict[ligand])
    testset_anno = Dataset_dir + '/{}'.format(testset_dict[ligand])

    seqanno = {}
    train_list = []
    test_list = []

    if ligand in ['PDNA', 'PRNA']:
        with open(trainset_anno, 'r') as f:
            train_text = f.readlines()
        if trans_anno:
            for i in range(0, len(train_text), 4):
                query_id = train_text[i].strip()[1:]
                if query_id[-1].islower():
                    query_id += query_id[-1]
                query_seq = train_text[i + 1].strip()
                query_anno = train_text[i + 2].strip()
                train_list.append(query_id)
                seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
        else:
            for i in range(0, len(train_text), 4):
                query_id = train_text[i].strip()[1:]
                if query_id[-1].islower():
                    query_id += query_id[-1]
                query_seq = train_text[i + 1].strip()
                query_anno = train_text[i + 3].strip()
                train_list.append(query_id)
                seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
        with open(testset_anno, 'r') as f:
            test_text = f.readlines()
        for i in range(0, len(test_text), 3):
            query_id = test_text[i].strip()[1:]
            if query_id[-1].islower():
                query_id += query_id[-1]
            query_seq = test_text[i + 1].strip()
            query_anno = test_text[i + 2].strip()
            test_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
    else:
        with open(trainset_anno, 'r') as f:
            train_text = f.readlines()
        for i in range(0, len(train_text), 3):
            query_id = train_text[i].strip()[1:]
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}

        with open(testset_anno, 'r') as f:
            test_text = f.readlines()
        for i in range(0, len(test_text), 3):
            query_id = test_text[i].strip()[1:]
            query_seq = test_text[i + 1].strip()
            query_anno = test_text[i + 2].strip()
            test_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}

    train_list, valid_list = tv_split(train_list, args.tvseed)
    StatisticsSampleNum(train_list, valid_list, test_list, seqanno)

    PDB_DF_dir = Dataset_dir + '/PDB_DF'
    seqlist = train_list + valid_list + test_list


    print('1.Extract the PDB information.')
    cal_PDBDF(seqlist, PDB_chain_dir, PDB_DF_dir)
    print('2.calculate the pseudo positions.')
    cal_Psepos(seqlist, PDB_DF_dir, Dataset_dir, psepos, ligand, seqanno)
    atomfea = False
    print('3.calculate the residue features.')
    if 'AF' in feature_list:
        atomfea = True
        feature_list.remove('AF')
    else:
        atomfea = False
    
    #计算各种特征（原子特征标志已经在上面设置）
    if 'PSSM' in feature_list:
        cal_PSSM(ligand, seqlist, Dataset_dir + '/feature/PSSM', Dataset_dir)
    if 'HMM' in feature_list:
        cal_HMM(ligand, seqlist, Dataset_dir + '/feature/HMM', Dataset_dir)
    if 'SS' in feature_list:
        cal_DSSP(ligand, seqlist, Dataset_dir + '/feature/SS', Dataset_dir)
    # 添加AF2特征计算
    if 'AF2' in feature_list:
        af2_dir = Dataset_dir + '/feature/AF2'
        print('Calculating AF2 features with complete fix...')
    
        # 使用自定义命名的AF2加载函数
        af2_result = cal_AF2_custom_naming_complete_fix(ligand, seqlist, af2_dir, Dataset_dir)
    
    PDBResidueFeature(seqlist, PDB_DF_dir, Dataset_dir, ligand, feature_list, feature_combine, atomfea)

    root_dir = Dataset_dir + '/' + ligand + '_{}_dist{}_{}'.format(psepos, dist, feature_combine)
    raw_dir = root_dir + '/raw'
    if os.path.exists(raw_dir):
        shutil.rmtree(root_dir)
    os.makedirs(raw_dir)
    print('4.Calculate the neighborhood of residues. Save to {}.'.format(root_dir))
    Create_NeighResidue3DPoint(psepos, dist, Dataset_dir, raw_dir, seqanno, feature_combine, train_list, valid_list,
                               test_list,ligand,batch_size=20)
    del psepos, dist, Dataset_dir, raw_dir, seqanno, feature_combine, train_list, valid_list, test_list
    gc.collect()
    print(f"系统总内存: {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")
    print(f"可用内存: {psutil.virtual_memory().available / 1024 ** 3:.2f} GB")
    _ = NeighResidue3DPoint(root=root_dir, dataset='train')
