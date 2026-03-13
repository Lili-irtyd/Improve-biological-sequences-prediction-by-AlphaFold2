import os
import pickle
import numpy as np
from pathlib import Path
import shutil

# 设置需要处理的根目录
#root_dir = "/media/gpux1/Pro1/output_full_sequences_train"
#root_dir = "/media/gpux1/Pro1/output_full_sequences_test"
root_dir = "/media/gpux1/Pro1/new_output_sequence_train"


# 遍历根目录下所有子文件夹
for subfolder in Path(root_dir).iterdir():
    if subfolder.is_dir():
        print(f"Processing folder: {subfolder.name}")

        for sub_subfolder in subfolder.iterdir():

            target_file = sub_subfolder / "result_model_1_pred_0.pkl"

            if target_file.exists():
                # 加载 pkl 文件内容
                print(f"Loading {target_file} ...")
                with open(target_file, "rb") as f:
                    data = pickle.load(f)

                # 保存为 .npy 文件，文件名是子文件夹名.npy
                npy_filename = f"{subfolder.name}.npy"
                npy_filepath = Path(root_dir) / npy_filename
                print(f"Saving to {npy_filepath} ...")
                np.save(npy_filepath, data)

                # 删除整个子文件夹
                print(f"Deleting folder {subfolder} ...")
                shutil.rmtree(subfolder)

            else:
                print(f"Warning: {target_file} not found in {subfolder}, skipping this folder.")
