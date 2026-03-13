import os
import numpy as np

def inspect_npy_files(search_directory):
    """
    递归扫描目录中的所有 .npy 文件，以找出哪些文件需要 allow_pickle=True。
    
    它会打印出问题文件的详细信息，以帮助诊断。
    """
    print(f"--- 开始扫描 .npy 文件于: {search_directory} ---")
    
    problem_files = []
    good_files_count = 0
    corrupted_files = []
    
    # 递归遍历所有子文件夹
    for root, dirs, files in os.walk(search_directory):
        for filename in files:
            if not filename.endswith('.npy'):
                continue
                
            filepath = os.path.join(root, filename)
            
            try:
                # 1. 尝试以安全模式（默认）加载
                np.load(filepath, allow_pickle=False)
                good_files_count += 1
                
            except ValueError as e:
                error_message = str(e)
                
                # 2. 检查是否是 "pickle" 相关的错误
                if "allow_pickle=False" in error_message or "Object arrays" in error_message:
                    
                    print(f"\n[!!!] 发现需要 Pickle 的文件:")
                    print(f"  文件路径: {filepath}")
                    
                    # 3. 既然知道它需要 pickle，我们现在用 allow_pickle=True 加载它来检查内容
                    try:
                        data = np.load(filepath, allow_pickle=True)
                        print(f"  -> 检查内容 (当 allow_pickle=True):")
                        print(f"     加载的数据类型: {type(data)}")
                        
                        if isinstance(data, np.ndarray):
                            print(f"     数组 'dtype': {data.dtype}")
                            print(f"     数组 'shape': {data.shape}")
                            
                            if data.dtype == 'object':
                                print(f"  -> 诊断: 失败原因是 dtype 为 'object'。")
                                print(f"     这意味着该 .npy 文件存储的不是纯数字，")
                                print(f"     而是通用的 Python 对象（例如 list, dict, 或混合类型的 list）。")
                                try:
                                    # 尝试显示第一个元素以获取更多线索
                                    print(f"     第一个元素的类型: {type(data.item(0))}")
                                except IndexError:
                                    print("     数组为空。")
                        else:
                            print(f"  -> 诊断: 该文件存储的甚至不是一个 'ndarray'，")
                            print(f"     而是一个被 pickle 的纯 Python 对象（例如一个 dict）。")
                            
                        problem_files.append(filepath)
                        
                    except Exception as e2:
                        print(f"  -> [错误] 尝试用 allow_pickle=True 加载，但仍然失败！")
                        print(f"     文件可能已损坏: {filepath}")
                        print(f"     错误信息: {e2}")
                        corrupted_files.append(filepath)
                        
                else:
                    # 其他类型的 ValueError
                    print(f"\n[!] 文件加载时遇到非 Pickle 的 ValueError: {filepath}")
                    print(f"    错误信息: {e}")
                    corrupted_files.append(filepath)
                    
            except Exception as e:
                # 其他错误 (例如: 权限不足, 文件不存在)
                print(f"\n[X] 无法加载文件 (未知错误): {filepath}")
                print(f"    错误信息: {e}")
                corrupted_files.append(filepath)

    # --- 4. 打印总结报告 ---
    print("\n--- 扫描完成 ---")
    print(f"  标准 .npy 文件 (加载成功): {good_files_count}")
    print(f"  需要 Pickle 的文件 (已识别): {len(problem_files)}")
    print(f"  损坏或无法加载的文件: {len(corrupted_files)}")
    
    if problem_files:
        print("\n--- 需要 Pickle 的文件列表 ---")
        for f in problem_files:
            print(f"  - {f}")
            
    if corrupted_files:
        print("\n--- 损坏/其他错误文件列表 ---")
        for f in corrupted_files:
            print(f"  - {f}")

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # --- 更改此路径 ---
    # 根据您之前的错误信息，您应该从这里开始扫描：
    SCAN_DIRECTORY = '/media/gpux1/Pro1/GraphBind/Datasets/PATP/feature/AF2/train'
    
    # 或者，您可以扫描整个 /AF2 目录
    # SCAN_DIRECTORY = '/media/gpux1/Pro1/GraphBind/Datasets/PATP/feature/AF2'
    # -----------------------------------------------------------------
    
    if not os.path.isdir(SCAN_DIRECTORY):
        print(f"错误：找不到目录 '{SCAN_DIRECTORY}'")
        print("请在脚本中编辑 'SCAN_DIRECTORY' 变量，使其指向您存储 .npy 文件的文件夹。")
    else:
        inspect_npy_files(SCAN_DIRECTORY)
