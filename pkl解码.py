# ####to get shape

# import pickle
# import numpy  # Make sure numpy is imported
# import os     # Import os to get file size

# file_path = '/media/gpux1/Pro1/GraphBind/Datasets/PATP/PATP_AF2.pkl'

# # --- Optional: Get file size (as requested before) ---
# try:
#     size_in_bytes = os.path.getsize(file_path)
#     print(f"--- File size: {size_in_bytes / (1024*1024):.2f} MB ---")
# except FileNotFoundError:
#     print(f"Error: File not found at {file_path}")
#     exit()
# # --------------------------------------------------

# print("\n--- Loading .pkl file content ---")
# # Load .pkl file
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Output all keys and check data types
# if isinstance(data, dict):  # Check if data is a dictionary
#     print("Keys in the .pkl file:")
#     for key in data.keys():
#         print(f"\nKey: {key}")
        
#         # Get the value
#         value = data[key]
        
#         # --- (NEW BLOCK) ---
#         # Check if it's a NumPy array to print its shape
#         if isinstance(value, numpy.ndarray):
#             print(f"Type: NumPy Array")
#             print(f"Shape: {value.shape}")  # This will output the xxx*xxx size
#             print(f"Data Type (dtype): {value.dtype}")
#         # -------------------
        
#         elif isinstance(value, list):
#             print(f"Type: List, Length: {len(value)}")
#             print(f"First 5 items: {value[:5]}")
            
#         elif isinstance(value, dict):
#             print(f"Type: Dictionary, Keys (first 5): {list(value.keys())[:5]}")
            
#         elif isinstance(value, str):
#             print(f"Type: String, Length: {len(value)}")
#             print(f"Content (first 50 chars): {value[:50]}")
            
#         else:
#             # Fallback for other types like int, float, etc.
#             print(f"Type: {type(value)}, Value: {value}")
# else:
#     print("The .pkl file does not contain a dictionary.")


import pickle
import numpy as np
import os

file_path = '/media/gpux1/Pro1/GraphBind/Datasets/PATP/AF2_test/4XJX_B/4XJX_B/result_model_1_pred_0.pkl'

# --- File size ---
try:
    size_in_bytes = os.path.getsize(file_path)
    print(f"--- File size: {size_in_bytes / (1024*1024):.2f} MB ---")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

print("\n--- Loading .pkl file content ---")
# Load .pkl file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Output all keys and check data types
if isinstance(data, dict):
    # ====== PRINT NUMBER OF KEYS ======
    print(f"\n***** TOTAL NUMBER OF KEYS: {len(data)} *****\n")
    # ==================================
    
    print("Keys in the .pkl file:")
    
    # Track statistics
    ndarray_count = 0
    list_count = 0
    dict_count = 0
    other_count = 0
    
    for key in data.keys():
        print(f"\nKey: {key}")
        
        # Get the value
        value = data[key]
        
        # Check if it's a NumPy array
        if isinstance(value, np.ndarray):
            print(f"  Type: NumPy Array")
            print(f"  Shape: {value.shape}")
            print(f"  Data Type (dtype): {value.dtype}")
            ndarray_count += 1
            
        elif isinstance(value, list):
            print(f"  Type: List, Length: {len(value)}")
            print(f"  First 5 items: {value[:5]}")
            list_count += 1
            
        elif isinstance(value, dict):
            print(f"  Type: Dictionary, Keys (first 5): {list(value.keys())[:5]}")
            print(f"  Number of sub-keys: {len(value)}")
            dict_count += 1
            
        elif isinstance(value, str):
            print(f"  Type: String, Length: {len(value)}")
            print(f"  Content (first 50 chars): {value[:50]}")
            other_count += 1
            
        else:
            print(f"  Type: {type(value)}, Value: {value}")
            other_count += 1
    
    # ====== SUMMARY STATISTICS ======
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"  Total Keys: {len(data)}")
    print(f"  NumPy Arrays: {ndarray_count}")
    print(f"  Lists: {list_count}")
    print(f"  Dictionaries: {dict_count}")
    print(f"  Other Types: {other_count}")
    print("="*50)
    
    # ====== SHOW ALL KEYS (optional, good for checking) ======
    print("\n--- All Keys in File ---")
    all_keys = sorted(data.keys())  # Sort for easier reading
    for i, key in enumerate(all_keys, 1):
        print(f"{i}. {key}")
    
else:
    print("The .pkl file does not contain a dictionary.")
    print(f"Type: {type(data)}")