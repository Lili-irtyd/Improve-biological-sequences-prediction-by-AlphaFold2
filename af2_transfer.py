#python af2_transfer.py /media/gpux1/Pro1/GraphBind/Datasets/PHEM/AF2_test /media/gpux1/Pro1/GraphBind/Datasets/PHEM/feature/AF2/test
#python af2_transfer.py /media/gpux1/Pro1/GraphBind/Datasets/PHEM/AF2_train /media/gpux1/Pro1/GraphBind/Datasets/PHEM/feature/AF2/train
import os
import pickle
import numpy as np # Must be imported for np.save
import argparse
import glob
from tqdm import tqdm
import pathlib

def extract_and_rename_single_npy(input_base_dir, output_dir):
    """
    Searches for result*.pkl files, extracts the 'single' representation,
    and saves it to output_dir as a .npy file (containing only the array),
    renamed based on the parent directory.
    Handles both {'single':...} and {'representations': {'single':...}} structures.
    """

    # --- 1. Create output folder if it doesn't exist ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Output folder created or confirmed: {output_dir}")
    except OSError as e:
        print(f"❌ Error creating output folder {output_dir}: {e}")
        return

    # --- 2. Find all relevant .pkl files recursively ---
    search_pattern = os.path.join(input_base_dir, '**', 'result_model_*_pred_*.pkl')
    print(f"Searching for files matching: {search_pattern}")
    pkl_files = glob.glob(search_pattern, recursive=True)

    if not pkl_files:
        print(f"⚠️ No files matching 'result_model_*_pred_*.pkl' found recursively under {input_base_dir}")
        return
    print(f"Found {len(pkl_files)} potential .pkl files to process.")

    # --- 3. Process each found .pkl file ---
    processed_count = 0
    error_count = 0
    skipped_count = 0

    for input_filepath in tqdm(pkl_files, desc="Processing .pkl files"):
        try:
            # --- Extract Protein ID from path ---
            path_obj = pathlib.Path(input_filepath)
            if len(path_obj.parts) < 3:
                 skipped_count += 1
                 continue
            protein_id = path_obj.parts[-3]
            
            # --- Construct output path: CHANGE EXTENSION HERE ---
            output_filename = f"{protein_id}.npy"  # <--- MODIFICATION 1: Change to .npy
            output_filepath = os.path.join(output_dir, output_filename)
            
            # --- Load the .pkl file ---
            with open(input_filepath, 'rb') as f:
                data = pickle.load(f)

            # --- Extract the 'single' representation ---
            single_representation = None
            if isinstance(data, dict):
                if 'single' in data:
                    single_representation = data['single']
                elif 'representations' in data and isinstance(data['representations'], dict) and 'single' in data['representations']:
                     single_representation = data['representations']['single']
                else:
                    error_count += 1
                    continue
            else:
                 error_count += 1
                 continue

            # --- Save as .npy file (directly saving the array) ---
            if single_representation is not None and isinstance(single_representation, np.ndarray):
                
                # <--- MODIFICATION 2: Use np.save instead of pickle.dump
                np.save(output_filepath, single_representation)
                # ----------------------------------------------------
                
                processed_count += 1
                
                # Show shape info for first few files
                if processed_count <= 3:
                    shape = single_representation.shape if hasattr(single_representation, 'shape') else 'unknown'
                    print(f"  ✓ Saved {protein_id}.npy with shape {shape}")
            else:
                 # Ensure it's a NumPy array before saving
                 error_count += 1


        except pickle.UnpicklingError:
            print(f"❌ Error loading {input_filepath}: File might be corrupted.")
            error_count += 1
        except Exception as e:
            print(f"❌ An unexpected error occurred processing {input_filepath}: {e}")
            error_count += 1

    # --- 4. Print Summary ---
    print("\n--- Processing Summary ---")
    print(f"Successfully processed and saved: {processed_count}")
    print(f"Files failed due to errors: {error_count}")
    print(f"Output format: .npy files containing the raw 'single' numpy array.")
    print("--------------------------")

# Note: The main function needs to be updated to call the new function name if you changed it.
def main():
    parser = argparse.ArgumentParser(description="Extract 'single' representations from nested AlphaFold result .pkl files and save as .npy files.")
    parser.add_argument("input_base_dir", help="Base directory to search recursively for .pkl files (e.g., /path/to/AF2_test).")
    parser.add_argument("output_dir", help="Directory where the renamed .npy files will be saved.")

    args = parser.parse_args()

    # Call the modified function
    extract_and_rename_single_npy(args.input_base_dir, args.output_dir)

if __name__ == "__main__":
    main()