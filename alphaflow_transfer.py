#python alphaflow_transfer.py /home/gpux1/evoformer_representation/test_ATP /media/gpux1/Pro1/GraphBind/Datasets/PATP/feature/AF2_pkl/test
#python alphaflow_transfer.py /home/gpux1/evoformer_representation/train_ATP /media/gpux1/Pro1/GraphBind/Datasets/PATP/feature/AF2_pkl/train
import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm # For progress bar

def extract_single_representation(input_folder, output_folder):
    """
    Loads .pkl files from input_folder, extracts the value under 
    ['representations']['single'], and saves it as a .pkl file 
    in output_folder with just the 'single' representation as the value.
    """
    
    # --- 1. Create output folder if it doesn't exist ---
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"✅ Output folder created or confirmed: {output_folder}")
    except OSError as e:
        print(f"❌ Error creating output folder {output_folder}: {e}")
        return # Stop if we can't create the output folder

    # --- 2. List all files in the input folder ---
    try:
        all_files = os.listdir(input_folder)
        # Filter for .pkl files only
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        if not pkl_files:
            print(f"⚠️ No .pkl files found in {input_folder}")
            return
        print(f"Found {len(pkl_files)} .pkl files to process.")
    except FileNotFoundError:
        print(f"❌ Error: Input folder not found: {input_folder}")
        return
    except Exception as e:
        print(f"❌ Error listing files in {input_folder}: {e}")
        return

    # --- 3. Process each .pkl file ---
    processed_count = 0
    error_count = 0
    
    # Wrap the loop with tqdm for a progress bar
    for filename in tqdm(pkl_files, desc="Processing .pkl files"):
        input_filepath = os.path.join(input_folder, filename)
        
        try:
            # --- Load the .pkl file ---
            with open(input_filepath, 'rb') as f:
                data = pickle.load(f)

            # --- Extract the 'single' representation ---
            # Check if the expected keys exist
            if isinstance(data, dict) and 'representations' in data and \
               isinstance(data['representations'], dict) and 'single' in data['representations']:
                
                single_representation = data['representations']['single']

                # --- Construct output path ---
                base_name = os.path.splitext(filename)[0] # Get filename without .pkl
                output_filename = f"{base_name}.pkl"  # Keep .pkl extension
                output_filepath = os.path.join(output_folder, output_filename)

                # --- Save as .pkl file with 'single' key ---
                output_data = {'single': single_representation}
                with open(output_filepath, 'wb') as f:
                    pickle.dump(output_data, f)
                
                processed_count += 1
                
                # Optional: Show shape info for first few files
                if processed_count <= 3:
                    shape = single_representation.shape if hasattr(single_representation, 'shape') else 'unknown'
                    print(f"\n  Processed {filename}: shape {shape}")
                    
            else:
                print(f"\n⚠️ Skipping {filename}: File structure invalid (missing 'representations' or 'single' key).")
                error_count += 1

        except pickle.UnpicklingError:
            print(f"\n❌ Error loading {filename}: File might be corrupted or not a valid pickle file.")
            error_count += 1
        except KeyError as e:
             print(f"\n❌ Error processing {filename}: Missing expected key {e}.")
             error_count += 1
        except Exception as e:
            print(f"\n❌ An unexpected error occurred processing {filename}: {e}")
            error_count += 1
            
    # --- 4. Print Summary ---
    print("\n--- Processing Summary ---")
    print(f"Total .pkl files found: {len(pkl_files)}")
    print(f"Successfully processed and saved: {processed_count}")
    print(f"Files skipped or failed due to errors: {error_count}")
    print(f"Output files saved in: {output_folder}")
    print(f"Output format: .pkl files with structure {{'single': numpy_array}}")
    print("--------------------------")


def main():
    parser = argparse.ArgumentParser(description="Extract 'single' representations from .pkl files and save as .pkl.")
    parser.add_argument("input_folder", help="Path to the folder containing the original .pkl files.")
    parser.add_argument("output_folder", help="Path to the folder where .pkl files with 'single' key will be saved.")
    
    args = parser.parse_args()
    
    extract_single_representation(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()