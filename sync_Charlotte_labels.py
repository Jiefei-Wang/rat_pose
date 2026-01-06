import os
import shutil
import pandas as pd

new_label_path = "../rat_label"
project_path = "projects/rat_pose"
project_label_path = os.path.join(project_path, "labeled-data")

# copy and replace all label files from new_label_path to project_label_path
for session_folder in os.listdir(new_label_path):
    # skip files starting with '.'
    if session_folder.startswith('.'):
        continue
    if os.path.isdir(os.path.join(new_label_path, session_folder)):
        src_folder = os.path.join(new_label_path, session_folder)
        
        # Check for CSV files
        csv_files = [f for f in os.listdir(src_folder) if f.endswith('.csv')]
        
        # Stop if more than 1 csv file
        if len(csv_files) > 1:
            print(f"ERROR: Multiple CSV files found in {session_folder}: {csv_files}")
            print("Stopping execution.")
            break
        
        if len(csv_files) == 0:
            print(f"WARNING: No CSV file found in {session_folder}, skipping.")
            continue
        
        # Read the CSV file
        csv_path = os.path.join(src_folder, csv_files[0])
        df = pd.read_csv(csv_path, header=None)
        
        # Set column A from 4th row to end to "labeled-data"
        df.iloc[3:, 0] = "labeled-data"
        
        # Set first row from D column to end to "rats"
        df.iloc[0, 3:] = "rats"
        
        # Create destination folder
        dst_folder = os.path.join(project_label_path, session_folder)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        
        # Save as CollectedData_rats.csv
        output_csv = os.path.join(dst_folder, "CollectedData_rats.csv")
        df.to_csv(output_csv, index=False, header=False)
        
        print(f"Processed {session_folder}: {csv_files[0]} -> CollectedData_rats.csv")
