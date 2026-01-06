import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
LABELED_DATA_DIR = r"F:\code\pose_track\projects\rat2\labeled-data"

# Videos that were originally 1920x1080 (need scaling to 1280x720)
VIDEOS_1920x1080 = [
    "RAT 11 FR1 10-02-25",
    "RAT 11 FR1 10-03-25",
    "RAT 2 FR1 10-02-25",
    "RAT 2 FR1 10-03-25",
    "RAT 4 FR1 10-02-25",
    "RAT 4 FR1 10-03-25",
    "RAT 6 FR1 10-02-25",
    "RAT 6 FR1 10-03-25",
    "RAT 8 FR1 10-02-25",
    "RAT 8 FR1 10-03-25",
    "2025-03-18 12-58-10",
    "2025-03-18 12-58-46",
    "2025-03-18 14-08-10"
]

# Videos that were already 1280x720 (no scaling needed)
VIDEOS_1280x720 = [
    "Camera4_stitched",
    "RAT 11 FR1",
    "ai1",
    "ai2",
    "ai3",
    "ai4",
    "ai5",
    "ai6",
    "ai7",
    "ai8"
]

SCALE_X = 1280 / 1920  # New width / Old width
SCALE_Y = 720 / 1080   # New height / Old height

def update_csv_coordinates(csv_path, scale_x, scale_y):
    """
    Update coordinates in a DeepLabCut CSV file.
    Scales x and y coordinates while preserving all other data.
    """
    print(f"Processing: {csv_path.name}")
    
    # Read the CSV without treating first rows as header
    df = pd.read_csv(csv_path, header=None)
    
    # Row 3 (index 2) contains 'x' and 'y' coordinate labels
    coord_row = df.iloc[2]
    
    # Iterate through all columns starting from column 3 (first data columns)
    for col_idx in range(3, len(df.columns)):
        coord_type = coord_row.iloc[col_idx]
        
        if coord_type == 'x':
            # Scale all x coordinates in this column (starting from row 4, index 3)
            df.iloc[3:, col_idx] = pd.to_numeric(df.iloc[3:, col_idx], errors='coerce') * scale_x
        elif coord_type == 'y':
            # Scale all y coordinates in this column
            df.iloc[3:, col_idx] = pd.to_numeric(df.iloc[3:, col_idx], errors='coerce') * scale_y
    
    # Save back to the same file without header or index
    df.to_csv(csv_path, header=False, index=False)
    print(f"  ✓ Updated coordinates (scale: x={scale_x:.4f}, y={scale_y:.4f})")

def process_all_labeled_data():
    """
    Process all CSV files in the labeled-data directory.
    Only scales coordinates for videos that were originally 1920x1080.
    """
    labeled_data_path = Path(LABELED_DATA_DIR)
    
    if not labeled_data_path.exists():
        print(f"Error: Directory not found: {LABELED_DATA_DIR}")
        return
    
    # Find all subdirectories
    subdirs = [d for d in labeled_data_path.iterdir() if d.is_dir() and d.name != '.git']
    
    print(f"Found {len(subdirs)} subdirectories in labeled-data")
    print("=" * 60)
    
    scaled_count = 0
    skipped_count = 0
    
    for subdir in subdirs:
        video_name = subdir.name
        
        # Look for CSV files in each subdirectory
        csv_files = list(subdir.glob("CollectedData_*.csv"))
        
        for csv_file in csv_files:
            if video_name in VIDEOS_1920x1080:
                # Scale coordinates for 1920x1080 videos
                print(f"\n{video_name} (1920x1080 -> 1280x720)")
                update_csv_coordinates(csv_file, SCALE_X, SCALE_Y)
                scaled_count += 1
            elif video_name in VIDEOS_1280x720:
                # Skip videos that were already 1280x720
                print(f"\n{video_name} (already 1280x720)")
                print(f"  ⊘ Skipped (no scaling needed)")
                skipped_count += 1
            else:
                # Unknown video - warn user
                print(f"\n{video_name} (UNKNOWN RESOLUTION)")
                print(f"  ⚠ Warning: Video not in config.yaml, skipping")
                skipped_count += 1
    
    print("=" * 60)
    print(f"Complete!")
    print(f"  Scaled: {scaled_count} CSV files")
    print(f"  Skipped: {skipped_count} CSV files")

process_all_labeled_data()
