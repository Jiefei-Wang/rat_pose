import pandas as pd

def convert_sleap_to_dlc_format(video_name, sleap_df, deeplab_config):
    bodyparts = deeplab_config['bodyparts']
    # Create header rows before 
    # Example: https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1/CollectedData_Mackenzie.csv
    scorer_row = ['scorer'] + [deeplab_config['scorer']] * (len(bodyparts) * 2)
    bodyparts_row = ['bodyparts'] + [bp for bp in bodyparts for _ in range(2)]
    coords_row = ['coords'] + ['x', 'y'] * len(bodyparts)
    
    headers = [scorer_row, bodyparts_row, coords_row]

    csv_body = []
    for idx, row in sleap_df.iterrows():
        # Gets the frame number from the csv file 
        frame_num = int(row['frame_idx'])

        # Name of image that contains the frame number with extension (later will be used for frame extraction)
        img_name = f"img{frame_num:03d}.png"
        img_path = f"labeled-data/{video_name}/{img_name}" 
       
       # Row name will be the image path 
        data_row = [img_path]
     
        # Add x,y coordinates for each bodypart in order
        for bodypart in bodyparts:
            x_col = f"{bodypart}.x"
            y_col = f"{bodypart}.y"
            
            if x_col in row and y_col in row:
                x_val = row[x_col] if not pd.isna(row[x_col]) else None
                y_val = row[y_col] if not pd.isna(row[y_col]) else None
                data_row.extend([x_val, y_val])
            else:
                data_row.extend([None, None])
        csv_body.append(data_row)

    final_data = headers + csv_body
    return final_data

