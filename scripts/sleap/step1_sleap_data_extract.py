import sleap
import json
from sleap.io.format.csv import CSVAdaptor
import os


labels = sleap.load_file("input/sleap_labels/camera_4_stitched_with_ai.slp")
skeleton = labels.skeletons[0]

print(f"This is labels {labels}")
print(f"This is the skeleton labels {skeleton}")



csv_file_list =[]

for i, video in enumerate(labels.videos):
    # extract to get name of video
    vid = os.path.splitext(os.path.basename(video.filename))[0]
    # creating name for csv file with video name
    output_filename = f"output/sleap_skeleton/{vid}.csv"
    sleap.io.format.csv.CSVAdaptor.write(
        filename= output_filename, #The name for the csv file created
        source_object=labels,
        video=video
    )
    
    # Prints where the csv files were exported to
    print(f"Exported {video.filename} to {output_filename}")
    csv_file_list.append(output_filename)
    
    


# Create dictionary for json format
skeleton_data ={
    'bodyparts': [node.name for node in skeleton.nodes],
    'skeleton': [[edge[0].name,edge[1].name]
                 for edge in skeleton.edges],
    'videos': [str(video.filename) for video in labels.videos]  
}


# saving skeleton as json file ( change json file name, only the part before .json )
with open ('output/sleap_skeleton/sleap_data.json', 'w') as f:
    json.dump(skeleton_data, f)


import pandas as pd 
for video_csv in csv_file_list:

    # loads csv file 
    df = pd.read_csv(video_csv)

    # Shows the data before removal
    # print(f"Before removal: {df}")

    # substrings of columns we want to remove
    substrings_remove = ["track","score"]

    # selects the columns that contains the substrings
    columns_list = [col for col in df.columns if any(s in col for s in substrings_remove)]

    # removes the columns 
    df.drop(columns=columns_list, inplace = True)

    # converts the data back to csv file (make sure to keep the same csv file name as it will affect conversion in other notebook)
    df.to_csv(video_csv, index= False)
    
    # Shows the data after removal
    # print(f"After removal \n{df}")