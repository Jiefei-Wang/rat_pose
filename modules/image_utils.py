import matplotlib.pyplot as plt 
import cv2
import os 
#return image , gets video path, idx frame, and output directory 
def get_image_from_video(video_path, idx_frame):
    # Opens the video
    cap = cv2.VideoCapture(video_path)

    # checks if video is open
    if not cap.isOpened():
        print("ERROR: Could not open")
        return -1
    # set to specify the frame 
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
    ret, frame = cap.read()
    cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def save_image(file_path, frame):
    #  Make folder in the directory the video is in 
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite(file_path, frame)

    return file_path

def read_image(file_path, frame):
    # Read image
    frame = cv2.imread(file_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def show_image(frame):
    # Display
    plt.figure(figsize=(10,6)) 
    plt.imshow(frame)
    plt.axis('off')
    plt.show()
