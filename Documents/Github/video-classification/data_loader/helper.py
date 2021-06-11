
import cv2
import numpy as np

def videoFile_to_array(video_path):
    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened()== False): 
        return -1

    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            return -1
            
    return np.array(frames)