# DMD Demo on video data

# Uses code adapted from this tutorial:
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

import os
import cv2
import numpy as np
from pytube import YouTube


def download_video(url):
    return YouTube(url).streams.first().download()


url = 'https://www.youtube.com/watch?v=qwz88S1P0os'
filepath = '/Users/billtubbs/HD Water in Swimming Pool Footage Loop 1080p.mp4'
data_dir = '../../data'
out_filename = 'video_data.npy'

if not os.path.exists(filepath):
    print("Downloading video from YouTube...")
    filepath = download_video(url)

print("Opening video file...")
cap = cv2.VideoCapture(filepath)

# Check if camera opened successfully
if (cap.isOpened() == False): 
    print("Error opening video stream or file.")

else:
    # Read until video is completed
    print("Reading and displaying video stream...")
    frames = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            frames.append(frame)
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
        # Break the loop
        else: 
            break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

n_frames = len(frames)
print(f"{n_frames} frames loaded.")
frame_shape = frames[0].shape
assert all([frame.shape == frame_shape for frame in frames])
frame_data = np.stack(frames)
del frames

filepath = os.path.join(data_dir, out_filename)
np.save(filepath, frame_data)
print(f"Frame data saved to '{filepath}'")
