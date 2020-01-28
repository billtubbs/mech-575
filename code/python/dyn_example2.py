# DMD Demo on video

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

if not os.path.exists(filepath):
    print("Downloading video from YouTube...")
    filepath = download_video(url)

print("Loading video file...")
cap = cv2.VideoCapture(filepath)

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

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
