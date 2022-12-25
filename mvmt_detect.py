# %%
import cv2
import imutils
import numpy as np
import argparse

# %%
# initialize the HOG descriptor/person detector
hogcv = cv2.HOGDescriptor()
hogcv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# %%
def detect(frame):
    # Detect where to ahd How to create a Bounding Box
    bounding_box_cordinates, weights =  hogcv.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    # Render the bounding Box
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Displaying Frames in a Window
    cv2.imshow('output', frame)

    return frame

# %%
# Read the source
def videoinput(path):
    # Frame by Frame Input Reading
    video = cv2.VideoCapture(path, apiPreference=cv2.CAP_MSMF)

    while True:
        # Reading Frames
        ret, frame =  video.read()
        # Resizing Frames for better performance
        frame = imutils.resize(frame , width=min(600,frame.shape[1]))
        # Callig detect function
        frame = detect(frame)
         # Closing the Window   
        if cv2.waitKey(5) & 0xFF == ord('d'):
            break

     # Removing capture variable from memory       
    video.release()
    cv2.destroyAllWindows()


# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Creating args argument
    ap.add_argument("-v", "--video", required=True, help="path to images directory")
    args = vars(ap.parse_args()) 
    # Passing path value to videoinput 
    videoinput(args["video"])


