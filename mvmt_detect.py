import cv2
import imutils
import numpy as np
import argparse
import os
import csv

# initialize the HOG descriptor/person detector
hogcv = cv2.HOGDescriptor()
hogcv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Global variables for coordinate's list and framecounter
global coord, framecount
coord = []
framecount = 0

#Detects people, creates bounding box and logs coordinates into a list
def detect(frame):
    global framecount
    peoplecount = 0
    framecount = framecount + 1

    # Detect where to and How to create a Bounding Box
    bounding_box_cordinates, weights =  hogcv.detectMultiScale(frame, winStride = (8, 8), padding = (8, 8), scale = 1.05)
    # Render the bounding Box
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        #Populate the list coord with x and y coordinate values
        peoplecount= peoplecount+1
        coord.append({"Frame": framecount, "People": peoplecount, "X_coord": x+(w/2), "Y_coord": y+(h/2)})

    # Displaying Frames in a Window
    cv2.imshow('output', frame)

    #Call csvoutput definition
    csvoutput(coord)

    return frame

# Read Frame Data for videowriter and extract the first frame
def framedata(path):
    video = cv2.VideoCapture(path, apiPreference=cv2.CAP_MSMF)
    _, frame = video.read()
    frame = imutils.resize(frame , width=min(600,frame.shape[1]))
    #height and width
    (h, w) = frame.shape[:2]
    #frames per second
    fps = video.get(cv2.CAP_PROP_FPS)

    #Extract a single frame from video
    cv2.imwrite("thumbnail.png", frame)

    return h,w,fps

# Export x and y coordinate values of tracked people in a .csv file
def csvoutput(coord):
    with open("coordinates.csv", mode="w", newline ='') as csvfile:
        fieldnames = ["Frame", "People", "X_coord", "Y_coord"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in coord:
            writer.writerow(row)

# Read the source video
def videoinput(path):
    # Frame by Frame Input Reading
    video = cv2.VideoCapture(path, apiPreference=cv2.CAP_MSMF)
    h,w,fps = framedata(path)
    # VideoWriter enables writing frames to output
    result = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
  
    while True:
        # Reading Frames
        isTrue, frame =  video.read()
        if isTrue == True:
            # Resizing Frames for better performance
            frame = imutils.resize(frame , width=min(600,frame.shape[1]))
            # Calling detect function
            frame = detect(frame)
            #writing video with bounding box
            result.write(frame)

            cv2.imshow('output', frame)
            # Closing the Window   
            if cv2.waitKey(10) & 0xFF == ord('d'):
                break

     # Removing capture variable from memory  
    result.release()     
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Creating args argument
    ap.add_argument("-v", "--video", required=True, help="path to images directory")
    args = vars(ap.parse_args()) 
    # Passing path value to videoinput 
    videoinput(args["video"])