# %%
import cv2
import imutils
import numpy as np
import argparse
import csv

# %%
# initialize the HOG descriptor/person detector
hogcv = cv2.HOGDescriptor()
hogcv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
global coord, framecount
coord = []
framecount = 0
# %%
def detect(frame):
    global framecount
    peoplecount = 0
    framecount = framecount + 1
    # Detect where to ahd How to create a Bounding Box
    bounding_box_cordinates, weights =  hogcv.detectMultiScale(frame, winStride = (8, 8), padding = (8, 8), scale = 1.05)
    # Render the bounding Box
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        #Populate the list corrd with x and y coordinate values
        peoplecount= peoplecount+1
        coord.append({"Frame": framecount, "People": peoplecount, "X_coord": x, "Y_coord": y})

    # Displaying Frames in a Window
    cv2.imshow('output', frame)

    #Call csvoutput definition
    csvoutput(coord)

    return frame

# %%
# Read Frame Data for videowriter
def framedata(path):
    video = cv2.VideoCapture(path, apiPreference=cv2.CAP_MSMF)
    _, frame = video.read()
    frame = imutils.resize(frame , width=min(600,frame.shape[1]))
    #height and width
    (h, w) = frame.shape[:2]
    #frames per second
    fps = video.get(cv2.CAP_PROP_FPS)

    return h,w,fps

# %%
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

# %%
#Export x and y coordinate values of tracked people in a .csv file
def csvoutput(coord):
    with open("coordinates.csv", mode="w") as csvfile:
        fieldnames = ["Frame", "People", "X_coord", "Y_coord"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in coord:
            writer.writerow(row)

# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Creating args argument
    ap.add_argument("-v", "--video", required=True, help="path to images directory")
    args = vars(ap.parse_args()) 
    # Passing path value to videoinput 
    videoinput(args["video"])