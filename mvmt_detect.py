import cv2
import argparse
import csv
import subprocess
import time
import numpy as np


# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights" , "yolov3_testing.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Read Frame Data for videowriter and extract the first frame
def framedata(path):
    video = cv2.VideoCapture(path, apiPreference=cv2.CAP_MSMF)
    _,frame = video.read()
    
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
        fieldnames = ["Frame", "Confidence", "X_coord", "Y_coord"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in coord:
            writer.writerow(row)
    
# Read the source video
def videoinput(path):
    coord = []
    # Frame by Frame Input Reading
    cap = cv2.VideoCapture(path, apiPreference=cv2.CAP_MSMF)
    height,width,fps = framedata(path)
    # VideoWriter enables writing frames to output
    result = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    while True:
        _, frame = cap.read()
        frame_id += 1

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    coord.append({"Frame": frame_id, "Confidence": confidence, "X_coord": x+(w/2), "Y_coord": y+h})
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = "Person"
                confidence = confidences[i]
                color = (0,255,0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
                
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
        csvoutput(coord)

        #writing video with bounding box
        result.write(frame)
        cv2.imshow("Image", frame)
    
        if cv2.waitKey(10) & 0xFF == ord('d'):
            break
    
    # Removing capture variable from memory  
    result.release()     
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Creating args argument
    ap.add_argument("-v", "--video", required=True, help="path to images directory")
    args = vars(ap.parse_args()) 
    # Passing path value to videoinput 
    videoinput(args["video"])

subprocess.run(["python", "heatmap.py"])