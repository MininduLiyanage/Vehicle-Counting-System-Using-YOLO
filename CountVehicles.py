from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *

cap = cv2.VideoCapture("E:/EEE/comvis/ComputerVision/ODCourse/cars3.mp4")  # For Video

mask = cv2.imread("E:/EEE/comvis/ComputerVision/ODCourse/mask-carcounter.png")
mask = cv2.resize(mask,(1280,720)) #same size as the video

model = YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
              "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
              "toilet", "tvmonitor", "laptop", "mouse", "remote",
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) # max_age- if object diappear howmany frames we wait until it appear

limits = [400, 297, 673, 297]  # crossing line coordinates
totalCount = []   #counter for vehicles

while True:

    success, img = cap.read()
    #img = cv2.resize(img2,(950,480))
    imgMask = cv2.bitwise_and(img, mask) # masking to focus only on required area
    results = model(imgMask, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3) #maually draw bounding box
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=1)  # display class name and confidence  on a rectangle using cvzone
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)  # easy way for bounding box of object detected by YOLO

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    # line to detect vehicle crossing
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)

        w, h = x2 - x1, y2 - y1
        # bounding box of object detected by tracker(check whether tracking ID same)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # Display vehicle ID
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),scale=1, thickness=2, offset=1)

        # mid point of bbox
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # crossing-line
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15: #limits to track vehicle crossing, change y coordinates as prefer
            # to remove repetitions
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) #change line color when vehicle crossing
        # display no of vehicles
        cv2.putText(img, str(len(totalCount)), (155, 130), cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    #cv2.imshow("mask", imgMask)

    cv2.waitKey(1)