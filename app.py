from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

cap = cv2.VideoCapture("./videos/cars.mp4") # For video
mask = cv2.imread("./mask.png") # For image

model = YOLO("../Yolo_weights/yolov8n.pt")


# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [423, 297, 673, 297]

totalCount = []
try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        imageRegion = cv2.bitwise_and(img, mask)

        imgGraphics = cv2.imread("./graphics.png", cv2.IMREAD_UNCHANGED)
        cvzone.overlayPNG(img, imgGraphics, [0, 0])
            
        results = model(imageRegion, stream=True)

        detections = np.empty((0, 5))   
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # cvzone.cornerRect(img, (x1, y1, w, h))
                
                conf = math.ceil(box.conf[0] * 100)

                cls = int(box.cls[0])
                currentClass = model.names[cls]

                if currentClass == "car" and conf>30:

                    # cvzone.putTextRect(img, f'{currentClass} {conf}%', 
                    #              (max(0, x1), max(35, y1-10)),
                    #              scale=0.6, thickness=1, offset=3)
                    cvzone.cornerRect(img, (x1, y1, w, h), l = 9, rt = 5)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
        resultsTracker = tracker.update(detections)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 2)

        for results in resultsTracker:
            x1, y1, x2, y2, id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(id)
            cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), colorR=(0, 255, 0))
            cvzone.putTextRect(img, f'{int(id)}', 
                                 (max(0, x1), max(35, y1-10)),
                                 scale=2, thickness=1, offset=10)
            cx, cy = x1 + w//2, y1 + h//2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limits[2] > cx > limits[0] and limits[1]-15 < cy < limits[3]+15:
                if len(totalCount) <= 20:
                    if id not in totalCount:
                        totalCount.append(id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

                else:
                    if id not in totalCount[-10: -1]:
                        totalCount.append(id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

        # cvzone.putTextRect(img, f'Total Count: {len(totalCount)}', (50, 50), scale=2, thickness=2, offset=10)
        cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 255), 8, cv2.LINE_AA)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

finally:
    cap.release()
    cv2.destroyAllWindows()