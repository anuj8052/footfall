import cv2
from ultralytics import YOLO
import numpy as np
import cv2
import pyresearch
import math
from sort import *


def footfall(data_address, dtr_class, x1, y1, x2, y2):
    
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    names = model.names
    classNames = list(names.values())

    # Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    limitsDown =   [x1, y1, x2, y2]

    totalCountUp = []
    totalCountDown = []

    # Open the video file
    video_path = data_address
    cap = cv2.VideoCapture(video_path)

    dtr_class = dtr_class
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        img = frame
        if success:
            # Run YOLOv8 inference on the frame
            results = model.track(frame, persist=True, conf = 0.5, classes= dtr_class)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            detections = np.empty((0, 5))

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    if currentClass == classNames[dtr_class] and conf > 0.3:
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))

            resultsTracker = tracker.update(detections)

            cv2.line(annotated_frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
                    if totalCountDown.count(id) == 0:
                        totalCountDown.append(id)

            cv2.putText(annotated_frame,str(len(totalCountDown)),(800,500),cv2.FONT_HERSHEY_PLAIN,5,(0,0,256),7)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # video_address, detection_class, x1, y1, x2, y2 (line coordinates)
    footfall('data/mall.mp4', 0, 50, 400, 935, 400)