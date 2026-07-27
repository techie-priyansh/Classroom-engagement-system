import cv2
from ultralytics import YOLO

model= YOLO("yolo11n.pt")


cap = cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    if not ret:
        break

    results= model(frame)
    annotated_frame= results[0].plot()
    cv2.imshow("YOLO11 Person Detected", annotated_frame)   
    key= cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam Closed")

