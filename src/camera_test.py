import cv2

print("="*50)
cap= cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to access webcam")
    exit()

print("Webcam opened")
print("Press q to stop")

while True:
    ret, frame= cap.read()
    if not ret:
        print("Failed to show frames")
        exit()

    cv2.imshow("Camera testing", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Webcam closed")