import cv2
import time

print("="*50)
cap= cv2.VideoCapture(0)
#cap= cv2.VideoCapture('http://10.147.234.168.8080/video')

if not cap.isOpened():
    print("Unable to access webcam")
    exit()

print("Webcam opened")
print("Press q to stop")

cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FPS,30)

frame_count = 0
start_time = time.time()


while True:
    ret, frame= cap.read()
    if not ret:
        print("Failed to show frames")
        exit()
    frame_count += 1
    elapsed = time.time() - start_time
    
    if elapsed > 0:
        fps = frame_count / elapsed
    else:
        fps = 0
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,200,0),
        2
    )
    cv2.imshow("Camera testing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
print("Frame received:", ret)
print("Frame shape:", frame.shape if ret else None)
cap.release()
cv2.destroyAllWindows()

print(f"Frames captured: {frame_count}")
print(f"Elapsed time: {elapsed:.2f} seconds")
print(f"Average FPS: {frame_count / elapsed:.2f}")

print("Webcam closed")