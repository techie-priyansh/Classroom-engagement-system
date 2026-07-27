from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

result = results[0]

print("=" * 50)
print("Detected Objects")
print("=" * 50)

for box in result.boxes:
    class_id= int(box.cls[0])
    class_name= result.names[class_id]
    confidence= float(box.conf[0])
    coordinates= box.xyxy[0].tolist()

    print(f"Object : {class_name}")
    print(f"confidence: {confidence:.2f}")
    print(f"Coordinates: {coordinates}")
    print("-"*40)
