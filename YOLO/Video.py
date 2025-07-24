from ultralytics import YOLO

model = YOLO("C:\\Users\\ethan\\OneDrive\\UVic things\\2025 Summer\\SENG474\\ASL-Fingerspelling\\YOLO\\ASL-fingerspelling\\asl-model\\weights\\epoch90.pt") # change this to your model path
results = model.predict(source=0, show=True)

for result in results:
    boxes = result.boxes
    classes = result.names