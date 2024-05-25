from ultralytics import YOLO

# # Load a model
model = YOLO("best.pt")  # build a new model from scratch

# # Use the model
results = model.train(data="config.yaml", epochs=150)  # train the model