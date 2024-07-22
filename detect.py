import ultralytics
from ultralytics import YOLO

# Load your model
MODEL = r'C:\Users\14168\1\Python\pythonProject\ultralytics\runs\segment\train2\weights\best.pt'
mymodel = YOLO(MODEL)

# Load your image
img = r'C:\Users\14168\1\Python\pythonProject\ultralytics\1E3DDC26B444CFC027D374516FB40826.jpg'

# Predict the result
results = mymodel.predict(img)

# Display results
results.show()