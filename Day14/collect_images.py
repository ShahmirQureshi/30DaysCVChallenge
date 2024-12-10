import os
import cv2

# Directory to save the dataset
DATA_DIR = './dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of gesture classes and dataset size
number_of_classes = 5  # Adjust according to the number of gestures
dataset_size = 100

# Capture from webcam
cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    gesture_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    print(f'Collecting data for class {j}')
    
    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect dataset images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(gesture_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
