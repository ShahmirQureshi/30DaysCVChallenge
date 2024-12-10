import os
import pickle
import mediapipe as mp
import cv2

# Initialize mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './dataset'
data = []
labels = []

try:
    # Iterate through each directory in the DATA_DIR
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue  # Skip if not a directory

        # Iterate through each image in the directory
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img_full_path = os.path.join(dir_path, img_path)
            try:
                # Read and process the image
                img = cv2.imread(img_full_path)
                if img is None:
                    print(f"Warning: Could not read image {img_full_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                # Process hand landmarks if found
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            x_.append(landmark.x)
                            y_.append(landmark.y)

                        for landmark in hand_landmarks.landmark:
                            data_aux.append(landmark.x - min(x_))
                            data_aux.append(landmark.y - min(y_))
                            # print(data_aux)

                    data.append(data_aux)
                    labels.append(dir_)
            except Exception as e:
                print(f"Error processing image {img_full_path}: {e}")
except Exception as e:
    print(f"Error accessing data directory or processing files: {e}")

# Save data and labels to a pickle file
try:
    with open('dataset.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data successfully saved to data.pickle")
except Exception as e:
    print(f"Error saving data to pickle file: {e}")
