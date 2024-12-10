import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_path = './model.p'
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary (adjust as needed)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Initialize variables
stable_prediction = None
stable_count = 0
stability_threshold = 15
recognized_characters = []

# Create an initial board window for recognized characters
board = np.ones((200, 600, 3), dtype=np.uint8) * 255

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Make a prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict.get(int(prediction[0]), '?')

        # Stability check
        if predicted_character == stable_prediction:
            stable_count += 1
        else:
            stable_prediction = predicted_character
            stable_count = 1

        if stable_count >= stability_threshold:
            recognized_characters.append(stable_prediction)
            board = np.ones((200, 600, 3), dtype=np.uint8) * 255
            cv2.putText(board, ''.join(recognized_characters), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

        # Display result on the frame
        cv2.putText(frame, predicted_character, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow('frame', frame)
    cv2.imshow('board', board)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
