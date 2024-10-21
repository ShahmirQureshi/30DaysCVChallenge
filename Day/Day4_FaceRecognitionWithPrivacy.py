import cv2
import mediapipe as mp
import face_recognition
import numpy as np

print(cv2.__version__)

# Initialize the face detection using Mediapipe
Width = 640
Height = 360
FPS = 30

# Attempt to open the webcam
try:
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use default webcam
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
    cam.set(cv2.CAP_PROP_FPS, FPS)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    if not cam.isOpened():
        raise ValueError("Error: Unable to open video file.")
except Exception as e:
    print(f"Error opening video: {e}")
    exit(1)

# Initialize MediaPipe face detection
findFace = mp.solutions.face_detection.FaceDetection()
drawFace = mp.solutions.drawing_utils

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Dummy known images and names (update with real paths and names)
known_images = [
    'path/to/known_image1.jpg',
    'path/to/known_image2.jpg',
    'path/to/known_image3.jpg',
    'path/to/known_image4.jpg',
    'path/to/known_image5.jpg'
]
known_names = ['Person 1', 'Person 2', 'Person 3', 'Person 4', 'Person 5']

# Try to load known images and create face encodings
for img_path, name in zip(known_images, known_names):
    try:
        image = face_recognition.load_image_file(img_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    except IndexError:
        print(f"Error: No face found in image {img_path}. Skipping this image.")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

while True:
    try:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Convert the frame from BGR to RGB for MediaPipe and face_recognition
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use MediaPipe for face detection
        results = findFace.process(frame_rgb)

        if results.detections:
            face_locations = []
            for detection in results.detections:
                ih, iw, _ = frame.shape
                bBoxCords = detection.location_data.relative_bounding_box
                xmin, ymin, w, h = int(bBoxCords.xmin * iw), int(bBoxCords.ymin * ih), \
                                   int(bBoxCords.width * iw), int(bBoxCords.height * ih)
                topLeft = (xmin, ymin)
                bottomRight = (xmin + w, ymin + h)

                # Add face location in top-left and bottom-right corner format for face recognition
                face_locations.append((ymin, xmin + w, ymin + h, xmin))

                # Draw the face bounding box
                cv2.rectangle(frame, topLeft, bottomRight, (255, 0, 0), 2)

            # Recognize faces using face_recognition
            if face_locations:
                face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown Person"

                    # Use the known face with the smallest distance if a match is found
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    # If the face is unknown, blur it
                    if name == "Unknown Person":
                        face_region = frame[top:bottom, left:right]
                        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)  # Apply Gaussian blur
                        frame[top:bottom, left:right] = blurred_face  # Replace the face with the blurred version

                    # Draw the label with the name above the face rectangle
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Show the processed frame
        cv2.imshow("My WEBcam", frame)
        cv2.moveWindow("My WEBcam", 0, 0)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error during frame processing: {e}")

# Release resources
try:
    cam.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error releasing resources: {e}")
