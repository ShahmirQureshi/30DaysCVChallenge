# Real-Time Face Recognition with Privacy Feature

This project implements real-time face detection and recognition using `OpenCV`, `MediaPipe`, and the `face_recognition` library. It includes a privacy feature that blurs unrecognized faces, ensuring that only known individuals' faces are clearly visible.

## Table of Contents
- [Real-Time Face Recognition with Privacy Feature](#real-time-face-recognition-with-privacy-feature)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [File Structure](#file-structure)
  - [Usage](#usage)
  - [Configuration](#configuration)
    - [Important Customizations](#important-customizations)
  - [Workflow](#workflow)
  - [How It Works](#how-it-works)
  - [Contributing](#contributing)

## Features
- **Real-Time Face Detection**: Efficiently detects faces in real-time using MediaPipe.
- **Face Recognition**: Compares detected faces with a set of known faces using pre-trained face encodings.
- **Privacy-First**: Unrecognized faces are automatically blurred to enhance privacy.
- **Webcam Input**: Captures live video from the webcam or can be customized to use a video file.
- **Exit Option**: The program allows you to exit cleanly by pressing the `q` key.

## Setup

### Prerequisites
Before running the project, ensure you have the following libraries installed:

```bash
pip install opencv-python mediapipe face_recognition numpy
```

### File Structure
Ensure your project has the following structure:
```
.
├── known_faces/           # Directory for storing images of known individuals
├── input_video/           # Input video file (optional if using video instead of live webcam)
└── output_video/          # Directory where the processed video can be saved
```

## Usage

1. Place images of known individuals in the `known_faces/` directory.
2. Modify the following variables in the script:

   - **`known_images`**: This list contains the paths of the images for known individuals. You need to update it with the correct paths for your own images.
     ```python
     known_images = [
         'path/to/known_image1.jpg',
         'path/to/known_image2.jpg',
         'path/to/known_image3.jpg',
         'path/to/known_image4.jpg',
         'path/to/known_image5.jpg'
     ]
     ```

   - **`known_names`**: This list contains the names corresponding to each image. Make sure the order of names matches the order of images.
     ```python
     known_names = ['Person 1', 'Person 2', 'Person 3', 'Person 4', 'Person 5']
     ```

3. Run the script:
   ```bash
   python face_recognition_with_blur.py
   ```

4. Press the `q` key to stop the webcam and exit the program.

## Configuration

### Important Customizations
1. **Adding More Known Faces**:  
   To add more known faces, simply:
   - Add the image path to the `known_images` list.
   - Add the corresponding name to the `known_names` list.

   Ensure the number of images matches the number of names.

2. **Blurring Settings**:  
   The degree of blurring for unrecognized faces can be adjusted by modifying the `cv2.GaussianBlur` parameters in the code:
   ```python
   blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)  # Change (99, 99) for more or less blur
   ```

3. **Switching Between Webcam and Video Input**:  
   The code uses a live webcam by default. To switch to a video file, modify the code where the webcam is opened and set the `video_path` variable (you can replace the webcam with a video capture from a file).

## Workflow

1. **Load Known Faces**:  
   The program loads images of known individuals and generates face encodings for each one using the `face_recognition` library.

2. **Video Capture**:  
   The program captures live video from the webcam. Alternatively, you can modify the script to capture from a video file.

3. **Face Detection (MediaPipe)**:  
   The program detects faces in the captured video frames using the MediaPipe face detection module.

4. **Face Recognition (face_recognition)**:  
   For each detected face, the program extracts face encodings and compares them with the known face encodings to identify if the person is known.

5. **Blurring Unknown Faces**:  
   If a detected face doesn't match any known faces, the program applies a Gaussian blur to that region of the frame.

6. **Output and Display**:  
   The processed video frame, including recognized names and blurred unknown faces, is displayed in real-time.

7. **Exit Option**:  
   The user can exit the program by pressing the `q` key at any time.

## How It Works

- **Face Detection**: MediaPipe's face detection module detects faces in each frame of the video.
- **Face Recognition**: The `face_recognition` library compares detected faces with the pre-loaded encodings of known faces.
- **Blurring Unrecognized Faces**: A Gaussian blur is applied to faces that aren't recognized, ensuring privacy for unknown individuals.
- **Real-Time Processing**: Frames are processed and displayed in real-time, allowing for smooth webcam video output.

## Contributing

Contributions are welcome! If you find any bugs or want to add new features, feel free to open an issue or submit a pull request.
