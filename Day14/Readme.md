
# Hand Gesture Recognition Project

This project implements a hand gesture recognition system using computer vision and machine learning. It captures images of hand signs, processes them, trains a classifier, and performs real-time gesture detection using a webcam.

## Project Structure
The project consists of the following scripts:

1. **Image Collection** (`collect_images.py`): 
   This script captures images from a webcam to create a dataset of hand gestures for different classes.
   
2. **Dataset Creation** (`create_dataset.py`): 
   This script processes the collected images using MediaPipe to extract hand landmarks, and saves the data for training.
   
3. **Train Classifier** (`train_classifier.py`): 
   This script trains a classifier on the processed dataset and saves the trained model.
   
4. **Run Inference** (`inference_classifier.py`): 
   This script uses the webcam for real-time gesture classification based on the trained model.

## Setup and Installation

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- OpenCV
- MediaPipe
- scikit-learn
- pickle

Install the necessary packages by running:
```bash
pip install opencv-python mediapipe scikit-learn
```

### Data Collection
1. Run `collect_images.py` to capture hand gesture images for each class. The images are saved in the `./dataset` folder.

   Modify the script to specify the number of gesture classes and the number of images per class:
   ```python
   number_of_classes = 5  # Number of gestures to collect
   dataset_size = 100     # Number of images per gesture
   ```

### Dataset Processing
2. Run `create_dataset.py` to extract hand landmarks from the collected images and save the processed data in a `.pickle` file for model training.

### Model Training
3. Use `train_classifier.py` to train a Random Forest classifier on the processed dataset. The trained model will be saved as `model.p`.

### Inference
4. Run `inference_classifier.py` to start real-time gesture recognition using your webcam. Make sure to update the `labels_dict` in the script to match the labels for the gestures you've collected:
   ```python
   labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
   ```
   Update this dictionary based on the signs you want to recognize.

## How to Use

1. **Collect Gesture Images**: 
   ```bash
   python collect_images.py
   ```
   Press "Q" to start collecting images for each class.

2. **Process the Dataset**:
   ```bash
   python create_dataset.py
   ```

3. **Train the Classifier**:
   ```bash
   python train_classifier.py
   ```

4. **Perform Real-Time Gesture Recognition**:
   ```bash
   python inference_classifier.py
   ```

## Customization

- **Changing Labels**: Update the `labels_dict` in `inference_classifier.py` to match the gesture labels you collected.
- **Adjusting Number of Classes**: Modify the `number_of_classes` in `collect_images.py` to define how many different hand gestures you want to detect.

## Example Workflow

1. **Collecting Data**: Collect images of hand signs for each class using your webcam.
   
2. **Processing Data**: Extract hand landmarks and save the processed data.
   
3. **Training the Classifier**: Train a classifier to recognize different gestures.

4. **Running Inference**: Use the webcam to detect and classify hand gestures in real time.

## Troubleshooting
- Ensure the webcam is accessible to OpenCV.
- Confirm that the dataset has been properly collected and processed before training the classifier.
- Make sure the labels in `labels_dict` match the gestures you're training the model to recognize.
