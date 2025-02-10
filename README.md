# Advertisement Display Based on Age and Gender

## Overview
This project leverages computer vision techniques to detect human faces, predict their age and gender, and display targeted advertisements accordingly. It utilizes OpenCV's deep learning module (DNN) for face detection, age prediction, and gender classification.

## Features
- **Face Detection** using OpenCV DNN models
- **Age and Gender Classification** with pre-trained Caffe models
- **Personalized Advertisement Display** based on the detected age group
- **Real-time Processing** from a video feed or an image input

## Technologies Used
- OpenCV (cv2)
- Deep Learning (DNN module in OpenCV)
- Pre-trained Caffe Models
- Python

## Installation
### Prerequisites
Ensure you have Python installed along with OpenCV and other required dependencies:
```bash
pip install opencv-python opencv-contrib-python argparse
```

## Usage
If no image path is provided, the webcam will be used.

### Advertisement Images
Each detected age group is mapped to a corresponding advertisement image:

| Age Group  | Advertisement Image |
|------------|--------------------|
| (0-2)     | advertisement_0_2.jpg |
| (4-6)     | advertisement_4_6.jpg |
| (8-12)    | advertisement_8_12.jpg |
| (15-20)   | advertisement_15_20.jpg |
| (25-32)   | advertisement_25_32.jpg |
| (38-43)   | advertisement_38_43.jpg |
| (48-53)   | advertisement_48_53.jpg |
| (60-100)  | advertisement_60_100.jpg |

Ensure these images are available in the working directory.

## Model Details
- **Face Detection Model**: `opencv_face_detector.pbtxt` & `opencv_face_detector_uint8.pb`
- **Age Prediction Model**: `age_deploy.prototxt` & `age_net.caffemodel`
- **Gender Prediction Model**: `gender_deploy.prototxt` & `gender_net.caffemodel`

## Expected Output
- Detect faces in an image or video feed.
- Predict the age and gender of detected individuals.
- Display the corresponding advertisement based on the detected age group.
## Acknowledgments
- OpenCV and its pre-trained models for face, age, and gender detection.

