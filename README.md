# 🤟 Sign Language Translator

A real-time Sign Language Translator that uses Computer Vision and Deep Learning to recognize sign language gestures through a webcam and convert them into text. The project uses MediaPipe for hand landmark detection and a trained machine learning model for gesture classification.

## Features

- Real-time sign language recognition
- Hand tracking using MediaPipe
- Webcam-based gesture detection
- Converts recognized signs into text
- Simple and user-friendly interface

## Tech Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Protagonist-1/Sign-Language-Translator.git
cd Sign-Language-Translator
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Start the application using:

```bash
python app.py
```

If your main file has a different name, replace `app.py` with the appropriate filename.

## Workflow

```text
Webcam Input
      ↓
MediaPipe Hand Detection
      ↓
Landmark Extraction
      ↓
Trained ML Model
      ↓
Gesture Classification
      ↓
Text Output
```

## Author

**Sameer Shaikh**

GitHub: https://github.com/Protagonist-1
