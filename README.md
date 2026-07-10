# Fitness Posture Correction System

A web-based fitness posture correction system developed using **Flask**, **Python**, and **Deep Learning (CNN & VGG19)** to help diabetes patients perform rehabilitation exercises with correct posture.

## Project Overview

This application analyzes exercise videos uploaded by users and predicts whether the posture is **Correct** or **Incorrect**. It provides an easy-to-use interface for posture assessment during rehabilitation exercises.

## Features

- User Login
- Upload exercise videos
- Deep learning-based posture classification
- Supports multiple rehabilitation exercises
- Displays prediction results
- Web interface built using Flask

## Technologies Used

- Python
- Flask
- TensorFlow / Keras
- CNN
- VGG19
- OpenCV
- HTML
- CSS
- SQLite

## Supported Exercises

- Arm Raise
- Knee Extension
- Sit To Stand
- Bhujangasana

## Project Structure

```
Fitness-Posture-Correction/
│
├── static/
│   ├── images/
│   └── styles/
│
├── templates/
│
├── Visual Data/
│
├── app.py
├── application.py
├── Prediction.py
├── FinModel.py
├── navigation.py
├── login.py
├── cnn.h5
├── cnn_vgg19.h5
├── dbs.db
└── README.md
```

## Installation

Clone the repository

```bash
git clone https://github.com/krishikachellappandiyan-cpu/Fitness-Posture-Correction.git
```

Move into the project

```bash
cd Fitness-Posture-Correction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

## Future Improvements

- Real-time webcam posture detection
- Pose estimation using MediaPipe
- Exercise repetition counting
- Performance dashboard
- User progress tracking
- Mobile application support

## Author

**Krishika Chellappandian**

Biomedical Engineering Graduate

GitHub:
https://github.com/krishikachellappandiyan-cpu
