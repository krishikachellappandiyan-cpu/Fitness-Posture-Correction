from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os



# Load the saved VGG-19 model
model = load_model('cnn.h5')

# Step 1: Preprocess the video (extract frames)
def extract_frames_from_video(video_path, frame_size=(50, 50)):
    """
    Extract frames from a video and preprocess them.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)  # Resize frame
        frame = frame / 255.0  # Normalize frame
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Step 2: Predict the class for a single video
def predict_video_class(model, video_path, frame_size=(50, 50)):
    """
    Predict the class of the input video by predicting on each frame.
    """
    frames = extract_frames_from_video(video_path, frame_size)  # Extract frames from the video
    predictions = []
    
    for frame in frames:
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 50, 50, 3)
        prediction = model.predict(frame)  # Predict for the frame
        predictions.append(prediction)
    
    # Average the predictions to get the final class (could also do majority voting)
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction, axis=1)  # Get class with highest avg probability
    
    return predicted_class

# Step 2: Load data and preprocess all videos in the dataset
def load_data_from_folder(data_folder):
    """
    Load all videos from the folder and extract frames with labels based on subfolder names.
    """
    video_data = []
    video_labels = []
    class_labels = {cls: i for i, cls in enumerate(os.listdir(data_folder))}
    
    for class_name, class_index in class_labels.items():
        class_folder = os.path.join(data_folder, class_name)
        for video_file in os.listdir(class_folder):
            if video_file.endswith('.mp4') or video_file.endswith('.avi'):
                video_path = os.path.join(class_folder, video_file)
                frames = extract_frames_from_video(video_path)
                # Add frames and corresponding label
                for _ in range(frames.shape[0]):
                    video_data.append(frames[0])  # Take the first frame of the video
                    video_labels.append(class_index)
    
    video_data = np.array(video_data)
    video_labels = np.array(video_labels)
    return video_data, video_labels, class_labels


class_labels = {
    'Arm Raise Correct': 0,
    'Arm Raise Incorrect': 1,
    'Knee Extension Correct': 2,
    'Knee Extension Incorrect': 3,
    'Sit To Stand Correct': 4,
    'Sit To Stand Incorrect': 5
}



# Step 3: Predict the class for a specific video
video_path = 'Visual Data/Arm Raise Incorrect/01.mp4'  # Path to the video you want to predict
predicted_class = predict_video_class(model, video_path)

# Assuming class_labels is available from the data loading process
predicted_class_label = list(class_labels.keys())[predicted_class[0]]  # Get the class name from predicted class index
print(f"The predicted class for the video is: {predicted_class_label}")
