import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Load and preprocess video data (Extract frames)
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


from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Step 3: Create a CNN model
def build_cnn_model(input_shape=(50, 50, 3), num_classes=6):
    """
    Create a simple CNN model for video classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the CNN model
# Step 4: Train the CNN model
def train_model(train_data, train_labels, val_data, val_labels, num_epochs=30):
    """
    Train the CNN model and calculate precision, recall, F1-score on validation data.
    """
    model = build_cnn_model(input_shape=train_data.shape[1:], num_classes=len(np.unique(train_labels)))
    
    # One-hot encode the labels
    train_labels_one_hot = to_categorical(train_labels, num_classes=len(np.unique(train_labels)))
    val_labels_one_hot = to_categorical(val_labels, num_classes=len(np.unique(train_labels)))
    
    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True)
    datagen.fit(train_data)
    
    # Training the model
    history = model.fit(datagen.flow(train_data, train_labels_one_hot, batch_size=32),
                        validation_data=(val_data, val_labels_one_hot), epochs=num_epochs)
    
    # Print the training and validation accuracy and loss after each epoch
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {history.history['loss'][epoch]} - Training Accuracy: {history.history['accuracy'][epoch]}")
        print(f"Validation Loss: {history.history['val_loss'][epoch]} - Validation Accuracy: {history.history['val_accuracy'][epoch]}")
    
    # Calculate precision, recall, F1-score using sklearn's classification_report
    val_predictions = model.predict(val_data)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    val_true_classes = np.argmax(val_labels_one_hot, axis=1)
    
    print("Classification Report:")
    print(classification_report(val_true_classes, val_pred_classes, target_names=[str(i) for i in range(len(np.unique(train_labels)))]))
    
    # Print final validation loss and accuracy
    print("Final Validation Loss:", history.history['val_loss'][-1])
    print("Final Validation Accuracy:", history.history['val_accuracy'][-1])
    
    plot_accuracy_loss_graphs(history)
    
    # Save the model as cnn.h5
    model.save('cnn.h5')
    print("Model saved as cnn.h5")
    
    return model


# Function to plot accuracy and loss graphs
def plot_accuracy_loss_graphs(history):
    """
    Plot training and validation accuracy and loss graphs.
    """
    # Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def predict_video_class(model, video_path):
    """
    Predict the class of the input video by predicting on each frame.
    """
    frames = extract_frames_from_video(video_path)  # Extract frames from the video
    predictions = []
    
    for frame in frames:
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 224, 224, 3)
        prediction = model.predict(frame)  # Predict for the frame
        predictions.append(prediction)
    
    # Average the predictions to get the final class (could also do majority voting)
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction, axis=1)  # Get class with highest avg probability
    
    return predicted_class

# Example of how to use the functions
# Set the path to the dataset (adjust accordingly)
data_folder = 'Visual Data'

# Step 1: Load data and preprocess all videos
video_data, video_labels, class_labels = load_data_from_folder(data_folder)

# Step 2: Split data into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(video_data, video_labels, test_size=0.2, random_state=42)

# Step 3: Train the model
model = train_model(train_data, train_labels, val_data, val_labels)




############### VGG-19


from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D

# Step 3: Create a VGG-19 model
def build_vgg19_model(input_shape=(50, 50, 3), num_classes=6):
    """
    Create a VGG-19 model for video classification.
    """
    # Load the pre-trained VGG-19 model without the top classification layers
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the VGG-19 layers to prevent them from being trained
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Global average pooling to reduce the output size
        Dense(128, activation='relu'),  # Add a dense layer for learning
        Dense(num_classes, activation='softmax')  # Output layer with softmax activation for classification
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the VGG-19 model
def train_model(train_data, train_labels, val_data, val_labels, num_epochs=30):
    """
    Train the VGG-19 model and calculate precision, recall, F1-score on validation data.
    """
    model = build_vgg19_model(input_shape=train_data.shape[1:], num_classes=len(np.unique(train_labels)))
    
    # One-hot encode the labels
    train_labels_one_hot = to_categorical(train_labels, num_classes=len(np.unique(train_labels)))
    val_labels_one_hot = to_categorical(val_labels, num_classes=len(np.unique(train_labels)))
    
    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True)
    datagen.fit(train_data)
    
    # Training the model
    history = model.fit(datagen.flow(train_data, train_labels_one_hot, batch_size=32),
                        validation_data=(val_data, val_labels_one_hot), epochs=num_epochs)
    
    # Print the training and validation accuracy and loss after each epoch
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {history.history['loss'][epoch]} - Training Accuracy: {history.history['accuracy'][epoch]}")
        print(f"Validation Loss: {history.history['val_loss'][epoch]} - Validation Accuracy: {history.history['val_accuracy'][epoch]}")
    
    # Calculate precision, recall, F1-score using sklearn's classification_report
    val_predictions = model.predict(val_data)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    val_true_classes = np.argmax(val_labels_one_hot, axis=1)
    
    print("Classification Report:")
    print(classification_report(val_true_classes, val_pred_classes, target_names=[str(i) for i in range(len(np.unique(train_labels)))]))
    
    # Print final validation loss and accuracy
    print("Final Validation Loss:", history.history['val_loss'][-1])
    print("Final Validation Accuracy:", history.history['val_accuracy'][-1])
    
    plot_accuracy_loss_graphs(history)
    
    # Save the model as cnn.h5
    model.save('cnn_vgg19.h5')
    print("Model saved as cnn_vgg19.h5")
    
    return model




model1 = train_model(train_data, train_labels, val_data, val_labels)









# # ============ PREDICTION



# # # Predict on a new video
# video_path = 'Test Data/Arm Raise Correct/136.mp4'  # Path to the video you want to predict
# predicted_class = predict_video_class(model, video_path)

# # Print the predicted class label
# predicted_class_label = list(class_labels.keys())[predicted_class[0]]  # Get the class name from predicted class index
# print(f"The predicted class for the video is: {predicted_class_label}")




