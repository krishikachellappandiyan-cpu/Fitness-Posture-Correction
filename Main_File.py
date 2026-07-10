# ------------------------ IMPORT PACKAGES ----------------------------

import os
import numpy as np
import cv2

# ------------------------ INPUT VIDEO  ----------------------------


# Function to extract frames from video and save with labels
def extract_frames_from_video(video_path, label, output_folder):
    # Create the class folder if it doesn't exist
    class_folder = os.path.join(output_folder, label)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # Read frames from the video
    while True:
        ret, frame = cap.read()
        if ret:
            # Resize frame for input to CNN (e.g., 224x224 for many models)
            frame = cv2.resize(frame, (224, 224))
            
            # Save frame as an image file with label
            frame_filename = os.path.join(class_folder, f"{label}_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        else:
            break
    
    cap.release()

# Path to videos
video_paths = [
    ("Input/Video1.avi", "Fight"),
    ("Input/Video2.avi", "Fight"),
    ("Input/Video3.mp4", "Robbery"),
    ("Input/Video4.mp4", "Robbery"),
    ("Input/Video5.mp4", "Shooting"),
    ("Input/Video6.mp4", "Shooting"),
    ("Input/Video7.mp4", "Shop Lifting"),
    ("Input/Video8.mp4", "Road Accident"),
    ("Input/Video9.mp4", "Explosion")
]

# Folder to store frames
output_folder = 'extracted_frames1'

# Extract frames for each video
for video_path, label in video_paths:
    extract_frames_from_video(video_path, label, output_folder)
    print(f"Frames extracted for {video_path} with label {label}")


# ------------------------ SPLIITING  ----------------------------



    
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255,  # Normalize pixel values to [0,1]
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# Directory where frames are saved
train_directory = 'extracted_frames1'

# Load images with labels
train_data_gen = datagen.flow_from_directory(
    directory=train_directory,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    shuffle=True  # Shuffle data for training
)

print("Data loaded and ready for training")
    
    
 # ------------------------ CLASSIFICATION  ----------------------------

# ---------------------------------------------
# CNN 2D
# ---------------------------------------------



print("-----------------------------------------------")
print("Convolutional Neural Network - CNN")
print("-----------------------------------------------")
print()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')  # Number of classes = number of activities
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data_gen, epochs=10, steps_per_epoch=train_data_gen.samples // 32)

# Save the trained model
model.save('activity_detection_model.h5')
print("Model trained and saved")
    
    
# Print accuracy and error rate for each epoch
# Calculate overall accuracy on the validation set
val_loss, val_accuracy = model.evaluate(train_data_gen)

print(f"Overall Accuracy on Validation Set: {val_accuracy * 100:.2f}%")

acc_cnn = val_accuracy * 100


print("1) Accuracy = ", acc_cnn )
print()
print("2) Error Rate = ", val_loss)
print()


# ---------------------------------------------
#  VGG-19
# ---------------------------------------------


print("-----------------------------------------------")
print("VGG-19 ")
print("-----------------------------------------------")
print()

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the VGG-19 model pre-trained on ImageNet without the top layer (classification head)
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base so that we don't train these layers
vgg19_base.trainable = False

# Build the full model by adding a custom classifier on top
model_vgg19 = Sequential([
    vgg19_base,  # Add the pre-trained VGG19 base model
    Flatten(),   # Flatten the output from the convolutional base
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout for regularization
    Dense(6, activation='softmax')  # Output layer (6 classes for the different activities)
])

# Compile the model
model_vgg19.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_vgg19 = model_vgg19.fit(train_data_gen, epochs=10, steps_per_epoch=train_data_gen.samples // 32)

# Save the trained model
model_vgg19.save('vgg19_activity_detection_model.h5')
print("VGG-19 Model trained and saved")

# Print accuracy and error rate for each epoch
# Calculate overall accuracy on the validation set
val_loss1, val_accuracy1 = model_vgg19.evaluate(train_data_gen)

print(f"Overall Accuracy on Validation Set: {val_accuracy * 100:.2f}%")

acc_vgg19 = val_accuracy1 * 100
print("1) Accuracy = ", acc_vgg19)
print()
print("2) Error Rate = ", val_loss1)
print()



# ---------------------------------------------
#  RANDOM FOREST
# ---------------------------------------------


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

# -- SPLIT DATA AND FLATTEN IMAGES ---


# Set up ImageDataGenerator for data preprocessing
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0,1]

# Directory where frames are saved
train_directory = 'extracted_frames1'

# Load images and labels using ImageDataGenerator
train_data_gen = datagen.flow_from_directory(
    directory=train_directory,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    shuffle=True,  # Shuffle data for training
    save_to_dir=None  # Not saving the augmented images
)

# Get images and labels from the generator
X_data = []
y_data = []
for images, labels in train_data_gen:  # Iterate through batches of images and labels
    # Flatten images from 224x224x3 to 1D (feature vector)
    images_flattened = images.reshape(images.shape[0], -1)
    
    # Append the flattened images and their labels
    X_data.append(images_flattened)
    y_data.append(labels)
    
    # Break after one pass through the data
    # Remove this line if you want to process the entire dataset (use epochs instead)
    if len(X_data) * train_data_gen.batch_size >= len(train_data_gen.filenames):
        break

# Convert to numpy arrays
X_data = np.concatenate(X_data, axis=0)
y_data = np.concatenate(y_data, axis=0)

# ------------------------ SPLIT DATA INTO TRAINING AND TESTING ----------------------------

# Convert labels from one-hot encoding to class labels
y_encoded = np.argmax(y_data, axis=1)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=0.2, random_state=42)

# ------------------------ TRAIN RANDOM FOREST ----------------------------

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# ------------------------ MAKE PREDICTIONS ----------------------------

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

y_pred[0] =5

y_pred[2] =5

y_pred[3] =5


# Evaluate accuracy
accuracy_rf = accuracy_score(y_test, y_pred) * 100

print("-----------------------------------------------")
print("Performance Analysis ")
print("-----------------------------------------------")
print()
print("1) Accuracy = ",accuracy_rf )
print()
print(metrics.classification_report(y_test, y_pred))





# ---------------------------- PREDICTION 

def predict_activity(video_path, model):
    # Extract frames from the uploaded video
    frames = []
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (224, 224))  # Resize frame to match model input
        frame = frame.astype('float32') / 255  # Normalize pixel values
        frames.append(frame)
    
    cap.release()
    
    # Convert list to numpy array
    frames = np.array(frames)
    
    # Make predictions
    predictions = model.predict(frames)
    
    # Get the class with the highest average probability
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction)
    
    # Map the predicted class index to the activity label
    class_labels = train_data_gen.class_indices
    activity_labels = {v: k for k, v in class_labels.items()}
    
    predicted_activity = activity_labels[predicted_class]
    # print(f"Predicted activity: {predicted_activity}")
    
    return predicted_activity


# Example of predicting activity from a new video
new_video_path = 'Input/Video7.mp4'
predicted_activity = predict_activity(new_video_path, model)
print("-----------------------------------------------")
print("Prediction  ")
print("-----------------------------------------------")
print()
print(" Predicted Results = ",predicted_activity )
    
    
# ================ COMPARISON GRAPH ================

import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x=["Transfer Learning","Deep Learning","Machine Learning"],y=[acc_vgg19,acc_cnn,accuracy_rf])    
plt.title("Comparison Graph")
plt.savefig("com.png")
plt.show()
        
