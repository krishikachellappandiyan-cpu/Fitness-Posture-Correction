import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import streamlit as st
import sqlite3
import base64


st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:32px;">{"Exercises for Diabetes Patients with Posture Correction"}</h1>', unsafe_allow_html=True)









def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('b1.jpg')




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

# Step 3: Display and predict video in Streamlit
def main():
    # Display title and description
    # st.title("Action Recognition with VGG-19")
    st.write("Upload a video file to predict the action and view the result.")

    # File uploader for video
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show the uploaded video in Streamlit
        # st.video(uploaded_file)
        
        
  
        import time
                     
      # Load the video
        video_file = "uploaded_video.mp4"  # Path to your video file
    
    # Open the video file using OpenCV
        cap = cv2.VideoCapture(video_file)
    
    # Get the total number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize frame index
        frame_index = 0
    
    # Create a placeholder for the video frame
        frame_placeholder = st.empty()
    
    # Loop through frames
        while frame_index < frame_count:
        # Set the video to the current frame
          cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
        # Read the frame
          ret, frame = cap.read()
    
          if ret:
            # Convert BGR to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
            # Display the frame
            frame_placeholder.image(frame, caption=f"Frame {frame_index}", use_column_width=True)
    
            # Update frame index
            frame_index += 1
            
            # Delay to control playback speed (adjust as needed)
            time.sleep(0.1)  # 10 frames per second
          else:
            break
    
    # Release the video capture object
        cap.release()              
                
        
        

        # Predict the class of the uploaded video
        predicted_class = predict_video_class(model, "uploaded_video.mp4")

        # Define class labels
        class_labels = {
            'Arm Raise Correct': 0,
            'Arm Raise Incorrect': 1,
            'Knee Extension Correct': 2,
            'Knee Extension Incorrect': 3,
            'Sit To Stand Correct': 4,
            'Sit To Stand Incorrect': 5
        }

        # Get the predicted class label
        predicted_class_label = list(class_labels.keys())[predicted_class[0]]

        
        fin_res="Identified Posture Exercise = " + str(predicted_class_label)



        # Display the prediction result
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:32px;">{fin_res}</h1>', unsafe_allow_html=True)

        # st.write(f"The predicted class for the video is: **{predicted_class_label}**")
        st.write("-----------------------------------------------------------------------")
        
        if predicted_class==1:
            
            # st.video("Visual Data/Arm Raise Correct/14.mp4")
            
             import time
                          
           # Load the video
             video_file = "Visual Data/Arm Raise Correct/14.mp4"  # Path to your video file
         
         # Open the video file using OpenCV
             cap = cv2.VideoCapture(video_file)
         
         # Get the total number of frames
             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
         
         # Initialize frame index
             frame_index = 0
         
         # Create a placeholder for the video frame
             frame_placeholder = st.empty()
         
         # Loop through frames
             while frame_index < frame_count:
             # Set the video to the current frame
               cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
         
             # Read the frame
               ret, frame = cap.read()
         
               if ret:
                 # Convert BGR to RGB format
                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
                 # Display the frame
                 frame_placeholder.image(frame, caption=f"Frame {frame_index}", use_column_width=True)
         
                 # Update frame index
                 frame_index += 1
                 
                 # Delay to control playback speed (adjust as needed)
                 time.sleep(0.1)  # 10 frames per second
               else:
                 break
         
         # Release the video capture object
             cap.release()                
                    
            
            
            
            
            
        
        elif predicted_class==3:
            
            # st.video("Visual Data/Knee Extension Correct/16.mp4")   
            
            
             import time
                          
           # Load the video
             video_file = "Visual Data/Knee Extension Correct/16.mp4"  # Path to your video file
         
         # Open the video file using OpenCV
             cap = cv2.VideoCapture(video_file)
         
         # Get the total number of frames
             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
         
         # Initialize frame index
             frame_index = 0
         
         # Create a placeholder for the video frame
             frame_placeholder = st.empty()
         
         # Loop through frames
             while frame_index < frame_count:
             # Set the video to the current frame
               cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
         
             # Read the frame
               ret, frame = cap.read()
         
               if ret:
                 # Convert BGR to RGB format
                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
                 # Display the frame
                 frame_placeholder.image(frame, caption=f"Frame {frame_index}", use_column_width=True)
         
                 # Update frame index
                 frame_index += 1
                 
                 # Delay to control playback speed (adjust as needed)
                 time.sleep(0.1)  # 10 frames per second
               else:
                 break
         
         # Release the video capture object
             cap.release()                
                        
            
            
            
            
            
            
            
            
            
            
            
            
        elif predicted_class==5:
             
             # st.video("Visual Data/Sit To Stand Correct/16.mp4")      
             
             
             import time
                          
           # Load the video
             video_file = "Visual Data/Sit To Stand Correct/16.mp4"  # Path to your video file
         
         # Open the video file using OpenCV
             cap = cv2.VideoCapture(video_file)
         
         # Get the total number of frames
             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
         
         # Initialize frame index
             frame_index = 0
         
         # Create a placeholder for the video frame
             frame_placeholder = st.empty()
         
         # Loop through frames
             while frame_index < frame_count:
             # Set the video to the current frame
               cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
         
             # Read the frame
               ret, frame = cap.read()
         
               if ret:
                 # Convert BGR to RGB format
                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
                 # Display the frame
                 frame_placeholder.image(frame, caption=f"Frame {frame_index}", use_column_width=True)
         
                 # Update frame index
                 frame_index += 1
                 
                 # Delay to control playback speed (adjust as needed)
                 time.sleep(0.1)  # 10 frames per second
               else:
                 break
         
         # Release the video capture object
             cap.release()                      
                     
             
            
        
        elif predicted_class==0 or predicted_class==2 or predicted_class==4:
           
             st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:32px;">{"Great job! You are performing the exercises correctly. Keep it up!"}</h1>', unsafe_allow_html=True)
           
        
        
        

# Run the app
if __name__ == "__main__":
    main()
