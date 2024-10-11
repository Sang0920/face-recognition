# import streamlit as st
# import face_recognition
# import cv2
# import numpy as np
# import pickle
# from PIL import Image
# import time

# # Load face encodings and names from your existing embeddings.pkl
# def load_embeddings():
#     with open('embeddings.pkl', 'rb') as f:
#         known_face_encodings, known_face_names = pickle.load(f)
#     return known_face_encodings, known_face_names

# # Recognize faces in the current frame
# def recognize_faces(frame, known_face_encodings, known_face_names):
#     # Convert frame to RGB (face_recognition uses RGB)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Find all face locations and face encodings in the current frame
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     face_names = []
#     for face_encoding in face_encodings:
#         # See if the face is a match for known faces
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
#         name = "Unknown"
        
#         # If a match was found, use the first match
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]

#         face_names.append(name)

#     return face_locations, face_names

# def process_image(image, known_face_encodings, known_face_names):
#     # Convert PIL Image to numpy array
#     frame = np.array(image)
    
#     # Convert RGB to BGR (for OpenCV)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
#     # Get face locations and names
#     face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)
    
#     # Draw rectangles and labels around recognized faces
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
    
#     # Convert back to RGB for display
#     return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# def main():
#     st.title("Real-Time Face Tracking and Recognition (Simulated Video)")
    
#     # Load known face encodings and names
#     known_face_encodings, known_face_names = load_embeddings()

#     # Create a placeholder for video stream
#     stframe = st.empty()

#     # Simulate video by capturing images at intervals
#     camera_image = st.camera_input("Capture your image")
    
#     if camera_image is not None:
#         image = Image.open(camera_image)

#         # Process the image and display it
#         processed_image = process_image(image, known_face_encodings, known_face_names)
        
#         # Display the processed image
#         stframe.image(processed_image, channels="RGB")

#         # Simulate video feed (you could wrap this in a loop to refresh it periodically)

# if __name__ == '__main__':
#     main()

import streamlit as st
import face_recognition
import cv2
import numpy as np
import pickle
from PIL import Image

# Load face encodings and names from your existing embeddings.pkl
def load_embeddings():
    with open('embeddings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

# Recognize faces in the current frame
def recognize_faces(frame, known_face_encodings, known_face_names):
    # Convert frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    cropped_faces = []
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # See if the face is a match for known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        
        # If a match was found, use the first match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)
        
        # Crop the face from the frame and convert to PIL Image
        cropped_face = frame[top:bottom, left:right]
        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)  # Convert to RGB
        cropped_faces.append(Image.fromarray(cropped_face_rgb))  # Convert to PIL Image

    return face_locations, face_names, cropped_faces

def process_image(image, known_face_encodings, known_face_names):
    # Convert PIL Image to numpy array
    frame = np.array(image)
    
    # Convert RGB to BGR (for OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Get face locations, names, and cropped faces
    face_locations, face_names, cropped_faces = recognize_faces(frame, known_face_encodings, known_face_names)
    
    # Draw rectangles and labels around recognized faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Convert back to RGB for display
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face_names, cropped_faces

def main():
    st.title("Real-Time Face Tracking and Recognition (Simulated Video)")
    
    # Load known face encodings and names
    known_face_encodings, known_face_names = load_embeddings()

    # Create a placeholder for video stream
    stframe = st.empty()

    # Simulate video by capturing images at intervals
    camera_image = st.camera_input("Capture your image")
    
    if camera_image is not None:
        image = Image.open(camera_image)

        # Process the image and display it
        processed_image, face_names, cropped_faces = process_image(image, known_face_encodings, known_face_names)
        
        # Display the processed image
        stframe.image(processed_image, channels="RGB")

        # Display detected face names and their cropped images
        st.subheader("Detected Faces")
        for name, cropped_face in zip(face_names, cropped_faces):
            st.image(cropped_face, caption=name, use_column_width=True)

if __name__ == '__main__':
    main()
