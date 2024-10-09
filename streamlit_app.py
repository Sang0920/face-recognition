import cv2
import streamlit as st
import numpy as np

# Face tracking and recognition in a video stream
# Install Streamlit and OpenCV
# pip install streamlit

import streamlit as st
import cv2
import numpy as np
import face_recognition
import time
import pickle

# Load face encodings and names from your existing embeddings.pkl
def load_embeddings():
    with open('embeddings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

# Recognize faces in the current frame
def recognize_faces(frame, known_face_encodings, known_face_names):
    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        
        # If a match was found, use the first match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    return face_locations, face_names

# Streamlit app for real-time face tracking and recognition
def main():
    st.title("Real-Time Face Tracking and Recognition")

    # Load known face encodings and names
    known_face_encodings, known_face_names = load_embeddings()

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    # Initialize some variables
    process_this_frame = True
    frame_id = 0
    fps = 0

    # Display the video feed with recognized faces
    stframe = st.empty()

    while video_capture.isOpened():
        start_time = time.time()
        ret, frame = video_capture.read()
        if not ret:
            st.warning("Failed to capture video feed.")
            break

        # Process every other frame to save CPU power
        if process_this_frame:
            face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Show FPS on the frame
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame using Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Increase frame ID for tracking purpose
        frame_id += 1

    # Release the webcam
    video_capture.release()

if __name__ == '__main__':
    main()
