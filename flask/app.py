'''
Faces folder structure:
faces
    - person1
        - image1.jpg
        - image2.jpg
        - ...
    - person2
        - image1.jpg
        - image2.jpg
        - ...
    - ...

Install OpenCV and face_recognition:
pip install opencv-python
pip install face_recognition

# Install Flask
pip install Flask
'''

# Web service that recongnizes face in the uploaded image.
# Upload an image and it will check if the image belongs to any of the persons in the faces folder.

# curl -X POST -F "file=@unkown" http://localhost:5000/recongnize

# Returns the name of the person if the face is recognized.
# Returns "Unknown" if the face is not recognized.

import cv2
import face_recognition
from flask import Flask, Response, render_template_string, request, jsonify, redirect
import os
import pickle
import base64
from io import BytesIO
from PIL import Image
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Save embeddings of all the faces in the faces folder
@app.route('/save_embeddings', methods=['GET'])
def save_embeddings():
    faces_folder = 'faces'
    known_face_encodings = []
    known_face_names = []

    valid_extensions = ('.jpg', '.jpeg', '.png')

    for person in os.listdir(faces_folder):
        person_folder = os.path.join(faces_folder, person)
        for image in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image)
            if image.lower().endswith(valid_extensions):
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person)
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")

    with open('embeddings.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return "Embeddings saved successfully."

# Detect faces in the uploaded image
@app.route('/recognize', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                return detect_faces_in_image(file)
        
        elif 'image_base64' in request.form:
            image_base64 = request.form['image_base64']
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            return detect_faces_in_image(image)
        
    # If no valid image file was uploaded, show the file upload form.
    return '''
    <!doctype html>
    <title>Is this a face?</title>
    <h1>Upload a face</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <h1>Or upload a base64 encoded image</h1>
    <form method=post>
      <textarea name=image_base64 rows=10 cols=30></textarea>
      <input type=submit value=Upload>
    </form>
    '''

def detect_faces_in_image(file_stream):
    # Load the uploaded image file
    if isinstance(file_stream, Image.Image):
        img = np.array(file_stream)
    else:
        img = face_recognition.load_image_file(file_stream)

    # Find all the faces and face encodings in the uploaded image
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # Load known face encodings
    with open('embeddings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

    face_names = []

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Return the results as a JSON response
    return jsonify(face_names)

# Real-time face tracking and recognition
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    video_capture = cv2.VideoCapture(0)
    known_face_encodings, known_face_names = load_embeddings()

    process_this_frame = True

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        if process_this_frame:
            face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)

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

        process_this_frame = not process_this_frame

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

def load_embeddings():
    with open('embeddings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

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

@app.route('/real_time_recognition')
def real_time_recognition():
    return render_template_string('''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Real-Time Face Tracking and Recognition</title>
      </head>
      <body>
        <h1>Real-Time Face Tracking and Recognition</h1>
        <img src="{{ url_for('video_feed') }}" width="100%">
      </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)