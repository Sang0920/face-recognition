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

import face_recognition
from flask import Flask, request, jsonify, redirect
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)