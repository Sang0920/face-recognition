from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Union
import face_recognition
import cv2
import numpy as np
import pickle
import base64
import os
import json
from datetime import datetime
from test_draco import checkin_realtime
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Base64Image(BaseModel):
    image: str

def load_embeddings():
    try:
        with open('embeddings.pkl', 'rb') as f:
            known_face_encodings, known_face_names, known_emails = pickle.load(f)
        return known_face_encodings, known_face_names, known_emails
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return [], [], []

def process_image_data(image_data: np.ndarray) -> dict:
    """Process image data and return face detection results"""
    try:
        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_details = []
        cropped_faces = []
        
        # Get face encodings from app state
        known_face_encodings = app.state.known_face_encodings
        known_face_names = app.state.known_face_names
        known_emails = app.state.known_emails
        
        # Process each detected face
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"
            email = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                email = known_emails[first_match_index]
            
            face_details.append({
                "name": name,
                "email": email,
                "location": (top, right, bottom, left)
            })
            
            # Crop face and convert to base64
            cropped_face = image_data[top:bottom, left:right]
            _, buffer = cv2.imencode('.jpg', cropped_face)
            cropped_face_base64 = base64.b64encode(buffer).decode('utf-8')
            cropped_faces.append(cropped_face_base64)
            
            # Draw rectangle and name on the original image
            cv2.rectangle(image_data, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(image_data, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(image_data, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', image_data)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "processed_image": processed_image_base64,
            "faces": face_details,
            "face_count": len(face_details),
            "datetime": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to image array"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
            
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
            
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    app.state.known_face_encodings, app.state.known_face_names, app.state.known_emails = load_embeddings()

@app.post("/capture")
async def capture_image(base64_data: Base64Image):
    """
    Capture image from webcam and detect faces.
    """
    try:
        # Decode base64 image
        image = decode_base64_image(base64_data.image)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
            
        # Process the image
        result = process_image_data(image)
        for face in result["faces"]:
            email = face["email"]
            checkin_realtime(email)

        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/home.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/detect_faces")
async def detect_faces(
    file: Optional[UploadFile] = File(None),
    base64_data: Optional[Base64Image] = None
):
    """
    Detect faces in an image. Accepts either a file upload or base64 encoded image.
    
    Returns:
    - processed_image: Base64 encoded image with face rectangles drawn
    - faces: List of detected faces with names, email, and (top, right, bottom, left) of the face location.
    - face_count: Number of faces detected
    - datetime: Datetime when it send to the server
    """
    try:
        if file and base64_data:
            raise HTTPException(
                status_code=400, 
                detail="Please provide either a file or base64 image, not both"
            )
            
        if file:
            # Read uploaded file
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif base64_data:
            # Decode base64 image
            image = decode_base64_image(base64_data.image)
            
        else:
            raise HTTPException(
                status_code=400,
                detail="No image provided. Please upload a file or provide a base64 encoded image"
            )
            
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
            
        # Process the image
        result = process_image_data(image)
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example usage with curl:
"""
# For file upload:
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/detect-faces

# For base64 image:
curl -X POST -H "Content-Type: application/json" \
    -d '{"image": "base64_encoded_image_string"}' \
    http://localhost:8000/detect-faces
"""

UPLOAD_FOLDER = 'faces'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

valid_extensions = ('.jpg', '.jpeg', '.png')

# Helper to ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Helper to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']

"""
Faces folder structure:
```
faces
    - username1
        - metadata.json
        - image1.jpg
        - image2.jpg
        - ...
    - username2
        - metadata.json
        - image1.jpg
        - image2.jpg
        - ...
    - ...
```
In the metadata.json we save the fullname and the email address. E.g.
```
{
    "fullname": "Do The Sang",
    "email": "sangdt@draco.biz"
}
```
"""
@app.post("/add_person")
async def add_person(
    fullname: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    file: UploadFile = File(...)
):
    if not fullname or not email or not username:
        raise HTTPException(status_code=400, detail="Fullname, email, and username are required")

    person_folder = os.path.join(UPLOAD_FOLDER, username)
    ensure_dir(person_folder)

    if not file or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only jpg, jpeg, png files are allowed.")

    file_ext = file.filename.rsplit('.', 1)[1].lower()
    file_path = os.path.join(person_folder, f"image.{file_ext}")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Save metadata.json
    metadata = {
        "fullname": fullname,
        "email": email
    }
    metadata_path = os.path.join(person_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return {"message": f"Person {fullname} added successfully with image saved as {file_ext}!"}

# Route to display a form for adding a new person (Optional, can be used for browser testing)
@app.get("/add_person_form", response_class=HTMLResponse)
async def add_person_form():
    return '''
        <html>
            <body>
                <h2>Add a New Person</h2>
                <form action="/add_person" enctype="multipart/form-data" method="post">
                    Fullname: <input type="text" name="fullname"><br><br>
                    Email: <input type="text" name="email"><br><br>
                    Username: <input type="text" name="username"><br><br>
                    Select image to upload: <input type="file" name="file"><br><br>
                    <input type="submit" value="Upload">
                </form>
            </body>
        </html>
    '''
            
# Route to update and save embeddings from images
@app.get("/save_embeddings")
async def save_embeddings():
    faces_folder = UPLOAD_FOLDER
    known_face_encodings = []
    known_face_names = []
    known_emails = []

    for person in os.listdir(faces_folder):
        person_folder = os.path.join(faces_folder, person)
        metadata_path = os.path.join(person_folder, "metadata.json")

        # Read metadata.json to get the email
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                email = metadata.get("email", "Unknown")
        else:
            email = "Unknown"

        for image in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image)
            if image.lower().endswith(valid_extensions):
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person)
                    known_emails.append(email)
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")

    # Save the embeddings to a pickle file
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names, known_emails), f)

    return {"message": "Embeddings updated and saved successfully."}
