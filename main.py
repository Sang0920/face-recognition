# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import face_recognition
# import cv2
# import numpy as np
# import pickle
# import base64
# from PIL import Image
# import io

# app = FastAPI()

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def load_embeddings():
#     try:
#         with open('embeddings.pkl', 'rb') as f:
#             known_face_encodings, known_face_names = pickle.load(f)
#         return known_face_encodings, known_face_names
#     except Exception as e:
#         print(f"Error loading embeddings: {str(e)}")
#         return [], []

# def process_frame(frame_data, known_face_encodings, known_face_names):
#     # Decode base64 image
#     img_bytes = base64.b64decode(frame_data.split(',')[1])
#     img_np = np.frombuffer(img_bytes, dtype=np.uint8)
#     frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
#     # Convert BGR to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Detect faces
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
#     face_names = []
#     for face_encoding in face_encodings:
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
#         name = "Unknown"
        
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
        
#         face_names.append(name)
    
#     # Draw rectangles and names
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
#         cv2.putText(frame, name, (left + 6, bottom - 6), 
#                     cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
#     # Convert processed frame back to base64
#     _, buffer = cv2.imencode('.jpg', frame)
#     processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
#     return {
#         'processed_frame': f"data:image/jpeg;base64,{processed_frame_base64}",
#         'face_names': face_names
#     }

# @app.on_event("startup")
# async def startup_event():
#     app.state.known_face_encodings, app.state.known_face_names = load_embeddings()

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             # Receive frame from client
#             frame_data = await websocket.receive_text()
            
#             # Process the frame
#             result = process_frame(
#                 frame_data, 
#                 app.state.known_face_encodings, 
#                 app.state.known_face_names
#             )
            
#             # Send processed frame back to client
#             await websocket.send_json(result)
#     except Exception as e:
#         print(f"WebSocket error: {str(e)}")
#     finally:
#         await websocket.close()

# # Serve the HTML page
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse

# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/")
# async def get_index():
#     return FileResponse("static/index.html")


from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import cv2
import numpy as np
import pickle
import base64
from PIL import Image
import io
import asyncio

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_embeddings():
    try:
        with open('embeddings.pkl', 'rb') as f:
            known_face_encodings, known_face_names, _ = pickle.load(f)
        return known_face_encodings, known_face_names
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return [], []

def process_frame(frame_data, known_face_encodings, known_face_names):
    # Decode base64 image
    img_bytes = base64.b64decode(frame_data.split(',')[1])
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        face_names.append(name)
    
    # Draw rectangles and names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    # Convert processed frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'processed_frame': f"data:image/jpeg;base64,{processed_frame_base64}",
        'face_names': face_names
    }

@app.on_event("startup")
async def startup_event():
    app.state.known_face_encodings, app.state.known_face_names = load_embeddings()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive frame from client
            frame_data = await websocket.receive_text()
            
            # Process the frame
            result = process_frame(
                frame_data, 
                app.state.known_face_encodings, 
                app.state.known_face_names
            )
            
            # Send processed frame back to client
            await websocket.send_json(result)
            
            # Wait for 5 seconds before processing the next frame
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# Serve the HTML page
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

