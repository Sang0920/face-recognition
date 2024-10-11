import requests
import base64
import os

def detect_faces_file(image_path):
    """Test face detection with file upload"""
    url = "http://localhost:8000/detect_faces"
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response.json()

def detect_faces_base64(image_path):
    """Test face detection with base64 image"""
    url = "http://localhost:8000/detect_faces"
    
    # Read image and convert to base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Send request
    response = requests.post(
        url,
        json={'image': image_data}
    )
    return response.json()

# Example usage
if __name__ == "__main__":
    # Ensure the directory for recognized faces exists
    recognized_faces_dir = "recognized_faces"
    if not os.path.exists(recognized_faces_dir):
        os.makedirs(recognized_faces_dir)

    # Test with file upload
    result = detect_faces_file("faces/sang/WIN_20240913_04_52_49_Pro.jpg")
    print("File upload result:")
    # print(result)
    print(f"Number of faces detected: {result['face_count']}")
    print(f"Datetime: {result['datetime']}")
    for face in result['faces']:
        print(f"Detected name: {face['name']}")
        print(f"Email: {face['email']}")
        print(f"Location: {face['location']}")
        # The face_image is a base64 string that can be saved or displayed
        # Save the face image to a file
        # face_image_path = os.path.join(recognized_faces_dir, f"face_{face['name']}.jpg")
        # with open(face_image_path, 'wb') as f:
        #     f.write(base64.b64decode(face['face_image']))

    # # Test with base64
    # result = detect_faces_base64("faces/sang/WIN_20240913_04_52_49_Pro.jpg")
    # print("\nBase64 result:")
    # print(result)
    # print(f"Number of faces detected: {result['face_count']}")
    # print(f"Datetime: {result['datetime']}")
    # for face in result['faces']:
    #     print(f"Detected name: {face['name']}")
    #     print(f"Email: {face['email']}")
    #     print(f"Location: {face['location']}")
    #     # The face_image is a base64 string that can be saved or displayed
    #     # Save the face image to a file
    #     face_image_path = os.path.join(recognized_faces_dir, f"face_{face['name']}_base64.jpg")
    #     with open(face_image_path, 'wb') as f:
    #         f.write(base64.b64decode(face['face_image']))