# FACE RECOGNITION API

1. Clone this code
   ```sh
   git clone https://github.com/Sang0920/face-recognition.git
   cd face-recognition
   ```

2. Create vitural enviroment (optional)
    ```sh
    sudo apt-get install python3-venv
    python3 -m venv myenv
    source myenv/bin/activate
    ```


3. Install Requirements
    ```sh
    pip install -r requirements.txt
    ```

4. Run Demo
    ```sh
    uvicorn main:app --reload
    ```

5. Run API
    ```sh
    uvicorn face_api:app --reload
    ```

6. Run Test
    ```sh
    python test_API.py
    ```