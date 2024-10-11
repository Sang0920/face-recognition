# FACE RECOGNITION API

1. Clone this code
   ```sh
   git clone https://github.com/Sang0920/face-recognition.git
   cd face-recognition
   ```

2. Create vitural enviroment (optional)
    `sudo apt-get install python3-venv`
    `python3 -m venv myenv`
    `source myenv/bin/activate`

3. Install Requirements
    `pip install -r requirements.txt`

4. Run Demo
    `uvicorn main:app --reload`

5. Run API
    `uvicorn face_api:app --reload`

6. Run Test
    `python test_API.py`