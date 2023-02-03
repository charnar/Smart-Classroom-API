## Project Information

### Publication:

- "Camera-Based Log System for Human Physical Distance Tracking in Classroom", [View on IEEE](https://ieeexplore.ieee.org/document/9980055)

## Setup

### Requirements

- Python 3.10
- Pre-trained (YOLO v3) weights, [Link to download](https://mega.nz/folder/Wk9mhZwY#DHWTOaLFBaAlYl4u4PgC2g)
- Sample footage (to test out the demo endpoints)

#### 1. Create a `weights` folder in the root directory and store the downloaded pre-trained weights

#### 2. Create a `sample-footage` folder in the root directory and store your sample footage

#### 3. Add your MongoDB connection string in `app/configs.py`

```
DATABASE_URL = "YOUR_MONGODB_CONNECTION_STRING"
```

#### 4. Create a python environment and install required libraries from `requirements.txt` and run!

```
$ uvicorn app.main:app --port 80
```

#### 5. See your work on [localhost:80](http://localhost:80)!
