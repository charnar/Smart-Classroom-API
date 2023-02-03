## Project Information

### Publication:

- "Camera-Based Log System for Human Physical Distance Tracking in Classroom", [View on IEEE](https://ieeexplore.ieee.org/document/9980055)

## Installation

### Requirements

- Docker Desktop
- Pre-trained (YOLO v3) weights, [Link to download]()

#### 1. Create a `weights` folder in the root directory and store the downloaded pre-trained weights

#### 2. Create a `sample-footage` folder in the root directory and store your sample footage

#### 3. Add your MongoDB connection string in `apps/configs.py`

```
DATABASE_URL = "YOUR_MONGODB_CONNECTION_STRING"
```
