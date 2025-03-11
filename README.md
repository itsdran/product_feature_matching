# Object Tracking and Recognition with LightGlue

## Overview
This project implements an object tracking and recognition system using **OpenCV**, **PyTorch**, and **LightGlue**. The system captures a reference frame, detects moving objects, tracks them until they stabilize, and then identifies the object in a final frame using **LightGlue** for feature matching.

## Features
- Captures a **reference frame** as a baseline.
- Detects **moving objects** using background subtraction and contour analysis.
- Tracks the object until it **stabilizes**.
- Uses **LightGlue** to recognize the object in a new frame.
- Displays the **detected object and matched object** side by side.

## Dependencies
Ensure you have the following installed before running the project:

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- LightGlue

### Install Dependencies
```bash
pip install opencv-python numpy torch
pip install git+https://github.com/cvg/LightGlue.git
```

## Usage
Run the script to start the object detection and tracking:
```bash
python object_tracking.py
```

### Steps:
1. **Capture Reference Frame**: Press **'c'** to capture a reference frame.
2. **Object Detection**: The system detects moving objects using background subtraction.
3. **Object Stabilization**: The object must remain stable for a certain number of frames.
4. **Feature Matching with LightGlue**: The detected object is matched in the final frame.
5. **Results Displayed**: The detected object and matched object are shown side by side.

## Credit
This project integrates **LightGlue** for feature matching. LightGlue is an advanced feature-matching framework developed by **cvg**.

- LightGlue Repository: [https://github.com/cvg/LightGlue](https://github.com/cvg/LightGlue)

## License
This project follows the **MIT License**. However, LightGlue follows its own licensing terms. Please check the LightGlue repository for details.

## Author
Your Name

