# Car-Counter: Vehicle Detection and Counting System 🚗📊

**Car-Counter** is a real-time vehicle detection and counting system that utilizes YOLO (You Only Look Once) for object detection. This project is designed to detect and count vehicles in real-time from video feeds. It leverages pre-trained YOLO weights for efficient object detection and can process both static and live video sources.

## Project Overview

This repository contains all the necessary files for running the vehicle detection system, including:

- YOLO model weights for object detection
- Python scripts for vehicle counting and video processing
- Sample images and videos for testing the model
- Graphical representations of vehicle counts

## Project Structure

The repository is structured as follows:

```
├── Yolo_weights/            # Folder containing pre-trained YOLO weights for object detection
├── videos/                  # Sample video files for vehicle detection and counting
├── .gitignore               # Git ignore file
├── LICENSE                  # Project license
├── README.md                # Project documentation
├── app.py                   # Main Python script to run the car counting system
├── graphics.png             # Example graphic showing vehicle count visualization
├── mask.png                 # Example mask image used for detection
├── requirements.txt         # Python dependencies for the project
└── sort.py                  # Python script for sorting and counting vehicles
```

## Features

- **Vehicle Detection**: Detects vehicles using YOLO object detection.
- **Real-Time Counting**: Counts vehicles as they move through the frame.
- **Multiple Video Inputs**: Supports live video and pre-recorded videos for detection and counting.
- **Graphical Output**: Visual representation of vehicle count and detection on video frames.
- **Pre-trained Weights**: Includes YOLO model weights for quick testing.

## Installation

To get started with the Car-Counter system, clone the repository:

```bash
git clone https://github.com/devvrat-hans/Car-Counter.git
cd Car-Counter
```

Next, install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

You can run the car counting system using the main script `app.py`.

### Run the Application:

```bash
python app.py
```

This will start the vehicle detection and counting system. You can customize the input video by modifying the script to use your own video file or webcam feed.

## Files

- **Yolo_weights/**: Contains the YOLO model weights needed for vehicle detection.
- **videos/**: Contains sample videos for testing the vehicle detection system.
- **app.py**: The main Python script to run the vehicle detection and counting system.
- **graphics.png**: A sample graphic used for displaying vehicle count information.
- **mask.png**: A sample image used for detection masking.
- **requirements.txt**: Lists the required Python libraries for the project.
- **sort.py**: Python script used for sorting and counting vehicles.

## Requirements

To run the Car-Counter system, the following Python libraries are required:

- OpenCV
- NumPy
- TensorFlow or PyTorch (depending on your YOLO version)
- other dependencies listed in `requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize and expand on the project for more specific use cases. Happy coding! 🚗💨
