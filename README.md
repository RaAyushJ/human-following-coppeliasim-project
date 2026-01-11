PioneerP3DX Autonomous Target Tracker

This project implements an autonomous tracking and following system for a PioneerP3DX mobile robot within the CoppeliaSim simulation environment. The system utilizes a custom-trained YOLO model for real-time visual perception and the ZeroMQ (ZMQ) Remote API for high-frequency robotic control.
Project Overview

The robot is designed to identify a specific target (the "red blob") via its onboard vision sensor and maintain a constant following distance. The control loop bridges high-level computer vision (ONNX inference) with low-level actuation (motor velocity control).
Core Functionality

    Target Detection: Uses a YOLO-based neural network optimized for red blob localization.

    Visual Servoing: Translates the target's pixel coordinates into differential drive commands for the robot's motors.

    Remote Control: Employs the ZMQ Remote API for synchronous communication between the Python controller and the simulation engine.

Technical Stack
Component	Technology
Simulator	CoppeliaSim (V-REP)
API	ZeroMQ (ZMQ) Remote API
Inference Engine	ONNX Runtime
Vision Model	

YOLOv8 / custom architecture

Language	Python 3.x
Libraries	OpenCV, NumPy, coppeliasim-zmqremoteapi-client
Repository Structure

    proj.ttt: The primary CoppeliaSim scene containing the robot, vision sensor, and environment.

red_blob_yolo.onnx: The optimized inference model for real-time target detection.

    red_blob_yolo.pth: PyTorch weights used during the model development phase.

    red1.png: Reference image from the robot's onboard camera showing a target acquisition.

API & Integration

This project uses the modern ZMQ Remote API, which allows for a synchronous, high-performance link to the simulator. This is critical for ensuring that motor commands are issued based on the exact frame captured by the vision sensor.
Installation
Bash

pip install coppeliasim-zmqremoteapi-client onnxruntime opencv-python numpy

Integration Code

The following Python snippet demonstrates how to initialize the ZMQ client, load the handles for the PioneerP3DX, and process frames from the vision sensor.
Python

import cv2
import numpy as np
import onnxruntime as ort
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Initialize ZMQ Client
client = RemoteAPIClient()
sim = client.require('sim')

# Load handles (Ensure names match the Scene Hierarchy)
left_motor = sim.getObject('/PioneerP3DX/leftMotor')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')
vision_sensor = sim.getObject('/PioneerP3DX/visionSensor')

# Load ONNX model
session = ort.InferenceSession("red_blob_yolo.onnx")

def run_tracker():
    sim.startSimulation()
    while True:
        # 1. Capture Image
        img, res = sim.getVisionSensorImg(vision_sensor)
        frame = np.frombuffer(img, dtype=np.uint8).reshape([res[1], res[0], 3])
        frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0)

        # 2. Run YOLO Inference
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True)
        results = session.run(None, {"images": blob})

        # 3. Control Logic (Example)
        # Calculate error and set velocities
        sim.setJointTargetVelocity(left_motor, 2.0)
        sim.setJointTargetVelocity(right_motor, 2.0)

if __name__ == "__main__":
    run_tracker()

Troubleshooting: Handle Errors

A common error in this setup is sim.getObject: object does not exist. To resolve this:

    Hierarchy Check: Verify the names in the Scene Hierarchy match the strings in your script (e.g., /PioneerP3DX/leftMotor).

    Absolute Paths: Always use absolute paths starting with / to ensure the API can locate the child objects of the Pioneer model.

    Alias Verification: If the name looks correct but fails, check the "Object Alias" in the object properties window to ensure no hidden characters are present.
