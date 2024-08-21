import cv2
import numpy as np
import json

# Paths to the images
image_path1 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera1/frame_0000.jpg'
image_path2 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera2/frame_0000.jpg'

# Load the images
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# Load points from JSON file
with open('calibration_points.json', 'r') as f:
    data = json.load(f)
    points1 = np.array(data['points1'], dtype=np.float32)
    points2 = np.array(data['points2'], dtype=np.float32)

# Debug: Print points to verify
print("Points1:", points1)
print("Points2:", points2)

# Check that points have the correct shape
if points1.shape[0] < 6 or points2.shape[0] < 6:
    raise ValueError("Not enough points for calibration. Ensure at least 6 points.")

# Define the 3D coordinates of these points relative to the aviary
object_points = np.array([
    [1.22, 2.286, 0],           # Halfway between corner 2 and 3
    [0, 4.572, 0],              # Corner 3
    [0.61, 3.429, 0],           # In-between the halfway point and corner 3
    [1.22, 2.286, 0.61],        # Quarter height from the ground at the halfway point
    [0.61, 3.429, 0.61],        # Quarter height from the ground at the in-between point
    [0, 4.572, 0.61]            # Quarter height from the ground at corner 3
], dtype=np.float32)

# Debug: Print object points to verify
print("Object Points:", object_points)

# Prepare object points in the correct shape for cv2.calibrateCamera
object_points_list = [object_points for _ in range(1)]  # List containing object points for one image

# Initialize the camera matrix using initCameraMatrix2D
image_size = image1.shape[1::-1]

# Note: we only need to initialize once, so let's reuse the initialized matrix for both cameras
initial_camera_matrix = cv2.initCameraMatrix2D(object_points_list, [points1], image_size)

# Debug: Print initial camera matrix
print("Initial Camera Matrix:", initial_camera_matrix)

# Set initial guesses for distortion coefficients
# Use only two coefficients: k1, k2
dist_coeffs1 = np.zeros((2, 1))
dist_coeffs2 = np.zeros((2, 1))

# Camera calibration using the collected points
# Ensure we provide the initial camera matrix properly
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera([object_points], [points1], image_size, initial_camera_matrix, dist_coeffs1, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera([object_points], [points2], image_size, initial_camera_matrix, dist_coeffs2, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

print("Camera 1 Matrix:")
print(mtx1)
print("Camera 1 Distortion Coefficients:")
print(dist1)
print("Camera 1 Rotation Vectors:")
print(rvecs1)
print("Camera 1 Translation Vectors:")
print(tvecs1)

print("Camera 2 Matrix:")
print(mtx2)
print("Camera 2 Distortion Coefficients:")
print(dist2)
print("Camera 2 Rotation Vectors:")
print(rvecs2)
print("Camera 2 Translation Vectors:")
print(tvecs2)
