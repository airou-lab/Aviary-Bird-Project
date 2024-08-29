import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load CSV files containing the bounding box detections for each camera
csv_path1 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output1.csv'
csv_path2 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output2.csv'
data1 = pd.read_csv(csv_path1)  # Load detections for camera 1
data2 = pd.read_csv(csv_path2)  # Load detections for camera 2

# Camera calibration parameters (intrinsic and distortion coefficients)
mtx1 = np.array([[272.71933895, 0, 886.17820811],
                 [0, 569.84377895, 400.12351546],
                 [0, 0, 1]])  # Camera matrix for camera 1
dist1 = np.array([0.06475292, 0.00490271, 0.00131659, -0.06968295, -0.00127087])  # Distortion coefficients for camera 1
rvecs1 = np.array([0.57380764, 1.8596441, -1.95509504])  # Rotation vector for camera 1
tvecs1 = np.array([-0.88466712, 0.29868002, 4.37445632])  # Translation vector for camera 1

mtx2 = np.array([[272.71933895, 0, 886.17820811],
                 [0, 569.84377895, 400.12351546],
                 [0, 0, 1]])  # Camera matrix for camera 2 (same as camera 1 in this case)
dist2 = np.array([-0.00427968, -0.00140623, -0.01699009, -0.01083722, -0.00140955])  # Distortion coefficients for camera 2
rvecs2 = np.array([1.75025535, 0.09443925, -0.34156242])  # Rotation vector for camera 2
tvecs2 = np.array([-3.74331806, 2.15330245, -0.73257324])  # Translation vector for camera 2

# Convert rotation and translation vectors to projection matrices
def get_projection_matrix(mtx, rvecs, tvecs):
    R, _ = cv2.Rodrigues(rvecs)  # Convert rotation vector to rotation matrix
    return np.dot(mtx, np.hstack((R, tvecs.reshape(-1, 1))))  # Compute projection matrix

proj_matrix1 = get_projection_matrix(mtx1, rvecs1, tvecs1)  # Projection matrix for camera 1
proj_matrix2 = get_projection_matrix(mtx2, rvecs2, tvecs2)  # Projection matrix for camera 2

# Load the initial frames from both cameras
initial_frame1_path = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera1/frame_0001.jpg'
initial_frame2_path = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera2/frame_0001.jpg'
initial_frame1 = cv2.imread(initial_frame1_path)  # Load frame 1 from camera 1
initial_frame2 = cv2.imread(initial_frame2_path)  # Load frame 1 from camera 2
initial_frame1_copy = initial_frame1.copy()  # Create a copy for visualization
initial_frame2_copy = initial_frame2.copy()  # Create a copy for visualization

# Function to display bounding box detections on a frame
def display_detections(frame, detections, window_name):
    for det in detections:
        xmin, ymin, xmax, ymax = map(int, det)  # Extract bounding box coordinates
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw green bounding box
    cv2.imshow(window_name, frame)  # Display the frame with detections

# Get initial detections from the first frame for both cameras
initial_detections1 = data1[data1['Frame'] == 1][['xmin', 'ymin', 'xmax', 'ymax']].values  # Detections from camera 1
initial_detections2 = data2[data2['Frame'] == 1][['xmin', 'ymin', 'xmax', 'ymax']].values  # Detections from camera 2

display_detections(initial_frame1_copy, initial_detections1, 'Frame 1')  # Display detections for camera 1
display_detections(initial_frame2_copy, initial_detections2, 'Frame 2')  # Display detections for camera 2

# Lists to store selected detections
selected_dets1, selected_dets2 = [], []

# Function to allow the user to match bounding boxes in Frame 1 by clicking on them
def match_bbox1(event, x, y, flags, param):
    global selected_dets1, initial_detections1, initial_frame1_copy
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_dets1) < 3:  # Check if left mouse button was clicked
        for det in initial_detections1:
            xmin, ymin, xmax, ymax = map(int, det)
            if xmin <= x <= xmax and ymin <= y <= ymax:  # Check if click is within a bounding box
                selected_dets1.append(det)  # Add detection to the list
                # Set color based on the order of selection
                color = (255, 0, 0) if len(selected_dets1) == 1 else (0, 0, 255) if len(selected_dets1) == 2 else (0, 255, 255)
                cv2.rectangle(initial_frame1_copy, (xmin, ymin), (xmax, ymax), color, 2)  # Draw the selected box
                cv2.imshow('Frame 1', initial_frame1_copy)  # Update the display
                break

# Function to allow the user to match bounding boxes in Frame 2 by clicking on them
def match_bbox2(event, x, y, flags, param):
    global selected_dets2, initial_detections2, initial_frame2_copy
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_dets2) < 3:  # Check if left mouse button was clicked
        for det in initial_detections2:
            xmin, ymin, xmax, ymax = map(int, det)
            if xmin <= x <= xmax and ymin <= y <= ymax:  # Check if click is within a bounding box
                selected_dets2.append(det)  # Add detection to the list
                # Set color based on the order of selection
                color = (255, 0, 0) if len(selected_dets2) == 1 else (0, 0, 255) if len(selected_dets2) == 2 else (0, 255, 255)
                cv2.rectangle(initial_frame2_copy, (xmin, ymin), (xmax, ymax), color, 2)  # Draw the selected box
                cv2.imshow('Frame 2', initial_frame2_copy)  # Update the display
                break

# Set up mouse callbacks to enable selection of bounding boxes
cv2.setMouseCallback('Frame 1', match_bbox1)
cv2.setMouseCallback('Frame 2', match_bbox2)

print("Please select three matching bounding boxes in Frame 1 and Frame 2 by clicking on them.")
cv2.waitKey(0)  # Wait for the user to select the bounding boxes
cv2.destroyAllWindows()  # Close the display windows

# Function to triangulate the 3D point corresponding to a pair of matching detections
def triangulate_point(det1, det2, proj_matrix1, proj_matrix2):
    center1 = np.array([(det1[0] + det1[2]) / 2, (det1[1] + det1[3]) / 2])  # Calculate the center of the bounding box in camera 1
    center2 = np.array([(det2[0] + det2[2]) / 2, (det2[1] + det2[3]) / 2])  # Calculate the center of the bounding box in camera 2
    point4D = cv2.triangulatePoints(proj_matrix1, proj_matrix2, center1.reshape(-1, 1), center2.reshape(-1, 1))  # Triangulate to get 4D point
    point3D = point4D[:3] / point4D[3]  # Normalize to get the 3D point
    return point3D

# Initialize Kalman Filters for tracking three birds, incorporating velocity in the state model
class KalmanFilter3DWithVelocity:
    def __init__(self, dt, std_acc, std_meas):
        self.x = np.zeros((9, 1))  # State vector [x, y, z, vx, vy, vz, ax, ay, az]
        # State transition matrix, accounting for position, velocity, and acceleration
        self.F = np.array([[1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
                           [0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
                           [0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2],
                           [0, 0, 0, 1, 0, 0, dt, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # Measurement matrix, relating the state to the measurements
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        # Process noise covariance matrix, accounting for uncertainty in the model
        self.Q = std_acc ** 2 * np.eye(9)
        # Measurement noise covariance matrix, accounting for uncertainty in the measurements
        self.R = std_meas ** 2 * np.eye(3)
        # Estimate error covariance matrix, initialized to identity
        self.P = np.eye(9)

    # Predict the next state based on the model
    def predict(self):
        self.x = np.dot(self.F, self.x)  # Predict state vector
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Predict estimate error covariance
        return self.x[:3]  # Return predicted 3D position

    # Update the state with a new measurement
    def update(self, z):
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)  # Update state estimate
        I = np.eye(self.H.shape[1])
        self.P = (I - np.dot(K, self.H)).dot(self.P)  # Update estimate error covariance

# Initialize Kalman filters for the three birds
kalman_filters = []
colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255)]  # Colors for birds: Red, Blue, Yellow
for i in range(3):
    point3D = triangulate_point(selected_dets1[i], selected_dets2[i], proj_matrix1, proj_matrix2)  # Triangulate initial 3D point
    print(f"Initial 3D point for bird {i+1}: {point3D.T}")
    kf = KalmanFilter3DWithVelocity(dt=1.0, std_acc=0.1, std_meas=0.1)  # Initialize Kalman filter
    kf.x[:3] = point3D  # Set the initial position in the state vector
    kalman_filters.append((kf, colors[i]))  # Add the Kalman filter and its associated color to the list

# Initialize 3D plot for visualizing trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Initialize video writer for saving the tracking result
image_dir1 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera1'
image_dir2 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera2'
output_video_path = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/output/tracking_result.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1920, 1080))

# Function to compute Euclidean distance between two points
def euclidean_distance(pointA, pointB):
    return np.linalg.norm(pointA - pointB)

# Store trajectories of the three birds
trajectories = {i: [] for i in range(3)}

# Tracking loop for subsequent frames
for frame_num in range(2, max(data1['Frame'].max(), data2['Frame'].max()) + 1):
    frame_path1 = os.path.join(image_dir1, f'frame_{frame_num:04d}.jpg')
    frame_path2 = os.path.join(image_dir2, f'frame_{frame_num:04d}.jpg')
    
    frame1 = cv2.imread(frame_path1)  # Load frame from camera 1
    frame2 = cv2.imread(frame_path2)  # Load frame from camera 2
    
    if frame1 is None or frame2 is None:
        print(f"Skipping frame {frame_num} due to loading issue.")
        continue
    
    # Get detections for the current frame from both cameras
    detections1 = data1[data1['Frame'] == frame_num][['xmin', 'ymin', 'xmax', 'ymax']].values
    detections2 = data2[data2['Frame'] == frame_num][['xmin', 'ymin', 'xmax', 'ymax']].values

    if len(detections1) == 0 or len(detections2) == 0:
        print(f"Skipping frame {frame_num} due to no detections.")
        continue
    
    unmatched_detections1 = set(range(len(detections1)))  # Initialize set of unmatched detections for camera 1
    unmatched_detections2 = set(range(len(detections2)))  # Initialize set of unmatched detections for camera 2

    # Process each Kalman filter (one for each bird)
    for i, (kf, color) in enumerate(kalman_filters):
        predicted_point3D = kf.predict()  # Predict the next position using the Kalman filter
        
        best_dist, best_det1, best_det2 = float('inf'), None, None
        best_idx1, best_idx2 = None, None

        # Match the predicted position to the closest detection
        for idx1 in unmatched_detections1:
            for idx2 in unmatched_detections2:
                det1 = detections1[idx1]
                det2 = detections2[idx2]
                current_point3D = triangulate_point(det1, det2, proj_matrix1, proj_matrix2)  # Triangulate 3D point
                
                dist = euclidean_distance(predicted_point3D, current_point3D)  # Calculate distance to predicted point
                
                if dist < best_dist:
                    best_dist = dist
                    best_det1, best_det2 = det1, det2
                    best_idx1, best_idx2 = idx1, idx2

        if best_det1 is not None and best_det2 is not None:
            new_point3D = triangulate_point(best_det1, best_det2, proj_matrix1, proj_matrix2)  # Update with the new detection
            kf.update(new_point3D)  # Update the Kalman filter with the new position
            trajectories[i].append(new_point3D.flatten())  # Save the 3D point to the trajectory
            
            # Draw bounding boxes for the matched detections
            cv2.rectangle(frame1, (int(best_det1[0]), int(best_det1[1])), (int(best_det1[2]), int(best_det1[3])), color, 2)
            cv2.rectangle(frame2, (int(best_det2[0]), int(best_det2[1])), (int(best_det2[2]), int(best_det2[3])), color, 2)
            
            unmatched_detections1.remove(best_idx1)  # Remove matched detections from the set
            unmatched_detections2.remove(best_idx2)
        else:
            print(f"No match found for Kalman filter with predicted 3D point {predicted_point3D.T}")

    # Draw unmatched detections in green
    for idx in unmatched_detections1:
        det = detections1[idx]
        cv2.rectangle(frame1, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)
    
    for idx in unmatched_detections2:
        det = detections2[idx]
        cv2.rectangle(frame2, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)

    # Combine the frames from both cameras side by side for visualization
    combined_frame = np.hstack((frame1, frame2))
    
    # Write the frame to the video
    video_writer.write(combined_frame)
    
    # Display the frame with projections
    cv2.imshow('Tracking', combined_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

# Plot 3D trajectories of the birds
for i, trajectory in trajectories.items():
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=np.array(colors[i])/255.0, label=f'Bird {i+1}')

ax.legend()
plt.show()

video_writer.release()  # Release the video writer
cv2.destroyAllWindows()  # Close all OpenCV windows
