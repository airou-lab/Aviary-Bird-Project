import cv2
import os

# Define the path to your video file
video_file = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/videos/Angle_2_Sep17_1_Min_Sync.mp4'

# Create a directory to save the frames
output_directory = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera2'
os.makedirs(output_directory, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Set the desired frame rate (30 FPS)
desired_fps = 30

# Calculate the frame interval for capturing frames
frame_interval = max(1, round(fps / desired_fps))

# Initialize variables
frame_count = 0
extracted_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop when we reach the end of the video

    # Save the frame as an image in the output directory at the desired frame rate
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_directory, f'frame_{extracted_frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        extracted_frame_count += 1

    frame_count += 1

# Release the video capture object
cap.release()

print(f'{extracted_frame_count} frames extracted at approximately 30 frames per second and saved to {output_directory}')
