import cv2
import os

# Parameters
input_folder = 'videos'
output_folder = 'dataset/images'
frame_rate = 1  # frames per second

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all video files in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

# Process each video file
for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frames per second of the video
    interval = int(fps / frame_rate)  # Determine interval to capture frames based on desired frame rate

    frame_count = 0
    saved_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames at the defined interval
        if frame_count % interval == 0:
            frame_name = f"{os.path.splitext(video_file)[0]}_frame_{saved_frame}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_frame += 1

        frame_count += 1

    cap.release()
    print(f"Processed {video_file}: Saved {saved_frame} frames.")

print("Finished processing all videos.")
