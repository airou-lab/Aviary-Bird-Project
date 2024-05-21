# Bird Aviary Detection

# Setup

First, clone the repository to your local machine. The HTTP should be fine

# 1). Open up the terminal in VSCode and run the command line below to activate a python work environment.
```bash
source env/bin/activate
```


# 2). Run these installation packages if not done so already:
```bash
pip install torch torchvision torchaudio  pip 
install opencv-python-headless scikit-learn matplotlib
```

# 3). Make a directory to store all the videos that you are going to run.
```bash
mkdir videos
```
# 4). Put all the videos in the newly made directory

# 5). Run Detection with YoloV5 with pre-trained model.
```bash
cd yolov5
pip install -r requirements.txt
```



Run the pretrained model:
Note: Change the name video1.mp4 to the actual video that you want to test
```bash
python detect.py --source ../videos/video1.mp4 --weights yolov5s.pt --conf 0.25 --save-txt --save-conf
```

You should be able to see results in the yolov5 folder -> runs -> detect -> exp

 

# 6). To run the fine-tuned model run the following command.

Note: Change the name video1.mp4 to the actual video that you want to test
```bash
python detect.py --weights runs/train/exp7/weights/best.pt --source ../videos/video1.mp4 --conf 0.25 --view-img --save-csv
```

You should be able to see results in the yolov5 folder -> runs -> detect -> exp2

or it will print out where the video and results are save within the terminal output


# Triangulation or 3D Reconstruction of two camera views

# 1). Make proper directories.
Make sure you are in home directory not in YOLOV5 directory
```bash
cd ..
mkdir processed_data
cd processed_data

mkdir frames
mkdir csv
```
# 2). Use Frame_Maker.py to make proper frames.
Go to the frame_maker.py file and modify these two lines of code below. Replace the video_file variable with the path to the first camera angle.
Next essentially get the path to your frames directory in the processed_data folder and add /Camera1 to the path.

Repeat this step for the second camera angle and make sure to add /Camera2 to the path. (Below is an example)

Running the script should create the frames in the respective folder.

```python
# Define the path to your video file
video_file = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/videos/Angle_2_Sep17_1_Min_Sync.mp4'

# Create a directory to save the frames
output_directory = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera2'
```

# 3). Clean CSV outputs from detection.

Now for each camera angle make sure you already have ran detection, which the detection results should have saved in a folder in yolov5/runs/detect/(name of a detection execution)/predictions.csv

We are essentially going to clean that csv and put the new csv in our new csv directory in processed_data.
Go to the cleanCSVs.py file and modify the last line of the original csv path and the new path but with the added new name of the file, which in my case I renamed it to /cleaned_output1.csv.

example of the first camera angle is below, repeat for each camera angle, since this is reconstruction on two views. we will repeat for just camera angle 2.
```python

# Usage: (1.path to csv you want to be organized, 2.new path to the new csv name and where you want it to be placed)
clean_csv('/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/yolov5/runs/detect/Angle1_Output/predictions.csv', 
          '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output1.csv')
```

# 4). Finally go to the reconstruction.py file and go to the bottom of the script and edit the setup section for the respective paths to see reconstruction result.

```python
# Setup
camera_dirs = [
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera1',
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera2'
]
csv_files = [
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output1.csv',
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output2.csv'
]
```

