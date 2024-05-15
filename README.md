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
