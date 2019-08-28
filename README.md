## Intuitive-Gesture-Drone-Control

Team Guardian (SFU Unmanned Aerial Veicle Team) research project in Computer Vision.

### Motivation
- Hand movement is one of the most intuitve way for us to move an object. 
- We decided to make it possible to control a drone with your hand, without requiring the user to wear any sensors. 

### Getting Started
1. Clone the project 
```
git clone https://github.com/tsa87/Intuitive-Gesture-Drone-Control.git
```
2. Install the prerequisites
```
pip install argparse
pip install numpy
pip install opencv-contrib-python
pip install scipy
pip install tensorflow
```
3. Localize the configuration path
```
# in helper/config/hand_config.py
# change the BASE_PATH var to the absolute path to your project folder.
# (optional) change parameters in the config file
```
4. Run the demo
```
# on video file
python demo.py -v PATH_TO_VIDEO
```
```
# from camera stream
python demo.py
```

### Project Demo
![](https://media.giphy.com/media/MB0S2CQ7dfTXFIbpTy/giphy.gif)

![](https://media.giphy.com/media/WoEyLRToBH8IY5WnZ7/giphy.gif)

### Improvements 
#### 1. Identify the same object in consequtive frames.
Developed an Identification system for all tracked objects. The euclidean distance of objects is calculated between two consequtive frames. Using that metric, the object in the previous frame is assigned to its closest counterpart in the following frame. 

#### 2. Detect the movement of the tracked hands.
Calculates the change in postion of the tracked object. Once the change passes a certain threshold, it is recognized as an instruction to control the drone.


### Author

* **Tony Shen** - *Initial work* - [TeamGuardian](https://github.com/Team-Guardian)

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

### Acknowledgment

* **Victor Dibia** - *Real-time Hand-Detection using Neural Networks (SSD) on Tensorflow, (2017)*  




