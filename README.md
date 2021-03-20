# Vehicle-Detection-Classification-YOLO-MobileNet

* Clone this repo
* Download yolo weights from https://pjreddie.com/darknet/yolo/ (Download TinyYOLO weights if you want to use that or else YOLO weights) For my work i have used TinyYolo as its faster and light weight.
* If you want to use TinyYOLO use command A else for YOLO command B

  A. python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolo_tiny.h5
  
  B. python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
* To test: python yolo_video.py --image
  Then provide path to any test image
* Create a seperate environment to avoid any dependency clash
  conda env create -f test\dependecies.yml car_env
  conda activate car-project-env
* pip inststall --upgrade Pillow
* python MobileNet_TransferLearning.py (Feel free to change Optimizer/Epoch or any other ML technique according to your requirement) By Default I haev used Adam optimer with lr=0.0001 for 20 Epoch
* python VideoReader.py

### Note: To know more about system please go through Vehicle_Detection_YOLO.pdf

## Working Demo:

![alt text](https://github.com/prakhargurawa/Vehicle-Detection-Classification-YOLO-MobileNet/blob/main/saved_models/Output.gif?raw=true)

## Base Structure of Mobilenet:

![alt text](https://github.com/prakhargurawa/Vehicle-Detection-Classification-YOLO-MobileNet/blob/main/saved_models/MobileNetModel.png?raw=true)

## MobileNet Transfer learning model:

![alt text](https://github.com/prakhargurawa/Vehicle-Detection-Classification-YOLO-MobileNet/blob/main/images/model.png?raw=true)

## Vision pipeline:

![alt text](https://github.com/prakhargurawa/Vehicle-Detection-Classification-YOLO-MobileNet/blob/main/images/pipeline.png?raw=true)

## Results when compared with GroundTruth 
* Use model/ScoreCalculator.py for F1 score calculation
* The results might seem low, but to be honest F1 Score wrt grouth truth is little harsh for this use case.

![alt text](https://github.com/prakhargurawa/Vehicle-Detection-Classification-YOLO-MobileNet/blob/main/saved_models/Stats_Output_Adam20Epoch.PNG?raw=true)

## Adam vs RMSProp Optimer 
* Adam (Used for this work as giving better F1 Scores and also lower overfitting)

![alt text](https://github.com/prakhargurawa/Vehicle-Detection-Classification-YOLO-MobileNet/blob/main/saved_models/Adam_20Epoch_Car.png?raw=true)

* RMSProp

![alt text](https://github.com/prakhargurawa/Vehicle-Detection-Classification-YOLO-MobileNet/blob/main/saved_models/RMSProp_20Epoch.png?raw=true)


## TODO
* Optimize pipeline for faster processing (using producer-consumer)
* Optimize transfer learning model
* Better Designing (OOPS Aspect)

