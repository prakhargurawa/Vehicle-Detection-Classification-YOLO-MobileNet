# Vehicle-Detection-Classification-YOLO-MobileNet

* Clone this repo
* Download yolo weights from https://pjreddie.com/darknet/yolo/
* python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
* To test: python yolo_video.py --image
  Then provide path to any test image
* Create a seperate environment to avoid any dependency clash
  conda env create -f test\dependecies.yml car_env
  conda activate car-project-env
* pip inststall --upgrade Pillow
* python MobileNet_TransferLearning.py
* python VideoReader.py


## TODO
* Optimize pipeline for faster processing (using producer-consumer,multithreading)
* add functionality to calculate f1score,analysis and ground truth comparision
* optimize transfer learning model
* Comments
* Better designing
* Report Generation
* Other minor/major things
